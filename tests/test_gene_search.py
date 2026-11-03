"""Gene -> candidate papers: the REST route, the subprocess adapter, and run_paper_search.py's assembly of the result."""
import json
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.paper_finder_subprocess import SubprocessPaperFinder
from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from tests.fakes.ports import FakeAuthProvider, FakePaperFinder, SyncJobRunner

TOKENS = {'alice': ('alice', 'curator'), 'bob': ('bob', 'curator'), 'viewer': ('v', 'viewer')}
AUTH = lambda t: {'Authorization': f'Bearer {t}'}
PAPERS = [{'pmid': str(24751536 + i), 'title': f'paper {i}', 'pmcid': 'PMC4003245' if i == 0 else None,
           'score': 9 - i, 'evidence_type': 'DIRECT', 'annotatable': True, 'justification': 'co-IP'}
          for i in range(8)]


class NoPipeline:
    def run(self, spec, report):
        raise AssertionError('a gene search must not start the annotation pipeline')


def client(finder=None, max_papers=10):
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), NoPipeline(),
                     max_papers=max_papers, paper_finder=finder)
    return TestClient(app)


def search(c, body, token='alice'):
    return c.post('/api/llm/papers/by-gene', json=body, headers=AUTH(token))


# ── route ──────────────────────────────────────────────────────────────────

def test_search_runs_as_a_job_and_the_job_holds_the_papers():
    finder = FakePaperFinder(PAPERS)
    c = client(finder)
    r = search(c, {'gene': ' pink1 ', 'max_papers': 3})
    assert r.status_code == 202
    assert finder.calls == [('PINK1', 3)]                 # trimmed and upper-cased
    j = c.get(f'/api/llm/jobs/{r.json()["jobId"]}', headers=AUTH('alice')).json()
    assert j['status'] == 'done' and j['progress'] == 'searching PubMed'
    res = j['result']
    assert res['gene'] == 'PINK1' and res['pool_size'] == 120 and res['candidate_pool'] == 50
    assert [p['pmid'] for p in res['papers']] == ['24751536', '24751537', '24751538']
    assert res['papers'][0]['pmcid'] == 'PMC4003245' and res['papers'][1]['pmcid'] is None


def test_max_papers_is_capped_by_the_session_limit():
    finder = FakePaperFinder(PAPERS)
    c = client(finder, max_papers=4)
    search(c, {'gene': 'PINK1', 'max_papers': 50})
    search(c, {'gene': 'PINK1', 'max_papers': 0})
    assert finder.calls == [('PINK1', 4), ('PINK1', 1)]


def test_default_is_five_papers():
    finder = FakePaperFinder(PAPERS)
    search(client(finder), {'gene': 'PINK1'})
    assert finder.calls == [('PINK1', 5)]


@pytest.mark.parametrize('gene', ['', '   ', 'PINK1 OR BRCA1', 'PINK1;rm', '-PINK1', 'A' * 40, '"PINK1"'])
def test_anything_but_a_gene_symbol_is_422(gene):
    finder = FakePaperFinder(PAPERS)
    assert search(client(finder), {'gene': gene}).status_code == 422
    assert finder.calls == []


@pytest.mark.parametrize('gene', ['HLA-A', 'NKX2-1', 'C9orf72', 'MT-ND1'])
def test_real_symbols_with_dashes_and_lower_case_are_accepted(gene):
    finder = FakePaperFinder(PAPERS)
    assert search(client(finder), {'gene': gene}).status_code == 202
    assert finder.calls[0][0] == gene.upper()


def test_not_configured_is_503():
    assert search(client(None), {'gene': 'PINK1'}).status_code == 503


def test_needs_a_curator():
    c = client(FakePaperFinder(PAPERS))
    assert c.post('/api/llm/papers/by-gene', json={'gene': 'PINK1'}).status_code == 401
    assert search(c, {'gene': 'PINK1'}, token='viewer').status_code == 403


def test_a_failed_search_is_a_failed_job():
    c = client(FakePaperFinder(fail=RuntimeError('PubMed is down')))
    jid = search(c, {'gene': 'PINK1'}).json()['jobId']
    j = c.get(f'/api/llm/jobs/{jid}', headers=AUTH('alice')).json()
    assert j['status'] == 'failed' and 'PubMed is down' in j['error'] and j['result'] is None


def test_another_curator_cannot_see_the_job():
    c = client(FakePaperFinder(PAPERS))
    jid = search(c, {'gene': 'PINK1'}).json()['jobId']
    assert c.get(f'/api/llm/jobs/{jid}', headers=AUTH('bob')).status_code == 404


# ── subprocess adapter ─────────────────────────────────────────────────────

def fake_run(result=None, code=0, lines=()):
    calls = []

    def run(cmd, cwd, timeout, on_line=None):
        calls.append(cmd)
        for ln in lines:
            on_line and on_line(ln)
        if result is not None:
            with open(cmd[cmd.index('--out') + 1], 'w') as f:
                json.dump(result, f)
        return subprocess.CompletedProcess(cmd, code, '\n'.join(lines), '')
    return run, calls


def test_adapter_runs_the_script_and_reads_its_result():
    run, calls = fake_run({'gene': 'PINK1', 'pool_size': 7, 'candidate_pool': 5, 'papers': PAPERS[:2]},
                          lines=['INFO - noise', '[progress] searching PubMed', '[progress] judging 5 candidate papers'])
    seen = []
    res = SubprocessPaperFinder(root='/repo', python='/py', run=run).find('PINK1', 2, seen.append)
    assert calls[0][:3] == ['/py', '/repo/run_paper_search.py', 'PINK1']
    assert calls[0][3:5] == ['--max-papers', '2']
    assert seen == ['searching PubMed', 'judging 5 candidate papers']     # only progress lines, prefix removed
    assert [p.pmid for p in res.papers] == ['24751536', '24751537'] and res.papers[0].score == 9


def test_adapter_raises_with_the_output_tail_on_failure():
    run, _ = fake_run(code=1, lines=['[progress] searching PubMed', 'RuntimeError: the judge scored none'])
    with pytest.raises(RuntimeError, match='judge scored none'):
        SubprocessPaperFinder(run=run).find('PINK1')


def test_adapter_reports_a_timeout():
    def run(cmd, cwd, timeout, on_line=None):
        raise subprocess.TimeoutExpired(cmd, timeout)
    with pytest.raises(RuntimeError, match='longer than 5s'):
        SubprocessPaperFinder(run=run, timeout=5).find('PINK1')


def test_adapter_defaults_to_the_service_python(monkeypatch):
    monkeypatch.delenv('GENE_SEARCH_PYTHON', raising=False)
    assert SubprocessPaperFinder().python == sys.executable
    monkeypatch.setenv('GENE_SEARCH_PYTHON', '/opt/full/bin/python')
    assert SubprocessPaperFinder().python == '/opt/full/bin/python'


# ── run_paper_search.py ────────────────────────────────────────────────────

@pytest.fixture
def script(monkeypatch):
    """run_paper_search with the extractor and NCBI calls faked."""
    import run_paper_search as rps
    state = {}

    class FakeExtractor:
        def extract(self, gene, max_papers, report):
            report('searching PubMed')
            return state['result']
    monkeypatch.setitem(sys.modules, 'ReactomeLiteratureExtractor',
                        types.SimpleNamespace(ReactomeLiteratureExtractor=FakeExtractor))
    monkeypatch.setattr(rps, 'paper_metadata', lambda pmids: {p: {'title': f't{p}', 'journal': 'J', 'year': '2014'}
                                                              for p in pmids})
    monkeypatch.setattr(rps, 'pmc_ids', lambda pmids: {pmids[0]: 'PMC1'} if pmids else {})
    return rps, state


def scored(*rows):
    return {'per_paper': [{'pmid': p, 'score': s, 'evidence_type': 'DIRECT', 'annotatable': True,
                           'justification': f'why {p}'} for p, s in rows]}


def test_script_joins_selection_verdicts_and_metadata(script, capsys):
    rps, state = script
    state['result'] = {'gene': 'PINK1', 'accession': 'Q9BXM7', 'pool_size': 200, 'candidate_pool': 3,
                       'papers': [{'pmid': '2'}, {'pmid': '1'}], 'scored': scored(('1', 7), ('2', 9), ('3', 2)),
                       'dropped_below_threshold': 1}
    res = rps.search('PINK1', 5)
    assert [p['pmid'] for p in res['papers']] == ['2', '1']                # the judge's order, not the pool's
    assert res['papers'][0] == {'pmid': '2', 'title': 't2', 'journal': 'J', 'year': '2014', 'pmcid': 'PMC1',
                                'score': 9, 'evidence_type': 'DIRECT', 'annotatable': True, 'justification': 'why 2'}
    assert res['papers'][1]['pmcid'] is None
    assert res['accession'] == 'Q9BXM7' and res['dropped_below_threshold'] == 1
    assert '[progress] searching PubMed' in capsys.readouterr().out


def test_script_fails_when_the_judge_scored_nothing(script):
    rps, state = script
    state['result'] = {'gene': 'PINK1', 'pool_size': 200, 'candidate_pool': 2, 'papers': [],
                       'scored': scored(('1', None), ('2', None)), 'dropped_below_threshold': 2}
    with pytest.raises(RuntimeError, match='scored none of the 2'):
        rps.search('PINK1', 5)


def test_script_no_candidates_is_an_empty_result_not_a_failure(script):
    rps, state = script
    state['result'] = {'gene': 'XYZ1', 'pool_size': 0, 'candidate_pool': 0, 'papers': [], 'scored': {},
                       'dropped_below_threshold': 0}
    assert rps.search('XYZ1', 5)['papers'] == []


def test_script_writes_the_result_for_the_adapter(script, tmp_path):
    rps, state = script
    state['result'] = {'gene': 'PINK1', 'pool_size': 5, 'candidate_pool': 1, 'papers': [{'pmid': '1'}],
                       'scored': scored(('1', 8)), 'dropped_below_threshold': 0}
    out = tmp_path / 'r.json'
    rps.main(['pink1', '--out', str(out)])
    from curator_llm.ports.paper_finder import GeneSearchResult
    assert GeneSearchResult.model_validate(json.loads(out.read_text())).papers[0].score == 8
