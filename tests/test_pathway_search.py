"""Gene -> Reactome pathways (Gene2Path): the REST route, the subprocess adapter, and run_pathway_search.py's assembly."""
import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.pathway_finder_subprocess import SubprocessPathwayFinder
from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from tests.fakes.ports import FakeAuthProvider, FakePathwayFinder, SyncJobRunner

TOKENS = {'alice': ('alice', 'curator'), 'bob': ('bob', 'curator'), 'viewer': ('v', 'viewer')}
AUTH = lambda t: {'Authorization': f'Bearer {t}'}
RESULT = {'annotated_pathways': [{'pathway_id': 5205647, 'pathway_name': 'Mitophagy'}],
          'status': 'ok', 'confident': True,
          'predicted_pathways': [{'pathway_id': 9013419, 'pathway_name': 'RHOT1 GTPase cycle', 'fdr': 6.2e-4,
                                  'overlap_count': 2, 'mapped_genes': ['RHOT1', 'RHOT2'], 'evidence_count': 41,
                                  'evidence': [{'pmid': '22078885', 'title': 'PINK1 and Parkin target Miro',
                                                'year': '2011', 'pmcid': 'PMC3261796',
                                                'partners_mentioned': ['RHOT1'], 'terms_mentioned': ['GTPase'],
                                                'sentences': ['PINK1 phosphorylates Miro (RHOT1).']}]},
                                 {'pathway_id': 6798695, 'pathway_name': 'Neutrophil degranulation', 'fdr': 4.9e-3,
                                  'overlap_count': 2, 'mapped_genes': ['SQSTM1', 'ULK1'],
                                  'evidence_error': 'HTTPError: 429'}],
          'partners': [{'gene': 'PRKN', 'score': 0.99}, {'gene': 'RHOT1', 'score': 0.97}]}


class NoPipeline:
    def run(self, spec, report):
        raise AssertionError('a pathway search must not start the annotation pipeline')


def client(finder=None):
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), NoPipeline(),
                     pathway_finder=finder)
    return TestClient(app)


def search(c, body, token='alice'):
    return c.post('/api/llm/pathways/by-gene', json=body, headers=AUTH(token))


# ── route ──────────────────────────────────────────────────────────────────

def test_search_runs_as_a_job_and_the_job_holds_the_pathways():
    finder = FakePathwayFinder(RESULT)
    c = client(finder)
    r = search(c, {'gene': ' pink1 '})
    assert r.status_code == 202
    assert finder.calls == [('PINK1', 0.8, 10, 0.05, 8, 5, True)]  # trimmed, upper-cased, defaults
    j = c.get(f'/api/llm/jobs/{r.json()["jobId"]}', headers=AUTH('alice')).json()
    assert j['status'] == 'done' and j['progress'] == 'ranking its functional-interaction partners'
    res = j['result']
    assert res['gene'] == 'PINK1' and res['confident'] is True
    assert [p['pathway_name'] for p in res['annotated_pathways']] == ['Mitophagy']
    miro, neutro = res['predicted_pathways']
    assert miro['pathway_name'] == 'RHOT1 GTPase cycle' and miro['evidence_count'] == 41
    assert miro['evidence'][0]['pmid'] == '22078885' and miro['evidence'][0]['pmcid'] == 'PMC3261796'
    assert miro['evidence'][0]['sentences'] == ['PINK1 phosphorylates Miro (RHOT1).']
    assert neutro['evidence'] == [] and neutro['evidence_error'] == 'HTTPError: 429'
    assert res['partners'][0] == {'gene': 'PRKN', 'score': 0.99}


def test_settings_are_passed_through():
    finder = FakePathwayFinder(RESULT)
    search(client(finder), {'gene': 'PINK1', 'fi_cutoff': 0.6, 'top_partners': 20, 'fdr_cutoff': 0.01,
                            'max_pathways': 3, 'max_evidence': 0, 'summarize': False})
    assert finder.calls == [('PINK1', 0.6, 20, 0.01, 3, 0, False)]


@pytest.mark.parametrize('body', [{'fi_cutoff': 1}, {'fi_cutoff': -0.1}, {'top_partners': 0}, {'top_partners': 51},
                                  {'fdr_cutoff': 0}, {'fdr_cutoff': 2}, {'max_pathways': 0},
                                  {'max_evidence': -1}, {'max_evidence': 21}])
def test_out_of_range_settings_are_422(body):
    finder = FakePathwayFinder(RESULT)
    assert search(client(finder), {'gene': 'PINK1', **body}).status_code == 422
    assert finder.calls == []


@pytest.mark.parametrize('gene', ['', '   ', 'PINK1 OR BRCA1', 'PINK1;rm', '-PINK1', 'A' * 40])
def test_anything_but_a_gene_symbol_is_422(gene):
    finder = FakePathwayFinder(RESULT)
    assert search(client(finder), {'gene': gene}).status_code == 422
    assert finder.calls == []


def test_no_partners_is_a_result_not_a_failure():
    c = client(FakePathwayFinder({'status': 'no_partners'}))
    jid = search(c, {'gene': 'TANC1'}).json()['jobId']
    j = c.get(f'/api/llm/jobs/{jid}', headers=AUTH('alice')).json()
    assert j['status'] == 'done' and j['result']['status'] == 'no_partners' and j['result']['predicted_pathways'] == []


def test_not_configured_is_503():
    assert search(client(None), {'gene': 'PINK1'}).status_code == 503


def test_needs_a_curator():
    c = client(FakePathwayFinder(RESULT))
    assert c.post('/api/llm/pathways/by-gene', json={'gene': 'PINK1'}).status_code == 401
    assert search(c, {'gene': 'PINK1'}, token='viewer').status_code == 403


def test_a_failed_search_is_a_failed_job_and_private():
    c = client(FakePathwayFinder(fail=RuntimeError('Neo4j is down')))
    jid = search(c, {'gene': 'PINK1'}).json()['jobId']
    j = c.get(f'/api/llm/jobs/{jid}', headers=AUTH('alice')).json()
    assert j['status'] == 'failed' and 'Neo4j is down' in j['error'] and j['result'] is None
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


def test_adapter_runs_the_script_with_the_settings_and_reads_its_result():
    run, calls = fake_run({'gene': 'PINK1', **RESULT},
                          lines=['INFO - noise', '[progress] looking up the pathways it is annotated in'])
    seen = []
    res = SubprocessPathwayFinder(root='/repo', python='/py', run=run).find('PINK1', 0.6, 20, 0.01, 3, 2,
                                                                           report=seen.append)
    assert calls[0][:3] == ['/py', '/repo/run_pathway_search.py', 'PINK1']
    assert calls[0][3:13] == ['--fi-cutoff', '0.6', '--top-partners', '20', '--fdr-cutoff', '0.01',
                              '--max-pathways', '3', '--max-evidence', '2']
    assert '--no-summaries' not in calls[0]                            # summaries are on by default
    assert seen == ['looking up the pathways it is annotated in']     # only progress lines, prefix removed
    assert res.predicted_pathways[0].evidence[0].partners_mentioned == ['RHOT1'] and res.partners[0].gene == 'PRKN'


def test_adapter_passes_no_summaries():
    run, calls = fake_run({'gene': 'PINK1', **RESULT})
    SubprocessPathwayFinder(run=run).find('PINK1', summarize=False)
    assert calls[0][-1] == '--no-summaries'


def test_adapter_raises_with_the_output_tail_on_failure():
    run, _ = fake_run(code=1, lines=['[progress] ranking', 'ServiceUnavailable: Neo4j is down'])
    with pytest.raises(RuntimeError, match='Neo4j is down'):
        SubprocessPathwayFinder(run=run).find('PINK1')


def test_adapter_reports_a_timeout():
    def run(cmd, cwd, timeout, on_line=None):
        raise subprocess.TimeoutExpired(cmd, timeout)
    with pytest.raises(RuntimeError, match='longer than 5s'):
        SubprocessPathwayFinder(run=run, timeout=5).find('PINK1')


def test_adapter_uses_the_gene_search_python(monkeypatch):
    monkeypatch.delenv('GENE_SEARCH_PYTHON', raising=False)
    assert SubprocessPathwayFinder().python == sys.executable
    monkeypatch.setenv('GENE_SEARCH_PYTHON', '/opt/full/bin/python')
    assert SubprocessPathwayFinder().python == '/opt/full/bin/python'


# ── run_pathway_search.py ──────────────────────────────────────────────────

@pytest.fixture
def script(monkeypatch):
    """run_pathway_search with its Neo4j, FI, enrichment and PubMed lookups faked. state['papers'] is
    {pmid: {title, journal, year, abstract}}; every evidence search returns all of them, in that order.
    The summary model is a FakeSummaryModel: state['prompts'] holds what it was asked, state['reply'] what it says."""
    import run_pathway_search as rps
    state = {'annotated': [], 'placement': ({'status': 'ok', 'partners_used': []}, False), 'enriched': [],
             'papers': {}, 'queries': [], 'prompts': [], 'reply': 'A summary.'}
    monkeypatch.setattr(rps, 'summary_model', lambda: FakeSummaryModel(state))

    def pubmed_search(query):
        state['queries'].append(query)
        if state.get('search_fails'):
            raise RuntimeError('esearch 429')
        return len(state['papers']) * 10, list(state['papers'])
    monkeypatch.setattr(rps, 'pubmed_search', pubmed_search)
    monkeypatch.setattr(rps, 'pubmed_articles', lambda pmids: {p: state['papers'][p] for p in pmids})
    monkeypatch.setattr(rps, 'pmc_ids', lambda pmids: {p: 'PMC' + p for p in pmids if p == '1'})
    monkeypatch.setattr(rps, 'annotated_pathways', lambda gene: state['annotated'])
    monkeypatch.setattr(rps, 'placement', lambda *a: state['placement'])

    def enriched_pathways(partners, fdr):
        state['asked'] = partners
        return state['enriched']
    monkeypatch.setattr(rps, 'enriched_pathways', enriched_pathways)
    return rps, state


class FakeSummaryModel:
    def __init__(self, state):
        self.state = state

    def invoke(self, prompt):
        self.state['prompts'].append(prompt)
        reply = self.state['reply']
        if isinstance(reply, Exception):
            raise reply
        return type('Reply', (), {'content': reply})()


def enriched(*names):
    return [{'pathway_id': i, 'pathway_name': n, 'fdr': 10.0 ** -(9 - i), 'overlap_count': 2,
             'mapped_genes': ['A', 'B']} for i, n in enumerate(names)]


def test_script_leaves_out_the_pathways_the_gene_is_already_in_then_caps_them(script, capsys):
    rps, state = script
    state['annotated'] = [{'pathway_id': 1, 'pathway_name': 'Mitophagy'}]
    state['placement'] = ({'status': 'ok', 'partners_used': [{'gene': 'PRKN', 'score': 0.99},
                                                            {'gene': 'RHOT1', 'score': 0.97}]}, True)
    state['enriched'] = enriched('Miro GTPase Cycle', 'Mitophagy', 'Autophagy', 'Macroautophagy', 'Aggrephagy')
    res = rps.search('PINK1', max_pathways=3)
    assert state['asked'] == ['PRKN', 'RHOT1']
    assert [(p['pathway_name'], p['annotated']) for p in res['predicted_pathways']] == [
        ('Miro GTPase Cycle', False), ('Autophagy', False), ('Macroautophagy', False)]
    assert len(state['queries']) == 3                                 # literature only for the pathways listed
    assert res['confident'] is True and res['status'] == 'ok'
    assert res['partners'] == [{'gene': 'PRKN', 'score': 0.99}, {'gene': 'RHOT1', 'score': 0.97}]
    assert '[progress] testing pathway enrichment over 2 partners' in capsys.readouterr().out


def test_script_no_partners_still_lists_the_annotated_pathways(script):
    rps, state = script
    state['annotated'] = [{'pathway_id': 1, 'pathway_name': 'Mitophagy'}]
    state['placement'] = ({'status': 'no_partners', 'partners_used': []}, False)
    res = rps.search('TANC1')
    assert res['status'] == 'no_partners' and res['predicted_pathways'] == [] and len(res['annotated_pathways']) == 1
    assert 'asked' not in state                                       # no enrichment without partners


def test_script_writes_the_result_for_the_adapter(script, tmp_path):
    rps, state = script
    state['placement'] = ({'status': 'ok', 'partners_used': [{'gene': 'PRKN', 'score': 0.9}]}, False)
    state['enriched'] = enriched('Mitophagy')
    out = tmp_path / 'r.json'
    rps.main(['pink1', '--fdr-cutoff', '0.01', '--out', str(out)])
    from curator_llm.ports.pathway_finder import GenePathwaysResult
    res = GenePathwaysResult.model_validate(json.loads(out.read_text()))
    assert res.gene == 'PINK1' and res.predicted_pathways[0].pathway_name == 'Mitophagy'
    assert res.params['fdr_cutoff'] == 0.01


# ── literature evidence ───────────────────────────────────────────────────

def predicted(name='PINK1-PRKN Mediated Mitophagy', genes=('PRKN', 'RHOT1')):
    return {'status': 'ok', 'partners_used': [{'gene': g, 'score': 0.9} for g in genes]}, True


def test_pathway_terms_drop_generic_words_the_gene_and_its_partners():
    import run_pathway_search as rps
    assert rps.pathway_term_groups('PINK1-PRKN Mediated Mitophagy', 'PINK1', ['PRKN']) == [['PINK1-PRKN', 'Mitophagy']]
    assert rps.pathway_term_groups('Signaling by ROBO receptors', 'SLIT2', []) == [['ROBO']]
    assert rps.pathway_term_groups('RHOT1 GTPase cycle', 'PINK1', ['RHOT1', 'RHOT2']) == [['GTPase']]
    assert rps.pathway_term_groups('Mitochondrial unfolded protein response (UPRmt)', 'PINK1', []) == [
        ['Mitochondrial', 'unfolded'], ['UPRmt']]
    assert rps.pathway_term_groups('Processing of SMDT1', 'PINK1', ['SMDT1']) == []


def test_evidence_query_needs_the_gene_and_any_partner_or_every_word_of_a_term_group():
    import run_pathway_search as rps
    assert rps.evidence_query('PINK1', ['RHOT1'], [['GTPase']]) == \
        '"PINK1"[tiab] AND ("RHOT1"[tiab] OR "GTPase"[tiab])'
    assert rps.evidence_query('PINK1', ['PARL'], [['Mitochondrial', 'calcium'], ['UPRmt']]) == \
        '"PINK1"[tiab] AND ("PARL"[tiab] OR ("Mitochondrial"[tiab] AND "calcium"[tiab]) OR "UPRmt"[tiab])'
    assert rps.evidence_query('PINK1', [], []) == '"PINK1"[tiab]'


def test_supporting_sentences_name_the_gene_with_a_partner_or_term_most_first():
    import run_pathway_search as rps
    abstract = ('Mitochondria move along microtubules. PINK1 is a kinase. PINK1 binds Miro in mitophagy. '
                'PINK1 phosphorylates RHOT1 and RHOT2 during mitophagy. RHOT1 is a GTPase. '
                'The PINK1A allele does not count with RHOT1.')
    assert rps.supporting_sentences(abstract, 'PINK1', ['RHOT1', 'RHOT2'], ['mitophagy']) == [
        'PINK1 phosphorylates RHOT1 and RHOT2 during mitophagy.', 'PINK1 binds Miro in mitophagy.']


def test_script_lists_each_pathways_papers_partner_co_mentions_first(script):
    rps, state = script
    state['placement'] = predicted()
    state['enriched'] = [{'pathway_id': 7, 'pathway_name': 'RHOT1 GTPase cycle', 'fdr': 1e-4, 'overlap_count': 2,
                          'mapped_genes': ['RHOT1', 'RHOT2']}]
    state['papers'] = {
        '3': {'title': 'PINK1 in GTPase signalling', 'journal': 'J', 'year': '2015',
              'abstract': 'PINK1 regulates a GTPase.'},
        '1': {'title': 'PINK1 and Parkin target Miro', 'journal': 'Cell', 'year': '2011',
              'abstract': 'Here PINK1 phosphorylates RHOT1. RHOT2 is degraded too, with PINK1.'},
        '2': {'title': 'PINK1 review', 'journal': 'R', 'year': '2020', 'abstract': 'Nothing on the pathway.'}}
    res = rps.search('PINK1', max_evidence=2)
    p = res['predicted_pathways'][0]
    assert state['queries'] == ['"PINK1"[tiab] AND ("RHOT1"[tiab] OR "RHOT2"[tiab] OR "GTPase"[tiab])']
    assert p['evidence_count'] == 30 and [e['pmid'] for e in p['evidence']] == ['1', '3']   # capped at 2
    first = p['evidence'][0]
    assert first == {'pmid': '1', 'title': 'PINK1 and Parkin target Miro', 'journal': 'Cell', 'year': '2011',
                     'pmcid': 'PMC1', 'partners_mentioned': ['RHOT1', 'RHOT2'], 'terms_mentioned': [],
                     'sentences': ['Here PINK1 phosphorylates RHOT1.', 'RHOT2 is degraded too, with PINK1.']}
    assert p['evidence'][1]['pmcid'] is None and p['evidence'][1]['terms_mentioned'] == ['GTPase']
    from curator_llm.ports.pathway_finder import GenePathwaysResult
    assert GenePathwaysResult.model_validate(res).predicted_pathways[0].evidence[0].pmcid == 'PMC1'


def test_a_failed_literature_lookup_is_reported_on_the_pathway_not_the_job(script):
    rps, state = script
    state['placement'] = predicted()
    state['enriched'] = enriched('Mitophagy')
    state['search_fails'] = True
    p = rps.search('PINK1')['predicted_pathways'][0]
    assert p['evidence'] == [] and p['evidence_count'] == 0 and 'esearch 429' in p['evidence_error']


def test_max_evidence_0_skips_the_literature(script):
    rps, state = script
    state['placement'] = predicted()
    state['enriched'] = enriched('Mitophagy')
    p = rps.search('PINK1', max_evidence=0)['predicted_pathways'][0]
    assert state['queries'] == [] and 'evidence' not in p


# ── summaries ─────────────────────────────────────────────────────────────

PAPERS = {'1': {'title': 'PINK1 and Parkin target Miro', 'journal': 'Cell', 'year': '2011',
                'abstract': 'PINK1 phosphorylates RHOT1 at S156.'},
          '2': {'title': 'PINK1 and RHOT2', 'journal': 'J', 'year': '2015', 'abstract': 'PINK1 binds RHOT2.'}}


def miro(state):
    state['placement'] = predicted()
    state['enriched'] = [{'pathway_id': 7, 'pathway_name': 'RHOT1 GTPase cycle', 'fdr': 1e-4, 'overlap_count': 2,
                          'mapped_genes': ['RHOT1', 'RHOT2']}]
    state['papers'] = dict(PAPERS)


def test_each_pathway_with_papers_is_summarized_from_its_abstracts(script):
    rps, state = script
    miro(state)
    state['reply'] = 'PINK1 phosphorylates RHOT1 [PMID:1] and binds RHOT2 [PMID:2].'
    p = rps.search('PINK1')['predicted_pathways'][0]
    assert p['summary'] == 'PINK1 phosphorylates RHOT1 [PMID:1] and binds RHOT2 [PMID:2].' and 'summary_error' not in p
    prompt, = state['prompts']
    assert '"RHOT1 GTPase cycle"' in prompt and 'RHOT1, RHOT2' in prompt
    assert '[PMID:1] PINK1 and Parkin target Miro (Cell, 2011)\nPINK1 phosphorylates RHOT1 at S156.' in prompt
    assert 'abstract' not in p['evidence'][0]                          # the abstracts stay out of the result
    from curator_llm.ports.pathway_finder import GenePathwaysResult
    assert GenePathwaysResult.model_validate(rps.search('PINK1')).predicted_pathways[0].summary.startswith('PINK1')


def test_citations_of_papers_the_pathway_does_not_list_are_removed(script):
    rps, state = script
    miro(state)
    state['reply'] = [{'type': 'text', 'text': 'PINK1 acts on Miro [PMID:1][PMID: 99]\n\nin neurons [PMID:42].'}]
    assert rps.search('PINK1')['predicted_pathways'][0]['summary'] == 'PINK1 acts on Miro [PMID:1] in neurons.'


def test_pathways_without_papers_are_not_summarized(script):
    rps, state = script
    state['placement'] = predicted()
    state['enriched'] = enriched('Mitophagy')
    p = rps.search('PINK1')['predicted_pathways'][0]
    assert p['evidence'] == [] and state['prompts'] == [] and 'summary' not in p and 'summary_error' not in p


def test_a_failed_summary_is_reported_on_the_pathway_not_the_job(script):
    rps, state = script
    miro(state)
    state['reply'] = RuntimeError('overloaded')
    res = rps.search('PINK1')
    p = res['predicted_pathways'][0]
    assert 'summary' not in p and 'overloaded' in p['summary_error'] and p['evidence']


def test_no_model_is_reported_on_every_pathway(script, monkeypatch):
    rps, state = script
    miro(state)
    def no_key():
        raise ValueError('ANTHROPIC_API_KEY is not set')
    monkeypatch.setattr(rps, 'summary_model', no_key)
    assert 'ANTHROPIC_API_KEY' in rps.search('PINK1')['predicted_pathways'][0]['summary_error']


def test_summaries_can_be_turned_off(script, tmp_path):
    rps, state = script
    miro(state)
    out = tmp_path / 'r.json'
    rps.main(['PINK1', '--no-summaries', '--out', str(out)])
    res = json.loads(out.read_text())
    assert state['prompts'] == [] and 'summary' not in res['predicted_pathways'][0]
    assert res['params']['summarize'] is False and res['predicted_pathways'][0]['evidence']
