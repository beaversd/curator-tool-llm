"""Gene2Path -> Paper2Path: a session for a gene and one FI partner, placed in a predicted pathway."""
import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, DefinedSetSpec, EwasSpec, PathwaySpec, ReactionSpec,
                                         ReactomeDraft, RegulationSpec, SimpleEntitySpec)
from curator_llm.ports.extractor import ExtractionResult
from curator_llm.ports.pipeline import PaperRef, PipelineInput
from curator_llm.services.partner_link import entities_of, link_partner, place_in_pathway
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.snapshots import snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, FakeUniProt, SyncJobRunner
from tests.test_pipeline_default import FakeExtractor, Model, reactions
from tests.test_session_api import AUTH, TOKENS, FakePipeline

MITOPHAGY = {'dbId': 5205685, 'displayName': 'PINK1-PRKN Mediated Mitophagy', 'schemaClassName': 'Pathway'}


def draft():
    d = ReactomeDraft()
    d.participants = {
        'pink1': EwasSpec(key='pink1', name='PINK1 [mitochondrial outer membrane]'),
        'parkin': EwasSpec(key='parkin', name='Parkin', uniprot='O60260'),
        'p_prkn': EwasSpec(key='p_prkn', name='p-S65-PRKN'),
        'ub': EwasSpec(key='ub', name='UB'),
        'atp': SimpleEntitySpec(key='atp', name='ATP'),
        'pink1a': EwasSpec(key='pink1a', name='PINK1A'),
        'cx': ComplexSpec(key='cx', name='TOM complex', components=['set']),
        'set': DefinedSetSpec(key='set', name='PINK1 forms', members=['pink1'])}
    d.reactions = [
        ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin', inputs=['ub', 'atp'], outputs=['ub'],
                     catalyst=CatalystSpec(entity='pink1')),
        ReactionSpec(key='r1', name='PINK1 phosphorylates Parkin', inputs=['parkin'], outputs=['p_prkn'],
                     catalyst=CatalystSpec(entity='cx')),
        ReactionSpec(key='r2', name='PINK1A binds ubiquitin', inputs=['pink1a', 'ub'], outputs=['ub']),
        ReactionSpec(key='r3', name='Ubiquitin activates Parkin', inputs=['parkin'], outputs=['parkin'],
                     regulations=[RegulationSpec(kind='positive', regulator='pink1')])]
    d.pathway = PathwaySpec(name='PINK1 candidate pathway', reactions=['r0', 'r1', 'r2', 'r3'])
    return d


def test_a_gene_is_found_by_name_by_accession_and_inside_complexes_and_sets():
    d = draft()
    assert entities_of(d, 'PINK1') == {'pink1', 'set', 'cx'}            # not PINK1A
    assert entities_of(d, 'PRKN') == {'p_prkn'}
    assert entities_of(d, 'PRKN', 'O60260') == {'p_prkn', 'parkin'}


def test_reactions_involving_both_come_first_with_an_issue_each():
    d = draft()
    issues = link_partner(d, 'PINK1', 'PRKN', {'PRKN': 'O60260'})
    assert [r.key for r in d.reactions] == ['r1', 'r3', 'r0', 'r2']
    assert d.pathway.reactions == ['r1', 'r3', 'r0', 'r2']
    assert [(i.code, i.severity, i.reaction_key) for i in issues] == [('partner_link', 'info', 'r1'),
                                                                     ('partner_link', 'info', 'r3')]
    assert 'involves both PINK1 and PRKN' in issues[0].message


def test_no_reaction_with_both_is_a_warning_and_changes_nothing():
    d = draft()
    issues = link_partner(d, 'PINK1', 'RHOT1')
    assert [r.key for r in d.reactions] == ['r0', 'r1', 'r2', 'r3']
    assert [(i.code, i.severity) for i in issues] == [('partner_link:none', 'warning')]


def test_the_draft_is_placed_in_the_predicted_pathway():
    d = draft()
    assert place_in_pathway(d, 5205685, FakeInstanceLookup([MITOPHAGY])) == []
    assert d.pathway.existing.db_id == 5205685 and d.pathway.name == 'PINK1-PRKN Mediated Mitophagy'
    assert d.pathway.reactions == ['r0', 'r1', 'r2', 'r3']


@pytest.mark.parametrize('instances, why', [([], 'not found'),
                                            ([{'dbId': 5, 'displayName': 'x', 'schemaClassName': 'Reaction'}],
                                             'a Reaction, not a Pathway')])
def test_a_pathway_id_that_is_not_a_pathway_keeps_the_proposed_one(instances, why):
    d = draft()
    issue, = place_in_pathway(d, 5, FakeInstanceLookup(instances))
    assert issue.code == 'pathway_placement' and why in issue.message
    assert d.pathway.existing is None and d.pathway.name == 'PINK1 candidate pathway'


def test_the_pipeline_links_the_partner_and_places_the_draft(monkeypatch):
    monkeypatch.setattr('curator_llm.services.pipeline_default.load_paper_text', lambda src: None)
    pipe = DefaultPipeline(FakeInstanceLookup([MITOPHAGY]), FakeUniProt({}, {'PINK1': 'Q9BXM7'}), None, model=Model(),
                           extractor=FakeExtractor(ExtractionResult(ok=True, reactions=reactions())))
    res = pipe.run(PipelineInput(pmid='24751536', focus='PINK1', partner='UB', pathway_id=5205685), lambda m: None)
    assert res.draft.pathway.existing.db_id == 5205685
    # the fake draft's one reaction (ubiquitin in, ubiquitin out) has no PINK1 in it
    assert [i.code for i in res.issues if i.code.startswith('partner_link')] == ['partner_link:none']
    plain = pipe.run(PipelineInput(pmid='24751536', focus='PINK1'), lambda m: None)
    assert plain.draft.pathway.existing is None and not any(i.code.startswith('partner_link') for i in plain.issues)


def test_the_partner_and_pathway_go_into_the_snapshot_key():
    refs = [PaperRef(pmid='1')]
    assert snapshot_key(PipelineInput(papers=refs, focus='PINK1')) == 'pmid-1__PINK1'
    assert snapshot_key(PipelineInput(papers=refs, focus='PINK1', partner='prkn', pathway_id=7)) == 'pmid-1__PINK1__PRKN__pw7'


# ── REST ───────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    pipe = FakePipeline()
    return TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), pipe)), pipe


def test_a_session_for_a_partners_papers_carries_the_partner_and_pathway(env):
    c, pipe = env
    r = c.post('/api/llm/sessions', json={'pmids': ['24751536', '29168502'], 'focus': 'PINK1', 'partner': ' prkn ',
                                          'pathway_id': 5205685}, headers=AUTH('alice'))
    assert r.status_code == 202
    spec = pipe.calls[0]
    assert [p.pmid for p in spec.papers] == ['24751536', '29168502']
    assert (spec.focus, spec.partner, spec.pathway_id) == ('PINK1', 'PRKN', 5205685)
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert (s['focus'], s['partner'], s['pathway_id']) == ('PINK1', 'PRKN', 5205685)


@pytest.mark.parametrize('body', [{'partner': 'PRKN'}, {'focus': 'PINK1', 'partner': 'PRKN OR X'},
                                  {'focus': 'PINK1', 'pathway_id': 0}])
def test_a_partner_needs_a_focus_and_a_gene_symbol(env, body):
    c, pipe = env
    assert c.post('/api/llm/sessions', json={'pmid': '1', **body}, headers=AUTH('alice')).status_code == 422
    assert pipe.calls == []
