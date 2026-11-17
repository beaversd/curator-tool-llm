"""Compartments from the paper, per reaction: code finds the passages naming a participant with a location word, the
model only picks among them (evidence only), suggestions never applied, one molecule in two compartments is two
entities (an existing one reused), a reaction can span compartments, and the existing Reactome instance of a placed
molecule is reused."""
import re
from types import SimpleNamespace

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (ComplexSpec, EwasSpec, ExistingRef, PathwaySpec, ReactionSpec, ReactomeDraft)
from curator_llm.services.compartments import (Passages, Placement, PlacementBatch, ReactionPlacements,
                                               assign_compartments, names_of, offer, reference_names, same_molecule,
                                               sentences, _name_regex)
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup

CYTOSOL = ExistingRef(db_id=70101, display_name='cytosol', schema_class='Compartment')
NUCLEO = ExistingRef(db_id=7660, display_name='nucleoplasm', schema_class='Compartment')
PM = ExistingRef(db_id=876, display_name='plasma membrane', schema_class='Compartment')
VOCAB = {c.display_name: c for c in (CYTOSOL, NUCLEO, PM)}
PAPER = ('beta-catenin is stabilized in the cytosol before translocating to the nucleus where it partners with TCF/LEF. '
         'K15 is a transmembrane protein that binds intracellular TRAF2. Nothing here about AXL.')
Q_CYT = 'beta-catenin is stabilized in the cytosol'
Q_NUC = 'translocating to the nucleus where it partners with TCF/LEF'
Q_K15 = 'K15 is a transmembrane protein that binds intracellular TRAF2'


class FakeModel:
    """A LangChain-like chat model answering from {reaction key: [(participant, compartment, basis, quote, suggested)]}:
    it cites the passage offered for that participant that contains the quote (else a passage id it was not offered)."""

    def __init__(self, table):
        self.table, self.messages = table, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages, config=None):
        self.messages.append(messages)
        text = messages[0].content[1]['text']
        passages = dict(re.findall(r'^\[(S[\d-]+)\] (.*)$', text, re.M))
        out = []
        for rk, rows in self.table.items():
            block = re.search(rf'^- {rk}:.*?(?=^- |\Z)', text, re.M | re.S)
            if block is None:
                continue
            pls = []
            for p, c, b, q, sg in rows:
                line = re.search(rf'^    [\w ]+: {p} = .*passages: (.*)$', block.group(0), re.M)
                ids = [] if line is None or line.group(1) == 'none' else line.group(1).split(', ')
                cite = [i for i in ids if q and q in passages[i]][:1] or (['S999'] if q else [])
                pls.append(Placement(participant=p, compartment=c, basis=b, passages=cite, suggested=sg, reason='r'))
            out.append(ReactionPlacements(reaction=rk, placements=pls))
        return PlacementBatch(reactions=out)


def draft():
    d = ReactomeDraft()
    d.participants = {
        'ctnnb1': EwasSpec(key='ctnnb1', name='CTNNB1', uniprot='P35222', aliases=['beta-catenin']),
        'ctnnb1_nuc': EwasSpec(key='ctnnb1_nuc', name='CTNNB1 nuclear', aliases=['beta-catenin']),  # no accession: same molecule
        'ctnnb1_deg': EwasSpec(key='ctnnb1_deg', name='CTNNB1 degraded', uniprot='P35222'),   # a state: not the same
        'tcf': EwasSpec(key='tcf', name='TCF7L2', uniprot='Q9NQB0', aliases=['TCF/LEF']),
        'cx': ComplexSpec(key='cx', name='CTNNB1:TCF7L2', components=['ctnnb1', 'tcf']),
        'k15': EwasSpec(key='k15', name='K15'), 'traf2': EwasSpec(key='traf2', name='TRAF2', uniprot='Q12933'),
        'axl': EwasSpec(key='axl', name='AXL', uniprot='P30530')}
    d.reactions = [
        ReactionSpec(key='r18', name='CTNNB1 is degraded', inputs=['ctnnb1'], outputs=['ctnnb1_deg'], evidence_ids=['e1']),
        ReactionSpec(key='r20', name='CTNNB1 translocates', inputs=['ctnnb1'], outputs=['ctnnb1_nuc'], evidence_ids=['e1']),
        ReactionSpec(key='r2', name='CTNNB1 binds TCF', inputs=['ctnnb1', 'tcf'], outputs=['cx'], evidence_ids=['e2']),
        ReactionSpec(key='r61', name='K15 binds TRAF2', inputs=['k15', 'traf2', 'axl'], outputs=[], evidence_ids=['e3'])]
    d.pathway = PathwaySpec(name='p', reactions=[r.key for r in d.reactions])
    return d


EVIDENCE = {'e1': Evidence(id='e1', quote=Q_CYT, source='PMID:1'), 'e2': Evidence(id='e2', quote=Q_NUC, source='PMID:1'),
            'e3': Evidence(id='e3', quote=Q_K15, source='PMID:1')}
TABLE = {'r18': [('ctnnb1', 'cytosol', 'stated', Q_CYT, None), ('ctnnb1_deg', None, 'not_in_paper', None, 'cytosol')],
         'r20': [('ctnnb1', 'cytosol', 'stated', Q_CYT, None), ('ctnnb1_nuc', 'nucleoplasm', 'stated', Q_NUC, None)],
         'r2': [('ctnnb1', 'nucleoplasm', 'stated', Q_NUC, None), ('tcf', 'nucleoplasm', 'stated', Q_NUC, None),
                ('cx', 'nucleoplasm', 'stated', Q_NUC, None)],
         'r61': [('k15', 'plasma membrane', 'stated', Q_K15, None), ('traf2', 'cytosol', 'stated', Q_K15, None),
                 ('axl', None, 'not_in_paper', None, 'plasma membrane')]}


def test_each_reaction_places_its_participants_from_the_text_and_a_second_compartment_is_another_entity():
    d = draft()
    issues = assign_compartments(d, EVIDENCE, {'PMID:1': PAPER}, VOCAB, FakeModel(TABLE))
    P = d.participants
    assert P['ctnnb1'].compartment == CYTOSOL and Q_CYT in P['ctnnb1'].compartment_quote    # the passage, from code
    # r2 binds in the nucleus: it now uses the nuclear beta-catenin already in the draft, and so does the complex it forms
    r2 = next(r for r in d.reactions if r.key == 'r2')
    assert r2.inputs == ['ctnnb1_nuc', 'tcf'] and P['cx'].components == ['ctnnb1_nuc', 'tcf']
    assert [c.display_name for c in r2.compartments] == ['nucleoplasm']
    assert any(i.code == 'compartment_split' and 'r2 now uses CTNNB1 nuclear' in i.message for i in issues)
    # a reaction across compartments
    r61 = next(r for r in d.reactions if r.key == 'r61')
    assert [c.display_name for c in r61.compartments] == ['plasma membrane', 'cytosol']
    # no evidence: nothing applied, a suggestion only
    assert P['axl'].compartment is None and P['ctnnb1_deg'].compartment is None
    assert any(i.code == 'compartment_not_in_paper' and 'AXL' in i.message and 'suggested: plasma membrane' in i.message
               for i in issues)
    r20 = next(r for r in d.reactions if r.key == 'r20')
    assert [c.display_name for c in r20.compartments] == ['cytosol', 'nucleoplasm']


def test_only_passages_are_sent_and_a_placement_without_an_offered_passage_or_with_an_unknown_name_is_not_used():
    d = draft()
    table = {'r61': [('k15', 'plasma membrane', 'stated', 'K15 sits on the plasma membrane', None),     # not in the paper
                     ('traf2', 'golgi', 'stated', Q_K15, None)]}                                     # not a compartment
    m = FakeModel(table)
    issues = assign_compartments(d, EVIDENCE, {'PMID:1': PAPER}, VOCAB, m)
    assert d.participants['k15'].compartment is None and d.participants['traf2'].compartment is None
    assert {i.code for i in issues} >= {'compartment_no_passage', 'compartment_unknown'}
    shared, batch = m.messages[0][0].content
    assert shared['cache_control'] == {'type': 'ephemeral'} and 'COMPARTMENTS' in shared['text']
    assert PAPER not in shared['text'] + batch['text'] and Q_K15 in batch['text']       # passages, not the whole paper


def test_a_state_is_not_the_same_molecule_but_a_location_word_is():
    P = draft().participants
    assert same_molecule(P['ctnnb1'], P['ctnnb1_nuc']) and not same_molecule(P['ctnnb1'], P['ctnnb1_deg'])


def test_each_reaction_is_placed_from_its_own_papers_text():
    d = draft()
    ev = dict(EVIDENCE, e3=Evidence(id='e3', quote='other', source='PMID:2'))
    assign_compartments(d, ev, {'PMID:1': PAPER, 'PMID:2': 'A different paper about K15.'}, VOCAB, FakeModel(TABLE))
    assert d.participants['k15'].compartment is None                      # its quote is not in its own paper's text
    assert d.participants['ctnnb1'].compartment == CYTOSOL


class GraphLookup(FakeInstanceLookup):
    def compartments(self, min_uses=50):
        return [{'dbId': c.db_id, 'displayName': c.display_name, 'uses': 100} for c in VOCAB.values()]

    def find_unmodified_entity(self, identifier, compartment_db_id):
        hits = {('Q12933', CYTOSOL.db_id): [{'dbId': 9, 'displayName': 'TRAF2 [cytosol]',
                                             'schemaClassName': 'EntityWithAccessionedSequence'}]}
        return hits.get((identifier, compartment_db_id), [])


def test_the_pipeline_places_participants_then_reuses_existing_reactome_instances():
    d = draft()
    pipe = DefaultPipeline(GraphLookup([]), compartment_model=FakeModel(TABLE), compartments=True)
    store = SimpleNamespace(all=lambda: list(EVIDENCE.values()))
    papers = {'PMID:1': SimpleNamespace(text=PAPER)}
    issues = pipe._place(d, store, papers, SimpleNamespace(refs=lambda: []), [], lambda m: None)
    assert d.participants['traf2'].compartment == CYTOSOL and not any(i.code == 'compartments_failed' for i in issues)
    notes = Resolver(GraphLookup([])).match_existing(d)
    assert d.participants['traf2'].existing.db_id == 9 and any('TRAF2 [cytosol]' in n for n in notes)
    assert d.participants['axl'].existing is None                          # not placed: not looked up


def test_a_failing_model_never_fails_the_annotation_and_a_lookup_without_compartments_skips():
    class Broken(FakeModel):
        def invoke(self, messages, config=None):
            raise ConnectionError('model down')
    d = draft()
    store = SimpleNamespace(all=lambda: list(EVIDENCE.values()))
    papers = {'PMID:1': SimpleNamespace(text=PAPER)}
    issues = DefaultPipeline(GraphLookup([]), compartment_model=Broken({}), compartments=True)._place(
        d, store, papers, SimpleNamespace(refs=lambda: []), [], lambda m: None)
    assert [i.code for i in issues] == ['compartments_failed']
    assert DefaultPipeline(FakeInstanceLookup([]), compartment_model=FakeModel(TABLE))._place(
        draft(), store, papers, SimpleNamespace(refs=lambda: []), [], lambda m: None) == []


def test_a_suggested_compartment_is_a_pending_fix_linked_to_its_issue_and_accepting_makes_it_the_curators():
    from curator_llm.models.session import Proposal
    from curator_llm.services import issues as iss
    from curator_llm.services.patching import apply_patch
    from curator_llm.services.suggestions import suggest_fixes
    d = draft()
    issues = assign_compartments(d, EVIDENCE, {'PMID:1': PAPER}, VOCAB, FakeModel(TABLE))
    assert d.participants['axl'].compartment is None and d.participants['axl'].compartment_suggested == PM
    fixes = [s for s in suggest_fixes(d, list(EVIDENCE.values())) if s.code == 'compartment']
    axl = next(s for s in fixes if '/participants/axl/' in s.ops[0]['path'])
    assert axl.reaction_key == 'r61' and 'general knowledge' in axl.reason
    p = Proposal(id='p1', actor='suggestions', ops=axl.ops, reaction_keys=[axl.reaction_key])
    issue = next(i for i in issues if i.code == 'compartment_not_in_paper' and i.participant_key == 'axl')
    iss.organize([issue], d, [p])
    assert issue.proposal_id == 'p1'
    after = apply_patch(d, axl.ops).participants['axl']
    assert after.compartment == PM and after.compartment_source == 'curator' and after.compartment_suggested is None


def test_passages_name_the_participant_with_a_location_word_outside_its_name():
    text = ('LANA is the latency-associated nuclear antigen of KSHV. LANA binds RB1 in infected cells. '
            'Both then move into the nucleus. Unrelated: SP1 is abundant.\n\nReferences\n'
            '1. LANA tethers episomes to the nucleus. J Virol.')
    ps = Passages(text)
    assert len(ps.sents) == 4                                     # the reference list is not searched
    lana = _name_regex(['LANA1', 'LANA', 'latency-associated nuclear antigen'])
    found = ps.find(lana)
    # sentence 1: "nuclear" only inside its name, not a placement; sentence 3 follows one naming it
    assert [pid for pid, _ in found] == ['S2-3'] and ps.text('S2-3').startswith('LANA binds RB1')
    rb1 = _name_regex(['RB1'])
    cx = ps.find(None, [lana, rb1], need_parts=2)               # a complex named by two of its parts
    assert [pid for pid, _ in cx] == ['S2-3']
    assert sentences('See Fig. 2 for details. Next sentence.') == ['See Fig. 2 for details.', 'Next sentence.']
    gsk = _name_regex(['GSK3B', 'GSK-3 beta'])
    nfkb = _name_regex(['NF-kB'])
    ps = Passages('LANA sequesters nuclear GSK-3 here. NF-κB then enters the nucleus. IκB stays in the cytoplasm.')
    assert [p for p, _ in ps.find(gsk)] == ['S1'] and [p for p, _ in ps.find(nfkb)] == ['S2']
    assert [p for p, _ in ps.find(_name_regex(['IkB-alpha']))] == ['S3']
    assert sentences('It accumulates.38 LANA binds it.12,13 Then 3.5 fold.') == ['It accumulates.', 'LANA binds it.',
                                                                                'Then 3.5 fold.']


def test_offered_passages_are_ranked_for_the_reaction_and_capped():
    d = draft()
    ps = Passages(PAPER)
    got = offer(d, EVIDENCE, ps, d.reactions)
    assert got[('r2', 'ctnnb1')] == ['S1'] and got[('r2', 'cx')] == ['S1'] and got[('r61', 'axl')] == []
    assert got[('r61', 'k15')] == got[('r61', 'traf2')] == ['S2']
    assert offer(d, EVIDENCE, Passages(PAPER), d.reactions) == got           # deterministic


def test_reference_names_come_from_reactome_and_uniprot():
    class L:
        def reference_names(self, ident):
            return {'P35222': ['CTNNB1', 'CTNNB'], 'CHEBI:30616': ['ATP(4-)', 'ATP']}.get(ident, [])

    class U:
        def fetch(self, acc):
            return {'genes': ['CTNNB1'], 'aliases': ['CTNNB'], 'names': ['Catenin beta-1', 'Beta-catenin']}
    extra = reference_names(L(), U())
    p = EwasSpec(key='c', name='CTNNB1', uniprot='P35222', aliases=['b-cat'])
    assert names_of(p, extra) == ['CTNNB1', 'b-cat', 'CTNNB', 'Catenin beta-1', 'Beta-catenin']
    from curator_llm.models.reactome import SimpleEntitySpec
    assert 'ATP' in names_of(SimpleEntitySpec(key='a', name='adenosine triphosphate', chebi='CHEBI:30616'), extra)
