"""Gene2Path sessions: papers chosen because they name the focus gene with one of its FI partners, annotated to be
placed in a predicted pathway. Extraction still reads the whole paper (its prompts are tuned and prompt-cached), so
the draft holds every reaction the papers describe; this puts the ones involving both genes first and says which they
are, or that there is none, and places the draft in the predicted pathway. Code only, no model call."""
import re
from typing import Dict, List, Optional, Set

from curator_llm.models.reactome import ComplexSpec, DefinedSetSpec, EwasSpec, ExistingRef, ReactionSpec, ReactomeDraft
from curator_llm.models.session import Issue
from curator_llm.ports.lookup import InstanceLookup


def _names(symbol: str, text: str) -> bool:
    """The symbol as a whole word of text, case-insensitive ('p-S65-PRKN' names PRKN, 'PINK1A' does not name PINK1)."""
    return bool(re.search(rf'(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])', text or '', re.I))


def entities_of(draft: ReactomeDraft, symbol: str, accession: Optional[str] = None) -> Set[str]:
    """Keys of the participants that are, or contain, the gene: an EWAS named by the symbol or carrying its UniProt
    accession, and every complex or set with one among its components or members (at any depth)."""
    found = {k for k, p in draft.participants.items() if isinstance(p, EwasSpec)
             and (_names(symbol, p.name) or (accession and p.uniprot == accession))}
    changed = True
    while changed:
        changed = False
        for k, p in draft.participants.items():
            parts = p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else []
            if k not in found and any(c in found for c in parts):
                found.add(k)
                changed = True
    return found


def _uses(r: ReactionSpec) -> Set[str]:
    return set(r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []))


def link_partner(draft: ReactomeDraft, gene: str, partner: str,
                 accessions: Optional[Dict[str, Optional[str]]] = None) -> List[Issue]:
    """Put the reactions involving both gene and partner first (in the draft and its pathway, otherwise in their order)
    and return one issue per such reaction, or a warning when there is none. `accessions` is {symbol: UniProt
    accession}, when known: it finds an entity the paper names differently (Parkin for PRKN)."""
    accessions = accessions or {}
    ours = entities_of(draft, gene, accessions.get(gene))
    theirs = entities_of(draft, partner, accessions.get(partner))
    linked = [r.key for r in draft.reactions if _uses(r) & ours and _uses(r) & theirs]
    if not linked:
        return [Issue(source='pipeline', severity='warning', code='partner_link:none',
                      message=f'no reaction in the draft involves both {gene} and {partner}: the papers may only '
                              f'mention them together, or describe the event in a way the extraction missed')]
    first = set(linked)
    draft.reactions.sort(key=lambda r: r.key not in first)
    if draft.pathway is not None:
        draft.pathway.reactions.sort(key=lambda k: k not in first)
    return [Issue(source='pipeline', severity='info', code='partner_link', reaction_key=r.key,
                  message=f'"{r.name}" involves both {gene} and {partner}')
            for r in draft.reactions if r.key in first]


def place_in_pathway(draft: ReactomeDraft, pathway_id: int, lookup: InstanceLookup) -> List[Issue]:
    """Make the draft's pathway the existing one (dbId), so the curator adds the new reactions to it instead of
    creating a pathway. A dbId that is not a pathway in the database leaves the draft's own pathway, with a warning."""
    if draft.pathway is None:
        return []
    try:
        inst = lookup.find_by_db_id(pathway_id)
    except Exception as e:
        inst, err = None, f'{type(e).__name__}: {e}'
    else:
        err = None
    if not inst or inst.get('schemaClassName') not in ('Pathway', 'TopLevelPathway'):
        why = err or ('not found' if not inst else f'a {inst.get("schemaClassName")}, not a Pathway')
        return [Issue(source='pipeline', severity='warning', code='pathway_placement',
                      message=f'the predicted pathway (dbId {pathway_id}) could not be used ({why}); the draft '
                              f'proposes a new pathway instead')]
    draft.pathway.existing = ExistingRef(db_id=inst['dbId'], display_name=inst.get('displayName', ''),
                                         schema_class=inst['schemaClassName'])
    draft.pathway.name = inst.get('displayName') or draft.pathway.name
    return []
