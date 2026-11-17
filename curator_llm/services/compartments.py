"""Where each participant is, in each reaction, according to the paper.

Reactome annotation is evidence based: a compartment goes into the draft only when the paper's text places the molecule
there for that reaction. CODE finds the candidate passages: the sentences that name the participant (by its name or an
alias: the paper's own name, UniProt short names and gene synonyms, Reactome's reference names) or follow one that does,
and contain a location word outside the name itself ("LANA" is latency-associated NUCLEAR antigen). They are ranked for the
reaction (naming its other participants, near its quotes) and capped. The MODEL then only judges, per reaction, which of
the passages offered for a participant place it, and where: "beta-catenin accumulates in the cytoplasm before
translocating to the nucleus where it partners with TCF/LEF" places beta-catenin in the nucleus for the TCF/LEF binding,
and in the cytoplasm before. A placement must cite offered passages (their text is the recorded quote), so the same small
input every run makes the placed-or-not decision repeatable, and a participant with no passage is never placed. General
knowledge of a protein is not evidence: a likely compartment may be SUGGESTED (a pending fix for the curator), never
applied. Reactome is used only for its vocabulary (compartment names -> Compartment instances).

Measured with the whole paper in the prompt instead (thinking on, KSHV review, 5 runs): no participant got two different
compartments, but 72 to 121 of 250 were placed, the model being stricter in some runs than others.

A reaction can span compartments (a membrane receptor and a cytosolic partner, a translocation): its compartments are its
participants'. One molecule placed in different compartments in different reactions is different entities, as in
Reactome: the reactions placing it elsewhere use an existing participant of the same molecule in that compartment, or a
new one (and a complex that reaction forms is updated to contain it). Every value records its basis
(Participant.compartment_source) and quote.

Measured before this existed: papers place 2 of 285 participants by an explicit compartment field (KSHV, PINK1, WEE1,
clock papers), and no reaction had a compartment.
"""
import collections
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from curator_llm.models.reactome import (ComplexSpec, DefinedSetSpec, EwasSpec, ExistingRef, ReactomeDraft,
                                         SimpleEntitySpec)
from curator_llm.models.session import Issue

BATCH = 12               # reactions per model call
WORKERS = 4
MAX_PASSAGES = 6         # per participant per reaction
MAX_PASSAGE_CHARS = 600


class Placement(BaseModel):
    # field order is the order the model writes them: the passages and the reason first, then the answer (asked first,
    # the compartment was often left null while the reason named one: "NF-kB exists in the cytoplasm ...")
    participant: str = Field(..., description='The participant key, as listed for the reaction')
    passages: List[str] = Field(default_factory=list, description='The ids of the passages offered for this participant '
                                                                  'that place it in this reaction (e.g. ["S12"]); empty '
                                                                  'when none does')
    reason: str = Field('', description='At most 25 words: how those passages place it in this reaction, or why none does')
    basis: Literal['stated', 'shown', 'not_in_paper'] = Field(..., description='stated: a passage says where it is in '
                                                              'this reaction; shown: an experiment in a passage places '
                                                              'it; not_in_paper: neither')
    compartment: Optional[str] = Field(None, description='Required for stated and shown: the name from the COMPARTMENTS '
                                                         'list. Null only for not_in_paper')
    suggested: Optional[str] = Field(None, description='Only for not_in_paper: the most likely compartment from the '
                                                       'COMPARTMENTS list, from general knowledge; shown to the curator, '
                                                       'never applied. Null if unsure.')


class ReactionPlacements(BaseModel):
    reaction: str = Field(..., description='The reaction key')
    placements: List[Placement]


class PlacementBatch(BaseModel):
    reactions: List[ReactionPlacements]


def _kind(p) -> str:
    if isinstance(p, EwasSpec):
        return {'dna': 'gene', 'rna': 'rna'}.get(p.molecule, 'protein')
    if isinstance(p, SimpleEntitySpec):
        return 'small molecule'
    return 'complex' if isinstance(p, ComplexSpec) else 'set'


def roles(r) -> List[Tuple[str, str]]:
    """(role, participant key) of a reaction."""
    out = [('input', k) for k in r.inputs] + [('output', k) for k in r.outputs]
    if r.catalyst:
        out.append(('catalyst', r.catalyst.entity))
    out += [(f'{g.kind} regulator', g.regulator) for g in r.regulations]
    return out


# ── finding the passages (code, deterministic) ──────────────────────────────────────────────────
# text and names are compared in one form: Greek letters, written or spelled, as one Latin letter (NF-κB = NF-kappaB =
# NF-kB, β-catenin = beta-catenin, IkappaBalpha = IkBa) and no hyphen inside a word (GSK-3 = GSK3)
_GREEK = {'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'σ': 's', 'ζ': 'z',
          'θ': 'th', 'ω': 'w'}
_SPELLED = re.compile(r'kappa|(alpha|beta|gamma|delta)(?![a-z])')
_GREEK_SUFFIX = re.compile(r'[\s-]+(alpha|beta|gamma|delta|kappa|[αβγδκ])$', re.I)
_DASH = re.compile('[‐-―−]')
CUE = re.compile(r'\b(nucle(?:us|i|ar|oplasm\w*|olus|olar)|chromatin|perinuclear|cytoplasm\w*|cytosol\w*|'
                 r'\w*membranes?|cell surface|surface of|extracellular|secret(?:ed|ion)|'
                 r'mitochondri\w*|endoplasmic|golgi|lysosom\w*|endosom\w*|vesicles?|centrosom\w*|'
                 r'translocat\w*|relocali[sz]\w*|locali[sz]\w*|shuttl\w*|import(?:ed|s)? into|export(?:ed|s)? (?:from|of))\b')
# location abbreviations, matched case-sensitively in the original sentence ("free Ub in the vicinity of the OMM")
ABBR = re.compile(r'\b(OMM|IMM|MOM|IMS|ER|ERGIC|PM)\b')
# a sentence referring back to the one before ("It then moves into the nucleus"): only then is the previous sentence's
# molecule the one placed
_ANAPHOR = re.compile(r'\b(it|its|they|their|them|both|this|these|the (?:protein|proteins|complex|kinase|factor|'
                      r'receptor|enzyme|ligase|substrate|latter|former))\b')
_NO_SPLIT = re.compile(r'(?:\bal|\be\.g|\bi\.e|\bfig|\bfigs|\bvs|\bref|\bapprox|\bca|\bno|\bsupp)\.$', re.I)
_REFS = re.compile(r'\n\s*(references|literature cited|bibliography)\s*\n', re.I)
_GENERIC = {'protein', 'proteins', 'complex', 'gene', 'genes', 'dna', 'rna', 'mrna', 'cell', 'cells', 'factor'}
_STRIP = re.compile(r'\b(gene|mrna|transcript|protein|complex)\b')


def _norm(s: str) -> str:
    s = _DASH.sub('-', s.lower())
    for g, n in _GREEK.items():
        s = s.replace(g, n)
    s = _SPELLED.sub(lambda m: m.group(0)[0], s)
    return re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', '', s)


def sentences(text: str) -> List[str]:
    """The paper's sentences, without the reference list; line breaks inside a sentence (PDF text) joined."""
    m = [x for x in _REFS.finditer(text) if x.start() > len(text) * 0.5]
    if m:
        text = text[:m[-1].start()]
    out = []
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s+', ' ', para).strip()
        start = 0
        # a sentence end, also before citation numbers stuck to the period ("accumulation.38 LANA ...")
        for b in re.finditer(r'(?<=[.!?])(?:\d{1,3}(?:[,–-]\d{1,3})*)?\s+(?=[A-Z(\[])|(?<=[.!?])\s+(?=\d)', para):
            if _NO_SPLIT.search(para[start:b.start()]):
                continue
            out.append(para[start:b.start()])
            start = b.end()
        if para[start:]:
            out.append(para[start:])
    return out


def _name_regex(names) -> Optional[re.Pattern]:
    alts = set()
    variants = []
    for n in names:
        variants += [n, _GREEK_SUFFIX.sub('', n.strip())]     # "GSK-3 beta" is also written "GSK-3"
    for n in dict.fromkeys(variants):
        for v in (_norm(n).strip(), _STRIP.sub(' ', _norm(n)).strip()):
            v = re.sub(r'\s+', ' ', v)
            if len(v) >= 2 and v not in _GENERIC:            # 2 letters too: "Ub", matched as a whole word
                alts.add(re.sub(r'(\\-|\\ )', r'[-\\s]?', re.escape(v)))
    if not alts:
        return None
    return re.compile(r'(?<![a-z0-9])(?:' + '|'.join(sorted(alts, key=len, reverse=True)) + r')(?![a-z0-9])')


def names_of(p, extra: Optional[Callable[[Any], List[str]]] = None) -> List[str]:
    """Every name the paper may use for a participant: its name, its aliases, and `extra` (reference names)."""
    out = [p.name] + list(p.aliases or [])
    if extra is not None:
        try:
            out += list(extra(p) or [])
        except Exception:                         # a lookup failure only means fewer names to search for
            pass
    return list(dict.fromkeys(n for n in out if n))


def reference_names(lookup: Any = None, uniprot: Any = None) -> Callable[[Any], List[str]]:
    """extra_names for assign_compartments: Reactome's names for a participant's reference (lookup.reference_names) and
    UniProt's gene names, synonyms, protein names and short names (uniprot.fetch). Cached per identifier."""
    cache: Dict[str, List[str]] = {}

    def names(p) -> List[str]:
        ident = getattr(p, 'uniprot', None) or ((f'CHEBI:{p.chebi.split(":")[-1]}') if getattr(p, 'chebi', None) else None)
        if not ident:
            return []
        if ident not in cache:
            out: List[str] = []
            if lookup is not None and hasattr(lookup, 'reference_names'):
                out += lookup.reference_names(ident) or []
            if uniprot is not None and getattr(p, 'uniprot', None):
                info = uniprot.fetch(p.uniprot) or {}
                out += info.get('genes', []) + info.get('aliases', []) + info.get('names', [])
            cache[ident] = list(dict.fromkeys(out))
        return cache[ident]
    return names


class Passages:
    """One paper's location sentences, and which of them mention each participant."""

    def __init__(self, text: str):
        self.sents = sentences(text)
        self.norm = [_norm(x) for x in self.sents]
        self.cue = [bool(CUE.search(x) or ABBR.search(raw)) for x, raw in zip(self.norm, self.sents)]
        self._text: Dict[str, str] = {}

    def text(self, pid: str) -> str:
        return self._text[pid]

    def _hits(self, rx: Optional[re.Pattern], need: int = 1) -> set:
        """Sentence indexes naming the participant (`need` different names, for a complex by its parts)."""
        if rx is None:
            return set()
        return {i for i, x in enumerate(self.norm) if len({m.group(0) for m in rx.finditer(x)}) >= need}

    def find(self, own: Optional[re.Pattern], parts: List[Optional[re.Pattern]] = (), need_parts: int = 1) -> List[Tuple[str, int]]:
        """(passage id, sentence index) of passages placing-candidates: a cue sentence naming it (the cue outside the
        name) or a cue sentence referring back ("it", "they", "the complex") to one naming it. A complex/set is named by its own name, or by `need_parts`
        of its parts."""
        named = self._hits(own)
        if parts:
            counts = collections.Counter(i for rx in parts for i in self._hits(rx))
            named |= {i for i, c in counts.items() if c >= need_parts}
        out = []
        for i in range(len(self.sents)):
            if not self.cue[i]:
                continue
            if i in named:
                blank = self.norm[i]
                for rx in [own, *parts]:
                    if rx is not None:
                        blank = rx.sub(' ', blank)
                names = {m.group(0) for rx in [own, *parts] if rx is not None for m in rx.finditer(self.norm[i])}
                abbr = any(m.group(0).lower() not in names for m in ABBR.finditer(self.sents[i]))
                if not CUE.search(blank) and not abbr:        # the location word was only inside its name
                    continue
                pid, txt = f'S{i + 1}', self.sents[i]
            elif i - 1 in named and _ANAPHOR.search(self.norm[i]):
                pid, txt = f'S{i}-{i + 1}', self.sents[i - 1] + ' ' + self.sents[i]
            else:
                continue
            self._text.setdefault(pid, txt[:MAX_PASSAGE_CHARS])
            out.append((pid, i))
        return out

    def position(self, quote: str) -> Optional[int]:
        """The sentence a reaction quote is in (its first words), to rank passages near it first."""
        q = re.sub(r'\s+', ' ', _norm(quote)).strip()[:60]
        if len(q) < 15:
            return None
        return next((i for i, x in enumerate(self.norm) if q in x), None)


INSTRUCTIONS = """You are a Reactome biocurator. For each reaction below, say where each of its participants is IN THAT
REACTION, according to the paper. For each participant you are given numbered PASSAGES from the paper: the sentences that
name it and mention a location (found by a program, so many will not be about this reaction). Use only those passages,
with the reaction's quotes for context. Example: "beta-catenin accumulates in the cytoplasm before translocating to the
nucleus where it partners with TCF/LEF" places beta-catenin in the nucleus for binding TCF/LEF, and in the cytoplasm
before it moves.

- stated: a passage says or clearly implies where the participant is in this reaction. Cite its id(s).
- shown: an experiment in a passage places it (nuclear or cytoplasmic fractions, chromatin binding, mitochondrial
  isolation, immunofluorescence, a receptor at the cell surface). Cite its id(s).
- not_in_paper: no passage places it in this reaction (a passage may be about another reaction, another molecule's
  location, or another condition). A participant with no passages is always not_in_paper. Leave the compartment null:
  Reactome annotation needs evidence, and what you know of the protein in general is not evidence. You may give the most
  likely compartment in `suggested`; it is only shown to the curator.
An experiment in vitro (purified or recombinant proteins, a cell lysate or supernatant, an in vitro kinase or
ubiquitination assay) places nothing in a cell compartment: a test tube is not a compartment. For such a reaction use only
passages about where the event happens in cells; otherwise not_in_paper.
A reaction can involve several compartments (a membrane receptor and a cytosolic partner; a translocation from one
compartment to another). Place each participant on its own; a complex is where it is formed or acts in this reaction.
Name compartments exactly as in the COMPARTMENTS list. Reactome practice: a soluble protein inside the cell is in the
cytosol (use "cytoplasm" only when the text means the whole cytoplasm, organelles included); a nuclear protein is in the
nucleoplasm. Answer for every participant of every reaction listed."""


def offer(draft: ReactomeDraft, evidence: Dict[str, Any], passages: Passages, reactions: list,
          extra_names: Optional[Callable[[Any], List[str]]] = None) -> Dict[Tuple[str, str], List[str]]:
    """{(reaction key, participant key): passage ids offered, best first}. Deterministic."""
    P = draft.participants
    rx: Dict[str, Optional[re.Pattern]] = {}

    def regex(k):
        if k not in rx:
            rx[k] = _name_regex(names_of(P[k], extra_names)) if k in P else None
        return rx[k]
    found: Dict[str, List[Tuple[str, int]]] = {}

    def candidates(k):
        if k not in found:
            p = P[k]
            inner = (p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else [])
            inner = [x for x in dict.fromkeys(inner) if x in P]
            found[k] = passages.find(regex(k), [regex(x) for x in inner],
                                     need_parts=min(2, len(inner)) if isinstance(p, ComplexSpec) else 1)
        return found[k]
    out = {}
    for r in reactions:
        keys = [k for _, k in roles(r) if k in P]
        near = [i for i in (passages.position(evidence[e].quote) for e in r.evidence_ids if e in evidence) if i is not None]
        for k in dict.fromkeys(keys):
            others = [regex(x) for x in keys if x != k and regex(x) is not None]

            def rank(c):
                pid, i = c
                txt = _norm(passages.text(pid))
                return (-sum(bool(o.search(txt)) for o in others), min((abs(i - n) for n in near), default=10 ** 6), i)
            out[(r.key, k)] = [pid for pid, _ in sorted(candidates(k), key=rank)[:MAX_PASSAGES]]
    return out


def prompt_parts(draft: ReactomeDraft, evidence: Dict[str, Any], passages: Passages, reactions: list,
                 offered: Dict[Tuple[str, str], List[str]], vocabulary: List[str]) -> Tuple[str, str]:
    """(the part shared by every call: instructions and compartment names -- cached; the batch: its passages and
    reactions)."""
    P = draft.participants
    used, lines = [], []
    for r in reactions:
        quotes = [evidence[i].quote for i in r.evidence_ids if i in evidence][:3]
        parts = []
        for role, k in roles(r):
            if k not in P:
                continue
            p = P[k]
            extra = ''
            if isinstance(p, (ComplexSpec, DefinedSetSpec)):
                inner = p.components if isinstance(p, ComplexSpec) else p.members
                extra = f' = {" + ".join(P[x].name for x in inner if x in P)}'
            ids = offered.get((r.key, k), [])
            used += ids
            parts.append(f'    {role}: {k} = {p.name} ({_kind(p)}){extra}; passages: {", ".join(ids) or "none"}')
        q = ''.join(f'\n    quote: "{x[:400]}"' for x in quotes)
        lines.append(f'- {r.key}: {r.name}\n' + '\n'.join(parts) + q)
    shown = ''.join(f'\n[{pid}] {passages.text(pid)}' for pid in dict.fromkeys(used))
    shared = f'{INSTRUCTIONS}\n\nCOMPARTMENTS (Reactome\'s names): {", ".join(vocabulary)}'
    return shared, (f'PASSAGES{shown or " (none)"}\n\nREACTIONS (participants by key)\n' + '\n'.join(lines))


def _ask(model, shared: str, batch: str):
    """One structured call; the shared part is marked for prompt caching."""
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.messages import HumanMessage
    counter = UsageMetadataCallbackHandler()
    msg = HumanMessage(content=[{'type': 'text', 'text': shared, 'cache_control': {'type': 'ephemeral'}},
                                {'type': 'text', 'text': batch}])
    out = model.with_structured_output(PlacementBatch, method='json_schema').invoke([msg], config={'callbacks': [counter]})
    return out, counter.usage_metadata


def _source_of(r, evidence: Dict[str, Any], texts: Dict[str, str]) -> Optional[str]:
    """Which paper a reaction comes from: its quotes' paper, or the only one."""
    for i in r.evidence_ids:
        src = getattr(evidence.get(i), 'source', None)
        if src in texts:
            return src
    return next(iter(texts)) if len(texts) == 1 else None


_PLACE_WORDS = re.compile(r'\b(nuclear|nucleus|nucleoplasmic|cytoplasmic|cytosolic|cytoplasm|cytosol|membrane|'
                          r'mitochondrial|extracellular|secreted|intracellular|cell surface)\b')


def _plain(name: str) -> str:
    """A name without words that only say where it is ("CTNNB1 nuclear" -> "ctnnb1"; "CTNNB1 degraded" keeps its state)."""
    return re.sub(r'[^a-z0-9]+', '', _PLACE_WORDS.sub(' ', name.lower()))


def same_molecule(p, q) -> bool:
    """Same molecule in the same state, wherever it is: same kind, modifications and fragment, and the same identifier
    when both have one, else the same name without location words ("CTNNB1 nuclear" = "CTNNB1", but not "CTNNB1
    degraded")."""
    if type(p) is not type(q):
        return False
    if isinstance(p, EwasSpec):
        mods = lambda x: sorted((m.psi_mod or '', m.residue or '', m.coordinate or 0) for m in x.modifications)
        if (p.molecule, mods(p), p.start_coordinate, p.end_coordinate) != (q.molecule, mods(q), q.start_coordinate,
                                                                             q.end_coordinate):
            return False
        if p.uniprot and q.uniprot and p.uniprot != q.uniprot:
            return False
        return _plain(p.name) == _plain(q.name)
    if isinstance(p, SimpleEntitySpec):
        if p.chebi and q.chebi:
            return p.chebi == q.chebi and _plain(p.name) == _plain(q.name)
        return _plain(p.name) == _plain(q.name)
    inner = lambda x: sorted(x.components if isinstance(x, ComplexSpec) else x.members)
    return _plain(p.name) == _plain(q.name) and inner(p) == inner(q)


def assign_compartments(draft: ReactomeDraft, evidence: Dict[str, Any], texts: Dict[str, str],
                        vocabulary: Dict[str, ExistingRef], model: Any, usage: Optional[list] = None,
                        extra_names: Optional[Callable[[Any], List[str]]] = None) -> List[Issue]:
    """Place every participant of every reaction from the paper (see the module docstring); `draft` is changed in place.
    `texts`: each paper's text by source tag (a reaction is placed from its own paper's passages); `vocabulary`: Reactome
    compartment name -> instance; `model`: a LangChain chat model; `extra_names(participant)`: more names to find it by
    (reference names from UniProt / Reactome). Returns issues."""
    P = draft.participants
    for p in P.values():                     # placed already by extraction (the paper stated it, with its quote)
        if p.compartment is not None and not p.compartment_source:
            p.compartment_source = 'stated'
    by_name = {n.lower(): ref for n, ref in vocabulary.items()}
    issues: List[Issue] = []
    names = sorted(vocabulary)
    per_paper: Dict[str, list] = collections.defaultdict(list)
    for r in draft.reactions:
        src = _source_of(r, evidence, texts)
        if src is not None:
            per_paper[src].append(r)
    jobs, offered, passages_of = [], {}, {}
    for src, rxs in per_paper.items():
        ps = Passages(texts[src])
        got = offer(draft, evidence, ps, rxs, extra_names)
        offered.update(got)
        for r in rxs:
            passages_of[r.key] = ps
        for i in range(0, len(rxs), BATCH):
            jobs.append(prompt_parts(draft, evidence, ps, rxs[i:i + BATCH], got, names))
    answers = []
    if jobs:
        first = _ask(model, *jobs[0])                        # writes the cache the other calls read
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            answers = [first] + list(pool.map(lambda pr: _ask(model, *pr), jobs[1:]))
    if usage is not None:
        from curator_llm.models.usage import UsageEntry
        for _, meta in answers:
            for name, u in meta.items():
                # LangChain's input_tokens INCLUDES cached tokens: keep the three apart so each is priced once
                detail = u.get('input_token_details') or {}
                read = detail.get('cache_read') or 0
                write = detail.get('cache_creation') or ((detail.get('ephemeral_5m_input_tokens') or 0)
                                                         + (detail.get('ephemeral_1h_input_tokens') or 0))
                usage.append(UsageEntry(step='compartments', calls=1, model=name,
                                        input_tokens=max(0, u.get('input_tokens', 0) - read - write),
                                        output_tokens=u.get('output_tokens', 0),
                                        cache_read_tokens=read, cache_write_tokens=write))
    # (reaction key, participant key) -> (compartment ref, basis, quote, reason); suggestions per participant
    placed: Dict[Tuple[str, str], Tuple[ExistingRef, str, str, str]] = {}
    suggested: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rkeys = {r.key: r for r in draft.reactions}
    for out, _ in answers:
        for rp in (out.reactions if isinstance(out, PlacementBatch) else []):
            r = rkeys.get(rp.reaction)
            if r is None:
                continue
            mine = {k for _, k in roles(r)}
            for pl in rp.placements:
                k = pl.participant
                if k not in mine or k not in P:
                    continue
                if pl.basis != 'not_in_paper' and not pl.compartment and pl.passages:
                    issues.append(Issue(source='compartments', severity='warning', code='compartment_missing',
                                        reaction_key=r.key, participant_key=k,
                                        message=f'{P[k].name} in {r.key}: placed by the paper ({pl.reason}) but no '
                                                'compartment was named; not set'))
                    continue
                if pl.basis == 'not_in_paper' or not pl.compartment:
                    if pl.suggested and pl.suggested.strip().lower() in by_name:
                        suggested[k][by_name[pl.suggested.strip().lower()].display_name] += 1
                    continue
                ref = by_name.get(pl.compartment.strip().lower())
                if ref is None:
                    issues.append(Issue(source='compartments', severity='warning', code='compartment_unknown',
                                        reaction_key=r.key, participant_key=k,
                                        message=f'{P[k].name}: "{pl.compartment}" is not a Reactome compartment'))
                    continue
                allowed = offered.get((r.key, k), [])
                cited = [x.strip().strip('[]') for x in pl.passages if x.strip().strip('[]') in allowed]
                if not cited:
                    issues.append(Issue(source='compartments', severity='warning', code='compartment_no_passage',
                                        reaction_key=r.key, participant_key=k,
                                        message=f'{P[k].name} in {r.key}: {ref.display_name} was given without one of '
                                                'the passages found for it in the paper; not set'))
                    continue
                quote = ' … '.join(passages_of[r.key].text(x) for x in cited)
                placed[(r.key, k)] = (ref, pl.basis, quote, pl.reason)
    # per participant: the compartments the text places it in, across its reactions
    where: Dict[str, Dict[int, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for (rk, k), (ref, *_rest) in placed.items():
        where[k][ref.db_id].append(rk)
    refs = {ref.db_id: ref for ref, *_ in placed.values()}
    original = list(P)
    for k in original:
        if k not in where:
            continue
        p = P[k]
        ranked = sorted(where[k].items(), key=lambda kv: -len(kv[1]))
        main_id, main_rx = ranked[0]
        if p.compartment is None or p.compartment.db_id == main_id:
            _set(p, refs[main_id], placed[(main_rx[0], k)])
        for cid, rxs in ranked[1:]:                       # the same molecule elsewhere: another entity, as in Reactome
            target = _entity_in(draft, k, refs[cid], placed[(rxs[0], k)])
            for rk in rxs:
                _swap(rkeys[rk], k, target, draft)
            issues.append(Issue(source='compartments', severity='info', code='compartment_split', participant_key=k,
                                message=f'{p.name}: the paper places it in {refs[main_id].display_name} in '
                                        f'{", ".join(main_rx)} and in {refs[cid].display_name} in {", ".join(rxs)}; '
                                        f'{", ".join(rxs)} now use{"s" if len(rxs) == 1 else ""} '
                                        f'{P[target].name} [{refs[cid].display_name}]'))
    # participants the text does not place anywhere: left empty, maybe with a suggestion
    used = {k for r in draft.reactions for _, k in roles(r)}
    for k in sorted(used, key=lambda x: list(P).index(x) if x in P else 0):
        p = P.get(k)
        if p is not None and p.compartment is None:
            hint = ''
            if suggested.get(k):
                name = suggested[k].most_common(1)[0][0]
                p.compartment_suggested = by_name[name.lower()]
                hint = f'; suggested: {name} (general knowledge, not this paper)'
            issues.append(Issue(source='compartments', severity='info', code='compartment_not_in_paper',
                                participant_key=k, message=f'{p.name}: the paper does not say where it is{hint}'))
    # reactions: the compartments of their participants
    for r in draft.reactions:
        seen: List[ExistingRef] = []
        for _, k in roles(r):
            c = P[k].compartment if k in P else None
            if c is not None and all(c.db_id != x.db_id for x in seen):
                seen.append(c)
        r.compartments = seen
        r.compartment = seen[0] if seen else r.compartment
    return issues


def _set(p, ref: ExistingRef, how: tuple):
    _, basis, quote, _ = how
    p.compartment, p.compartment_name, p.compartment_source, p.compartment_quote = ref, ref.display_name, basis, quote
    p.compartment_suggested = None
    p.needs_resolution = [f for f in p.needs_resolution if f != 'compartment']


def _entity_in(draft: ReactomeDraft, key: str, ref: ExistingRef, how: tuple) -> str:
    """A participant that is `key`'s molecule in compartment `ref`: an existing one placed there (or unplaced, which is
    then placed), else a new copy."""
    P = draft.participants
    for k, p in P.items():
        if k != key and same_molecule(P[key], p) and (p.compartment is None or p.compartment.db_id == ref.db_id):
            if p.compartment is None:
                _set(p, ref, how)
            return k
    slug = re.sub(r'[^a-z0-9]+', '_', ref.display_name.lower()).strip('_')
    new_key = f'{key}_{slug}'
    n = 2
    while new_key in P:
        new_key, n = f'{key}_{slug}{n}', n + 1
    copy = P[key].model_copy(deep=True)
    copy.key, copy.existing = new_key, None
    _set(copy, ref, how)
    P[new_key] = copy
    return new_key


def _swap(r, old: str, new: str, draft: ReactomeDraft):
    """Reaction `r` uses `new` instead of `old`, and so does any complex it forms that contains `old`."""
    r.inputs = [new if k == old else k for k in r.inputs]
    r.outputs = [new if k == old else k for k in r.outputs]
    if r.catalyst and r.catalyst.entity == old:
        r.catalyst.entity = new
    for g in r.regulations:
        if g.regulator == old:
            g.regulator = new
    if old in r.inputs or new in r.inputs:
        for k in r.outputs:
            p = draft.participants.get(k)
            if isinstance(p, ComplexSpec) and old in p.components:
                p.components = [new if x == old else x for x in p.components]
