"""InstanceLookup backed directly by a Reactome Neo4j graph (read-only).

Much faster than going through ws, and needs no ws login. CAVEAT: the graph configured in .env
(REACTOME_NEO4J_*) is a Reactome release graph, not gk_central: work that curators have not yet
released is not in it, so a lookup miss here does not prove an instance is absent from gk_central.
dbIds of released objects are the same in both. Before committing, resolve against gk_central
(WsInstanceLookup) when unreleased duplicates matter.

Class and attribute names cannot be Cypher parameters, so they are checked against the schema and
a strict identifier pattern before being interpolated; values are always parameters."""
import re
from typing import Dict, List, Optional

from curator_llm.services import schema_index

_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_HUMAN = 'Homo sapiens'
_RETURN = ('RETURN n.dbId AS dbId, n.displayName AS displayName, '
           'coalesce(n.schemaClass, head([l IN labels(n) WHERE l <> "DatabaseObject"])) AS schemaClassName, '
           # reference sequences have no speciesName property: their species is a relationship
           'coalesce(n.speciesName, head([(n)-[:species]->(sp:Species) | sp.displayName])) AS species')


def _check(class_name: str, attribute: Optional[str] = None):
    if not _IDENT.match(class_name) or not schema_index.has_class(class_name):
        raise ValueError(f'unknown class {class_name!r}')
    if attribute is not None and (not _IDENT.match(attribute) or attribute not in schema_index.attributes_of(class_name)):
        raise ValueError(f'{attribute!r} is not an attribute of {class_name}')


def search_query(class_name: str, attribute: str, operand: str, limit: int):
    """(cypher, extra_params) for a single-attribute search. Works on Neo4j 4.4."""
    _check(class_name, attribute)
    is_list = attribute in schema_index.collection_attributes_of(class_name)
    ref = f'n.`{attribute}`'
    test = {'equal': 'toString({x}) = $v',
            'contains': 'toLower(toString({x})) CONTAINS toLower($v)',
            'regex': 'toString({x}) =~ $v'}.get(operand)
    if test is None:
        raise ValueError(f'unsupported operand {operand!r}')
    cond = f'any(x IN {ref} WHERE {test.format(x="x")})' if is_list else test.format(x=ref)
    return (f'MATCH (n:`{class_name}`) WHERE {ref} IS NOT NULL AND {cond} {_RETURN} LIMIT $limit',
            {'limit': int(limit)})


def _row(r) -> Dict:
    return {'dbId': r['dbId'], 'displayName': r['displayName'] or '',
            'schemaClassName': r['schemaClassName'] or '', 'species': r['species']}


class Neo4jInstanceLookup:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None, driver=None):
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver, self.database = driver, database

    @classmethod
    def from_env(cls) -> 'Neo4jInstanceLookup':
        import os
        return cls(os.getenv('REACTOME_NEO4J_URI'), os.getenv('REACTOME_NEO4J_USER'),
                   os.getenv('REACTOME_NEO4J_PWD'), os.getenv('REACTOME_NEO4J_DATABASE'))

    def _run(self, cypher: str, **params) -> List[Dict]:
        with self.driver.session(database=self.database) as s:
            return [_row(r) for r in s.run(cypher, **params)]

    def close(self):
        self.driver.close()

    def find_by_display_name(self, display_name: str, class_names: List[str]) -> Optional[Dict]:
        for c in class_names:
            _check(c)
        rows = self._run(f'MATCH (n:DatabaseObject) WHERE n.displayName = $dn '
                         f'AND any(l IN labels(n) WHERE l IN $classes) {_RETURN} LIMIT 5',
                         dn=display_name, classes=list(class_names))
        human = [r for r in rows if r['species'] == _HUMAN]
        rows = human or rows
        return rows[0] if len(rows) == 1 else None          # several matches: ambiguous, no guess

    def search(self, class_name: str, attribute: str, value: str, operand: str = 'equal',
               limit: int = 5) -> List[Dict]:
        cypher, extra = search_query(class_name, attribute, operand, limit)
        return self._run(cypher, v=str(value), **extra)

    def find_human_accession(self, name: str) -> Optional[str]:
        return self.find_accession(name, _HUMAN)

    def find_accession(self, name: str, species: str = _HUMAN) -> Optional[str]:
        """UniProt accession of the one ReferenceGeneProduct of `species` in the graph known by `name`, or None.

        Tries gene names and synonyms first (PRKN, PARK2), then the UniProt short name kept in the description
        ("shortName: Parkin"). For a species other than human, last, a gene name that starts with `name` when exactly
        one does (a paper's "LANA" is KSHV's LANA1; "EBNA" matches six EBV genes, so no guess). Several different
        accessions is ambiguous: no guess."""
        n = name.strip()
        if not n:
            return None
        base = ('MATCH (n:ReferenceGeneProduct)-[:species]->(:Taxon {displayName: $species}) WHERE n.identifier IS NOT NULL AND ')
        with self.driver.session(database=self.database) as s:
            rows = [dict(r) for r in s.run(
                base + 'any(g IN n.geneName WHERE toLower(g) = toLower($n)) '
                'RETURN DISTINCT n.identifier AS id, toLower(head(n.geneName)) = toLower($n) AS primary',
                species=species, n=n)]
            # a symbol can also be another gene's synonym (SP1 of DAND5, TRAF2 of TANK): the gene it is the primary
            # name of wins; with none, any gene that has it as a synonym
            ids = {r['id'] for r in rows if r['primary']} or {r['id'] for r in rows}
            if not ids:
                short = re.compile(r'shortName: ' + re.escape(n) + r'(?= [A-Za-z]+(?: evidence=|:)|$)', re.I)
                ids = {r['id'] for r in s.run(
                    base + 'any(d IN n.description WHERE toLower(d) CONTAINS toLower($frag)) '
                    'RETURN n.identifier AS id, n.description AS d', species=species, frag=f'shortName: {n}')
                    if any(short.search(d) for d in r['d'])}
            if not ids and species != _HUMAN and len(n) >= 3:
                ids = {r['id'] for r in s.run(
                    base + 'any(g IN n.geneName WHERE toLower(g) STARTS WITH toLower($n)) '
                    'RETURN DISTINCT n.identifier AS id', species=species, n=n)}
        return next(iter(ids)) if len(ids) == 1 else None

    def find_unmodified_entity(self, identifier: str, compartment_db_id: int) -> List[Dict]:
        """The physical entities (protein, gene, RNA or small molecule) whose reference has this identifier (UniProt
        accession, ChEBI number, Ensembl id), in this compartment, without modified residues. Several: the caller does
        not guess."""
        if not identifier:
            return []
        return self._run('MATCH (r:ReferenceEntity {identifier: $i})<-[:referenceEntity]-(n:PhysicalEntity)'
                         '-[:compartment]->(:Compartment {dbId: $c}) WHERE NOT (n)-[:hasModifiedResidue]->() '
                         f'WITH DISTINCT n {_RETURN} LIMIT 5', i=str(identifier), c=int(compartment_db_id))

    def compartments(self, min_uses: int = 50) -> List[Dict]:
        """Reactome's compartments in real use (on at least `min_uses` physical entities), most used first:
        [{dbId, displayName, schemaClassName, uses}]. The vocabulary services/compartments.py maps answers onto."""
        with self.driver.session(database=self.database) as s:
            rows = [dict(r) for r in s.run(
                'MATCH (c:Compartment) OPTIONAL MATCH (e:PhysicalEntity)-[:compartment]->(c) '
                'WITH c, count(e) AS uses WHERE uses >= $m '
                'RETURN c.dbId AS dbId, c.displayName AS displayName, uses ORDER BY uses DESC', m=int(min_uses))]
        return [{**r, 'schemaClassName': 'Compartment'} for r in rows]

    def find_species(self, name: str) -> Optional[Dict]:
        """The Species whose name or synonym is `name`, ignoring case ("KSHV", "HHV8" and "Human herpesvirus 8" are
        one), with its NCBI taxId: {dbId, displayName, schemaClassName, taxId}. None when there is not exactly one."""
        n = name.strip()
        if not n:
            return None
        with self.driver.session(database=self.database) as s:
            rows = [dict(r) for r in s.run(
                'MATCH (n:Species) WHERE toLower(n.displayName) = toLower($n) '
                'OR any(x IN n.name WHERE toLower(x) = toLower($n)) '
                'RETURN DISTINCT n.dbId AS dbId, n.displayName AS displayName, n.taxId AS taxId LIMIT 3', n=n)]
        if len(rows) != 1:
            return None
        return {**rows[0], 'schemaClassName': 'Species'}

    def candidate_genes(self, name: str) -> List[str]:
        """Gene symbols of the members of human Reactome sets named `name` (e.g. "Ub [cytosol]"), in alphabetical order.

        For a name that stands for a family of gene products and so has no single accession."""
        n = name.strip()
        if not n:
            return []
        with self.driver.session(database=self.database) as s:
            return sorted({r['g'] for r in s.run(
                'MATCH (s:DefinedSet)-[:species]->(:Taxon {displayName: $human}) '
                'WHERE toLower(s.displayName) STARTS WITH toLower($prefix) '
                'MATCH (s)-[:hasMember]->(:EntityWithAccessionedSequence)-[:referenceEntity]->(r:ReferenceEntity) '
                'WHERE r.geneName IS NOT NULL RETURN DISTINCT head(r.geneName) AS g', human=_HUMAN, prefix=f'{n} [')})

    def find_by_db_id(self, db_id: int) -> Optional[Dict]:
        rows = self._run(f'MATCH (n:DatabaseObject {{dbId: $id}}) {_RETURN} LIMIT 1', id=int(db_id))
        return rows[0] if rows else None


# ── existing-event lookup (EventLookup port) ────────────────────────────────────────────────────
_DEPRECATED = re.compile(r'^\s*clone of\b|\breplaced by\b|\bdeleted\b', re.I)
_SPAN = '-[:hasComponent|hasMember|hasCandidate*0..]->(:EntityWithAccessionedSequence)-[:referenceEntity]->(r:ReferenceEntity)'
_PARTICIPANTS = f"""
MATCH (e:ReactionLikeEvent) WHERE e.dbId IN $ids
CALL {{ WITH e OPTIONAL MATCH (e)-[:input]->(:PhysicalEntity){_SPAN} RETURN collect(DISTINCT r.identifier) AS ins }}
CALL {{ WITH e OPTIONAL MATCH (e)-[:output]->(:PhysicalEntity){_SPAN} RETURN collect(DISTINCT r.identifier) AS outs }}
CALL {{ WITH e OPTIONAL MATCH (e)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(:PhysicalEntity){_SPAN}
        RETURN collect(DISTINCT r.identifier) AS cats }}
RETURN e.dbId AS dbId, e.displayName AS displayName, e.stId AS stId, e.schemaClass AS schemaClass,
       ins, outs, cats
"""


def _event_rows(driver, database, ids):
    from curator_llm.ports.events import EventRow
    if not ids:
        return []
    with driver.session(database=database) as s:
        rows = [EventRow(r['dbId'], r['displayName'] or '', r['stId'] or '', r['schemaClass'] or '',
                         list(r['ins']), list(r['outs']), list(r['cats']))
                for r in s.run(_PARTICIPANTS, ids=list(ids))]
    return [r for r in rows if not _DEPRECATED.search(r.display_name)]


def events_citing(self, pmid: str):
    with self.driver.session(database=self.database) as s:
        ids = [r['id'] for r in s.run(
            'MATCH (e:ReactionLikeEvent)-[:literatureReference]->(:LiteratureReference {pubMedIdentifier: $p}) '
            'RETURN DISTINCT e.dbId AS id', p=int(pmid))]
    return _event_rows(self.driver, self.database, ids)


def candidate_events(self, catalyst_accessions, participant_accessions, limit: int = 100):
    with self.driver.session(database=self.database) as s:
        if catalyst_accessions:
            q = ('MATCH (e:ReactionLikeEvent)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(:PhysicalEntity)'
                 + _SPAN + ' WHERE r.identifier IN $accs RETURN DISTINCT e.dbId AS id LIMIT $limit')
            ids = [r['id'] for r in s.run(q, accs=list(catalyst_accessions), limit=int(limit))]
        elif participant_accessions:
            q = ('MATCH (e:ReactionLikeEvent)-[:input|output]->(:PhysicalEntity)' + _SPAN +
                 ' WHERE r.identifier IN $accs WITH e, count(DISTINCT r.identifier) AS k '
                 'ORDER BY k DESC LIMIT $limit RETURN e.dbId AS id')
            ids = [r['id'] for r in s.run(q, accs=list(participant_accessions), limit=int(limit))]
        else:
            ids = []
    return _event_rows(self.driver, self.database, ids)


Neo4jInstanceLookup.events_citing = events_citing
Neo4jInstanceLookup.candidate_events = candidate_events


# ── read-only browsing for the chat agent (ReactomeGraph port, ports/graph.py) ─────────────────────
_KIND_LABEL = {'pathway': 'Pathway', 'reaction': 'ReactionLikeEvent', 'protein': 'EntityWithAccessionedSequence',
               'complex': 'Complex', 'set': 'EntitySet', 'small_molecule': 'SimpleEntity', 'drug': 'Drug',
               'any': 'DatabaseObject'}
_ID = 'n.dbId AS dbId, n.stId AS stId, n.displayName AS displayName, n.schemaClass AS schemaClass'
_NAMED = '{dbId: x.dbId, name: x.displayName}'


# a prefix names the reference database (a bare number can be a ChEBI, Guide to Pharmacology or PubChem id)
_DB_PREFIX = {'chebi': 'ChEBI', 'gtop': 'Guide to Pharmacology', 'iuphar': 'Guide to Pharmacology',
              'guidetopharmacology': 'Guide to Pharmacology', 'pubchem': 'PubChem', 'ensembl': 'ENSEMBL',
              'uniprot': 'UniProt', 'mirbase': 'miRBase', 'ncbigene': 'NCBI Gene', 'kegg': 'KEGG'}
# the reference's database: its databaseName, else its referenceDatabase (most UniProt entries carry only the latter)
_REF_DB = 'coalesce(x.databaseName, [(x)-[:referenceDatabase]->(d) | d.displayName][0])'
_REF_WHERE = f'x.identifier = $acc AND ($db IS NULL OR toLower({_REF_DB}) STARTS WITH toLower($db))'


def _ref_id(identifier: str) -> str:
    """'CHEBI:15422' -> '15422', 'GtoP:8708' -> '8708' (Reactome stores ids without the prefix); others unchanged."""
    return _ref_parts(identifier)[0]


def _ref_parts(identifier: str):
    """(identifier without a known prefix, the database it names or None)."""
    s = (identifier or '').strip()
    head, sep, rest = s.partition(':')
    db = _DB_PREFIX.get(re.sub(r'[^a-z]', '', head.lower())) if sep else None
    return (rest.strip(), db) if db else (s, None)


def _clean(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if v not in (None, [], '')}


def search_reactome(self, text: str, kind: str = 'any', species: str = 'Homo sapiens', limit: int = 15) -> List[Dict]:
    label = _KIND_LABEL.get(kind)
    if label is None:
        raise ValueError(f'kind must be one of {sorted(_KIND_LABEL)}')
    where = 'toLower(n.displayName) CONTAINS toLower($t)'
    if kind == 'any':
        where += ' AND (n:Event OR n:PhysicalEntity)'
    if species:
        where += ' AND (n.speciesName IS NULL OR n.speciesName = $sp)'
    q = (f'MATCH (n:`{label}`) WHERE {where} '
         f'WITH n, toLower(n.displayName) AS low, toLower($t) AS t '
         f'RETURN {_ID}, n.speciesName AS species '
         # the name itself, then a name starting with it ("PINK1 [cytosol]"), then a word in it, then anything
         'ORDER BY CASE WHEN low = t THEN 0 WHEN low STARTS WITH t THEN 1 '
         "WHEN (' ' + low) CONTAINS (' ' + t) OR low CONTAINS ('-' + t) OR low CONTAINS (':' + t) THEN 2 ELSE 3 END, "
         'size(n.displayName) LIMIT $limit')
    with self.driver.session(database=self.database) as s:
        rows = [_clean({k: r[k] for k in ('dbId', 'stId', 'displayName', 'schemaClass', 'species')})
                for r in s.run(q, t=text.strip(), sp=species, limit=int(min(limit, 50)) + 10)]
    return [r for r in rows if not _DEPRECATED.search(r.get('displayName', ''))][:limit]


_INSTANCE = f"""
MATCH (n:DatabaseObject) WHERE n.dbId = $id OR n.stId = $st
WITH n LIMIT 1
RETURN {_ID}, n.speciesName AS species,
  [(n)-[:compartment]->(x) | x.displayName] AS compartments,
  [(n)-[:input]->(x) | {_NAMED}] AS inputs,
  [(n)-[:output]->(x) | {_NAMED}] AS outputs,
  [(n)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(x) | {_NAMED}] AS catalysts,
  [(n)-[:regulatedBy]->(g)-[:regulator]->(x) | {{dbId: x.dbId, name: x.displayName, kind: g.schemaClass}}] AS regulators,
  [(n)-[:literatureReference]->(x) WHERE x.pubMedIdentifier IS NOT NULL | x.pubMedIdentifier] AS pmids,
  [(n)-[:hasEvent]->(x) | {_NAMED}][..60] AS events,
  [(p:Pathway)-[:hasEvent]->(n) | {{dbId: p.dbId, name: p.displayName}}] AS pathways,
  [(n)-[:hasComponent]->(x) | {_NAMED}] AS components,
  [(n)-[:hasMember|hasCandidate]->(x) | {_NAMED}] AS members,
  [(n)-[:referenceEntity]->(x) | x.databaseName + ':' + x.identifier][0] AS reference,
  [(n)-[:hasModifiedResidue]->(x) | x.displayName] AS modifications,
  [(n)-[:summation]->(x) | x.text][0] AS summary
"""


def reactome_instance(self, ident: str) -> Optional[Dict]:
    ident = str(ident).strip()
    db_id = int(ident) if ident.isdigit() else -1
    with self.driver.session(database=self.database) as s:
        r = s.run(_INSTANCE, id=db_id, st=ident).single()
    if r is None:
        return None
    out = _clean(dict(r))
    for k in ('events', 'pathways', 'inputs', 'outputs', 'catalysts', 'regulators', 'components', 'members'):
        if k in out:
            out[k] = [x for x in out[k] if not _DEPRECATED.search(x.get('name') or '')]
    if out.get('summary') and len(out['summary']) > 800:
        out['summary'] = out['summary'][:800] + '...'
    return out


def entities_of(self, identifier: str, limit: int = 25) -> List[Dict]:
    q = f"""
    MATCH (n:PhysicalEntity)-[:referenceEntity]->(x:ReferenceEntity) WHERE {_REF_WHERE}
    RETURN n.dbId AS dbId, n.stId AS stId, n.displayName AS displayName, n.schemaClass AS schemaClass,
           n.speciesName AS species, {_REF_DB} AS database, x.schemaClass AS reference_type,
           [(n)-[:compartment]->(c) | c.displayName][0] AS compartment,
           exists((n)-[:hasModifiedResidue]->()) AS modified,
           size([(x)-[:hasComponent|hasMember|hasCandidate]->(n) | x]) AS in_complexes_or_sets
    ORDER BY modified, in_complexes_or_sets DESC LIMIT $limit"""
    acc, db = _ref_parts(identifier)
    with self.driver.session(database=self.database) as s:
        return [_clean(dict(r)) | {'modified': r['modified']} for r in s.run(q, acc=acc, db=db, limit=int(min(limit, 50)))]


def events_of(self, identifier: str, limit: int = 25) -> List[Dict]:
    q = f"""
    MATCH (pe:PhysicalEntity)-[:referenceEntity]->(x:ReferenceEntity) WHERE {_REF_WHERE}
    WITH DISTINCT pe
    MATCH (x:PhysicalEntity)-[:hasComponent|hasMember|hasCandidate*0..4]->(pe)
    WITH DISTINCT x
    CALL {{
      WITH x
      MATCH (e:ReactionLikeEvent)-[:input]->(x) RETURN e, 'input' AS role
      UNION WITH x MATCH (e:ReactionLikeEvent)-[:output]->(x) RETURN e, 'output' AS role
      UNION WITH x MATCH (e:ReactionLikeEvent)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(x)
            RETURN e, 'catalyst' AS role
      UNION WITH x MATCH (e:ReactionLikeEvent)-[:regulatedBy]->()-[:regulator]->(x) RETURN e, 'regulator' AS role
    }}
    WITH e, collect(DISTINCT role) AS roles
    RETURN e.dbId AS dbId, e.stId AS stId, e.displayName AS displayName, roles,
           [(p:Pathway)-[:hasEvent]->(e) WHERE NOT p.displayName =~ '(?i).*(clone of|replaced by).*' | p.displayName][..3]
             AS pathways
    ORDER BY size(roles) DESC, e.displayName LIMIT $limit"""
    acc, db = _ref_parts(identifier)
    with self.driver.session(database=self.database) as s:
        rows = [_clean(dict(r)) for r in s.run(q, acc=acc, db=db, limit=int(min(limit, 50)))]
    return [r for r in rows if not _DEPRECATED.search(r.get('displayName', ''))]


def reference_names(self, identifier: str) -> List[str]:
    """The names Reactome holds for a UniProt accession or ChEBI id (ReferenceEntity name and geneName lists)."""
    q = (f'MATCH (x:ReferenceEntity) WHERE {_REF_WHERE} '
         'RETURN coalesce(x.name, []) + coalesce(x.geneName, []) AS names LIMIT 1')
    acc, db = _ref_parts(identifier)
    with self.driver.session(database=self.database) as s:
        r = s.run(q, acc=acc, db=db).single()
    return list(dict.fromkeys(r['names'])) if r else []


Neo4jInstanceLookup.reference_names = reference_names
Neo4jInstanceLookup.search_reactome = search_reactome
Neo4jInstanceLookup.reactome_instance = reactome_instance
Neo4jInstanceLookup.entities_of = entities_of
Neo4jInstanceLookup.events_of = events_of
