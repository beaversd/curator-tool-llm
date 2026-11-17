"""Read-only questions about the Reactome graph, for the chat agent.

A few fixed, parameterized queries rather than open Cypher (or a Neo4j MCP server): the model can ask only these,
every answer is small, and nothing can write. The graph is a Reactome release, not gk_central, so a miss does not
prove an instance is absent from gk_central (see adapters/neo4j_lookup.py)."""
from typing import Dict, List, Optional, Protocol

KINDS = ('pathway', 'reaction', 'protein', 'complex', 'set', 'small_molecule', 'drug', 'any')


class ReactomeGraph(Protocol):
    def search_reactome(self, text: str, kind: str = 'any', species: str = 'Homo sapiens',
                        limit: int = 15) -> List[Dict]:
        """Instances whose name contains `text` (case-insensitive), of one kind in KINDS: dbId, stId, displayName,
        schemaClass, species. Exact name matches first."""
        ...

    def reactome_instance(self, ident: str) -> Optional[Dict]:
        """One instance by dbId or stable id (R-HSA-...): its class, name, species, compartments, and by kind the
        inputs/outputs/catalysts/regulators and PMIDs of a reaction, the events of a pathway, the components or
        members of a complex or set, the reference identifier and modifications of an entity, and the pathways an
        event is in. Names with dbIds, so the agent can follow one."""
        ...

    def entities_of(self, identifier: str, limit: int = 25) -> List[Dict]:
        """The physical entities on one reference identifier of any kind: a UniProt accession, ChEBI id, Guide to
        Pharmacology ligand (a drug), PubChem, ENSEMBL or miRBase id. A prefix (CHEBI:, GtoP:, PubChem:, ENSEMBL:,
        UniProt:, miRBase:) picks the database, as a bare number can be several. dbId, stId, displayName, schemaClass,
        database, reference_type, compartment, modified (bool), species, and how many complexes/sets contain it."""
        ...

    def events_of(self, identifier: str, limit: int = 25) -> List[Dict]:
        """Reactions in which an entity on this reference identifier (any kind, prefix as in entities_of) takes part,
        also inside a complex or set: dbId, stId, displayName, roles (input | output | catalyst | regulator), up to
        3 pathways. A drug's reactions are usually its binding to its target, with the drug as a regulator elsewhere."""
        ...
