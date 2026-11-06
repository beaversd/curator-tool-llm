from typing import Callable, List, Optional, Protocol

from pydantic import BaseModel, Field


class AnnotatedPathway(BaseModel):
    """A released human Reactome pathway the gene already takes part in."""
    pathway_id: int                             # dbId
    pathway_name: str


class FiPartner(BaseModel):
    """A functional-interaction partner of the gene, with its FI score (0-1)."""
    gene: str
    score: float


class PathwayEvidence(BaseModel):
    """A PubMed paper mentioning the gene together with a predicted pathway's partners or name terms."""
    pmid: str
    title: str = ''
    journal: str = ''
    year: str = ''
    pmcid: Optional[str] = None                 # set when PMC has its full text
    partners_mentioned: List[str] = Field(default_factory=list)   # the pathway's partners named in title/abstract
    terms_mentioned: List[str] = Field(default_factory=list)      # key words of the pathway name found there
    sentences: List[str] = Field(default_factory=list)  # abstract sentences naming the gene with a partner/term


class PartnerEvidence(BaseModel):
    """The papers naming the gene with one of a predicted pathway's partners (PubMed: gene AND partner, title/abstract)."""
    partner: str
    count: int = 0                              # PubMed hits; `papers` holds the best few
    papers: List[PathwayEvidence] = Field(default_factory=list)   # a sentence tying the two first, then relevance
    error: Optional[str] = None                 # the search failed for this partner


class PredictedPathway(BaseModel):
    """A pathway the gene is not annotated in yet, enriched among its top FI partners (binomial test,
    Benjamini-Hochberg FDR), with the literature that links the gene to it."""
    pathway_id: int                             # dbId
    pathway_name: str
    fdr: float
    overlap_count: int                          # how many of the partners are in the pathway
    mapped_genes: List[str] = Field(default_factory=list)   # which ones
    annotated: bool = False                     # always False: pathways the gene is already in are left out
    evidence: List[PathwayEvidence] = Field(default_factory=list)  # best first
    evidence_count: int = 0                     # PubMed hits for the evidence query (evidence holds the top few)
    evidence_error: Optional[str] = None        # the literature lookup failed for this pathway
    partner_evidence: List[PartnerEvidence] = Field(default_factory=list)  # one per mapped_genes partner, same order
    partner_evidence_error: Optional[str] = None
    # LLM summary of what `evidence` says about the gene in this pathway, citing it as [PMID:n]. None when the pathway
    # has no papers, or the search ran without summaries; summary_error says why one could not be written.
    summary: Optional[str] = None
    summary_error: Optional[str] = None


class GenePathwaysResult(BaseModel):
    gene: str
    annotated_pathways: List[AnnotatedPathway] = Field(default_factory=list)
    # ok | no_partners (no FI partner above fi_cutoff) | no_significant_pathway (none below fdr_cutoff)
    status: str = 'ok'
    # Whether the top prediction is strong enough for the gene-first pipeline to anchor on
    # (ReactomeUtils.is_confident_placement: FDR < 1e-3 with at least 2 partners).
    confident: bool = False
    predicted_pathways: List[PredictedPathway] = Field(default_factory=list)   # not annotated; lowest FDR first
    partners: List[FiPartner] = Field(default_factory=list)                    # the partners used, best first
    params: dict = Field(default_factory=dict)


class PathwayFinder(Protocol):
    def find(self, gene: str, fi_cutoff: float = 0.8, top_partners: int = 10, fdr_cutoff: float = 0.05,
             max_pathways: int = 8, max_evidence: int = 5, summarize: bool = True,
             report: Optional[Callable[[str], None]] = None) -> GenePathwaysResult:
        """Gene symbol -> the Reactome pathways it is annotated in, and those its FI partners predict that it is not in
        yet, each with up to max_evidence supporting papers and, with `summarize`, an LLM summary of them. The
        prediction itself is LLM-free.
        Raises when the lookup itself fails. `report` (optional) is called with a short progress message."""
        ...
