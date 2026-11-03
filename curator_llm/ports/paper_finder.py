from typing import Callable, List, Optional, Protocol

from pydantic import BaseModel, Field


class CandidatePaper(BaseModel):
    """A paper suggested for a gene, with the curator-rubric judge's verdict on it."""
    pmid: str
    title: str = ''
    journal: str = ''
    year: str = ''
    pmcid: Optional[str] = None                 # None: PMC has no full text, so annotating it needs the PDF uploaded
    score: Optional[int] = None                 # 1-10 annotation usefulness
    evidence_type: Optional[str] = None         # DIRECT | INDIRECT | INSUFFICIENT | NONE
    annotatable: Optional[bool] = None
    justification: str = ''


class GeneSearchResult(BaseModel):
    gene: str
    accession: Optional[str] = None
    pool_size: int = 0                          # papers the PubMed searches returned
    candidate_pool: int = 0                     # of those, how many the judge read
    dropped_below_threshold: int = 0
    papers: List[CandidatePaper] = Field(default_factory=list)   # best first; fewer than asked = little usable literature


class PaperFinder(Protocol):
    def find(self, gene: str, max_papers: int = 5,
             report: Optional[Callable[[str], None]] = None) -> GeneSearchResult:
        """Gene symbol -> the papers worth annotating it from. Raises when the search itself fails.
        `report` (optional) is called with a short progress message as the work advances."""
        ...
