"""PathwayFinder that runs run_pathway_search.py in its own process. The lookup uses reactome_llm/ReactomeUtils.py, which
imports the gene-first pipeline's heavy dependencies (scanpy, faiss, sentence-transformers), so the service never
imports it; the script runs under GENE_SEARCH_PYTHON (default: the service's own interpreter), as the paper search does."""
import json
import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional

from curator_llm.ports.pathway_finder import GenePathwaysResult
from curator_llm.services.extraction import ROOT, _subprocess_run

_PROGRESS = '[progress] '


class SubprocessPathwayFinder:
    def __init__(self, root: str = ROOT, python: Optional[str] = None, timeout: int = 300,
                 run: Callable = _subprocess_run):
        self.root, self.timeout, self._run = root, timeout, run
        self.python = python or os.getenv('GENE_SEARCH_PYTHON') or sys.executable

    def find(self, gene: str, fi_cutoff: float = 0.8, top_partners: int = 10, fdr_cutoff: float = 0.05,
             max_pathways: int = 8, max_evidence: int = 5, summarize: bool = True,
             report: Optional[Callable[[str], None]] = None) -> GenePathwaysResult:
        def on_line(line: str):
            if report and line.startswith(_PROGRESS):
                report(line[len(_PROGRESS):])

        fd, out = tempfile.mkstemp(prefix='pathway_search_', suffix='.json')
        os.close(fd)
        try:
            cmd = [self.python, os.path.join(self.root, 'run_pathway_search.py'), gene,
                   '--fi-cutoff', str(fi_cutoff), '--top-partners', str(top_partners),
                   '--fdr-cutoff', str(fdr_cutoff), '--max-pathways', str(max_pathways),
                   '--max-evidence', str(max_evidence), '--out', out] + ([] if summarize else ['--no-summaries'])
            try:
                proc = self._run(cmd, cwd=self.root, timeout=self.timeout, on_line=on_line)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'pathway search for {gene} took longer than {self.timeout}s')
            if proc.returncode != 0:
                tail = '\n'.join((proc.stdout or '').strip().splitlines()[-5:])
                raise RuntimeError(f'pathway search for {gene} failed (exit {proc.returncode}): {tail}')
            with open(out) as f:
                return GenePathwaysResult.model_validate(json.load(f))
        finally:
            os.unlink(out)
