"""PaperFinder that runs run_paper_search.py in its own process. The search needs the gene-first pipeline's heavy
dependencies (torch, sentence-transformers, scanpy, faiss), so the service never imports it; the script runs under
GENE_SEARCH_PYTHON (default: the service's own interpreter), which must have requirements.txt installed."""
import json
import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional

from curator_llm.ports.paper_finder import GeneSearchResult
from curator_llm.services.extraction import ROOT, _subprocess_run

_PROGRESS = '[progress] '


class SubprocessPaperFinder:
    def __init__(self, root: str = ROOT, python: Optional[str] = None, timeout: int = 900,
                 run: Callable = _subprocess_run):
        self.root, self.timeout, self._run = root, timeout, run
        self.python = python or os.getenv('GENE_SEARCH_PYTHON') or sys.executable

    def find(self, gene: str, max_papers: int = 5,
             report: Optional[Callable[[str], None]] = None) -> GeneSearchResult:
        def on_line(line: str):
            if report and line.startswith(_PROGRESS):
                report(line[len(_PROGRESS):])

        fd, out = tempfile.mkstemp(prefix='gene_search_', suffix='.json')
        os.close(fd)
        try:
            cmd = [self.python, os.path.join(self.root, 'run_paper_search.py'), gene,
                   '--max-papers', str(max_papers), '--out', out]
            try:
                proc = self._run(cmd, cwd=self.root, timeout=self.timeout, on_line=on_line)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'paper search for {gene} took longer than {self.timeout}s')
            if proc.returncode != 0:
                tail = '\n'.join((proc.stdout or '').strip().splitlines()[-5:])
                raise RuntimeError(f'paper search for {gene} failed (exit {proc.returncode}): {tail}')
            with open(out) as f:
                return GeneSearchResult.model_validate(json.load(f))
        finally:
            os.unlink(out)
