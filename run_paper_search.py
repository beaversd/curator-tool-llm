"""Gene -> the papers worth annotating it from: PubMed search, cross-encoder re-rank, curator-rubric judge
(reactome_llm/ReactomeLiteratureExtractor.py, the gene-first pipeline's retrieval). The REST service runs this as a
subprocess (curator_llm/adapters/paper_finder_subprocess.py) because it needs the gene-first pipeline's heavy
dependencies (torch, sentence-transformers, scanpy, faiss), which the service itself does not import.

Usage:
    python run_paper_search.py PINK1                          # print the selection
    python run_paper_search.py PINK1 --max-papers 8 --out results/pink1_papers.json

Progress lines start with "[progress] "; the result is written to --out as JSON.
Exit status 1 when the search fails, including when the judge scored no paper at all (its call failed).
"""
import os, sys, json, argparse
PROJECT_ROOT = os.environ.get('CURATOR_LLM_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import requests

ESUMMARY_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'


def progress(msg):
    print(f'[progress] {msg}', flush=True)


def paper_metadata(pmids):
    """{pmid: {title, journal, year}} from PubMed esummary. The abstract cache the search reads has none of them.
    Best-effort: a failure leaves the fields empty."""
    import PubMedFetcher as f
    if not pmids:
        return {}
    params = {'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'json', 'tool': f.TOOL, 'email': f.EMAIL}
    if f.API_KEY:
        params['api_key'] = f.API_KEY
    try:
        f._throttle()
        r = requests.get(ESUMMARY_URL, params=params, timeout=30)
        r.raise_for_status()
        res = r.json().get('result', {})
    except Exception as e:
        print(f'[warn] esummary failed: {type(e).__name__}: {e}', flush=True)
        return {}
    return {p: {'title': (res.get(p) or {}).get('title', ''),
                'journal': (res.get(p) or {}).get('fulljournalname') or (res.get(p) or {}).get('source', ''),
                'year': ((res.get(p) or {}).get('pubdate') or '')[:4]} for p in pmids}


def pmc_ids(pmids):
    """{pmid: pmcid or None}. Best-effort: on failure every paper is reported as unknown (None)."""
    import PubMedFetcher as f
    try:
        return f.resolve_pmcids(pmids)
    except Exception as e:
        print(f'[warn] PMC id lookup failed: {type(e).__name__}: {e}', flush=True)
        return {}


def search(gene, max_papers):
    from ReactomeLiteratureExtractor import ReactomeLiteratureExtractor
    r = ReactomeLiteratureExtractor().extract(gene, max_papers=max_papers, report=progress)
    per_paper = (r.get('scored') or {}).get('per_paper') or []
    if r.get('candidate_pool') and not any(isinstance(pp.get('score'), (int, float)) for pp in per_paper):
        raise RuntimeError(f"the judge scored none of the {r['candidate_pool']} candidates (its call failed)")
    verdict = {str(pp.get('pmid')): pp for pp in per_paper}
    pmids = [str(p['pmid']) for p in r.get('papers') or []]
    progress('looking up titles and full-text availability')
    meta, pmc = paper_metadata(pmids), pmc_ids(pmids)
    papers = []
    for p in pmids:
        v, m = verdict.get(p, {}), meta.get(p, {})
        papers.append({'pmid': p, 'title': m.get('title', ''), 'journal': m.get('journal', ''),
                       'year': m.get('year', ''), 'pmcid': pmc.get(p),
                       'score': v.get('score'), 'evidence_type': v.get('evidence_type'),
                       'annotatable': v.get('annotatable'), 'justification': v.get('justification') or ''})
    return {'gene': r['gene'], 'accession': r.get('accession'), 'pool_size': r.get('pool_size') or 0,
            'candidate_pool': r.get('candidate_pool') or 0,
            'dropped_below_threshold': r.get('dropped_below_threshold') or 0, 'papers': papers}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Find the papers worth annotating a gene from.')
    ap.add_argument('gene', help='gene symbol, e.g. PINK1')
    ap.add_argument('--max-papers', type=int, default=5)
    ap.add_argument('--out', help='write the result here as JSON')
    args = ap.parse_args(argv)
    result = search(args.gene.strip().upper(), args.max_papers)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(result, fh, indent=2)
    print(f"\n{result['gene']}: {result['pool_size']} found, {result['candidate_pool']} judged, "
          f"{len(result['papers'])} selected", flush=True)
    for p in result['papers']:
        print(f"  {p['pmid']:<10} score={p['score']} {p['evidence_type'] or '':<12} "
              f"{'PMC' if p['pmcid'] else 'no PMC'}  {p['title'][:80]}", flush=True)


if __name__ == '__main__':
    main()
