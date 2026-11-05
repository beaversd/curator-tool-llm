"""Gene -> Reactome pathways: the released human pathways the gene is already annotated in (Neo4j), and the
pathways it is NOT yet in that are enriched among its top functional-interaction partners (FI scores from Mongo; binomial
test, BH FDR), each with its literature evidence: PubMed papers that mention the gene together with the pathway's partners
or the key words of its name, and the abstract sentences that do. The prediction itself is LLM-free; the only LLM step is
an optional summary per predicted pathway (role 'pathway_summary'), written from that pathway's papers alone and citing
them as [PMID:n], so the curator can read what the literature says before opening the papers. The prediction is the gene-first pipeline's own
placement (ReactomeUtils.suggest_pathway_placement), so `confident` means what it means there. Used by Gene2Path in
curator-tool-frontend. The REST service runs this as a subprocess (curator_llm/adapters/pathway_finder_subprocess.py)
because ReactomeUtils imports the gene-first pipeline's heavy dependencies, which the service itself does not.

Usage:
    python run_pathway_search.py PINK1
    python run_pathway_search.py PINK1 --fi-cutoff 0.8 --top-partners 10 --fdr-cutoff 0.05 --max-pathways 8 \
        --max-evidence 5 --out r.json
    python run_pathway_search.py PINK1 --no-summaries      # LLM-free

Progress lines start with "[progress] "; the result is written to --out as JSON. Exit status 1 when a lookup fails.
A gene with no partners or no significant pathway is not a failure: see `status` in the result.
"""
import os, re, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
PROJECT_ROOT = os.environ.get('CURATOR_LLM_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import requests

ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
SEARCH_POOL = 20          # papers fetched per pathway, before ranking them here
SUMMARY_WORKERS = 4       # pathway summaries written at the same time
# Words of a pathway name that say nothing about what it is about, left out of its literature search.
GENERIC_WORDS = {'signaling', 'signalling', 'pathway', 'pathways', 'regulation', 'regulates', 'regulated', 'mediated',
                 'activation', 'activated', 'activity', 'cycle', 'process', 'processes', 'response', 'events', 'event',
                 'role', 'formation', 'processing', 'receptor', 'receptors', 'human', 'cells', 'cell', 'protein', 'proteins', 'other', 'with', 'from',
                 'into', 'through', 'during', 'involved', 'various', 'associated', 'downstream', 'upstream'}


def progress(msg):
    print(f'[progress] {msg}', flush=True)


def annotated_pathways(gene):
    """[{pathway_id, pathway_name}] the gene is annotated in, released human pathways only, by name."""
    import ReactomeNeo4jUtils as neo4jutils
    return [{'pathway_id': int(r['pathway_id']), 'pathway_name': r['pathway']}
            for r in neo4jutils.query_pathways_for_gene(gene)]


def placement(gene, fi_cutoff, top_partners, fdr_cutoff):
    """The gene-first pipeline's placement: status, primary, partners_used (and whether it is confident)."""
    import ReactomeUtils as utils
    p = utils.suggest_pathway_placement(gene, fi_cutoff=fi_cutoff, top_n=top_partners, fdr_cutoff=fdr_cutoff)
    return p, utils.is_confident_placement(p)


def enriched_pathways(partners, fdr_cutoff):
    """Every pathway enriched among `partners` below fdr_cutoff, lowest FDR first. suggest_pathway_placement keeps only
    the top one and its close ties; Gene2Path lists more."""
    import ReactomeUtils as utils
    map_df = utils.map_interactions_in_pathways({p: set() for p in partners})
    if map_df is None or map_df.empty:
        return []
    df = utils.pathway_binomial_enrichment_df(map_df, partners, fdr_cutoff=fdr_cutoff)
    if df is None or df.empty:
        return []
    return [{'pathway_id': int(r['pathway_id']), 'pathway_name': r['pathway_name'], 'fdr': float(r['FDR']),
             'overlap_count': int(r['overlap_count']), 'mapped_genes': list(r['mapped_genes'])}
            for _, r in df.iterrows()]


def _ncbi_params(**kw):
    import PubMedFetcher as f
    params = dict(kw, tool=f.TOOL, email=f.EMAIL)
    if f.API_KEY:
        params['api_key'] = f.API_KEY
    f._throttle()
    return params


def pathway_term_groups(pathway_name, gene, partners):
    """The words of a pathway's name worth searching for, as groups a paper must mention all of: the name, and apart from
    it what is in parentheses (usually an abbreviation). Words are kept when not generic, not the gene or one of its
    partners (those are searched already), and at least 4 characters.
    'Mitochondrial unfolded protein response (UPRmt)' -> [['Mitochondrial', 'unfolded'], ['UPRmt']]."""
    skip = {gene.lower()} | {p.lower() for p in partners}
    seen, groups = set(), []
    for part in (re.sub(r'\([^)]*\)', ' ', pathway_name), *re.findall(r'\(([^)]*)\)', pathway_name)):
        group = []
        for w in re.findall(r'[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9]', part):
            k = w.lower()
            if len(w) >= 4 and k not in GENERIC_WORDS and k not in skip and k not in seen:
                seen.add(k)
                group.append(w)
        if group:
            groups.append(group)
    return groups


def evidence_query(gene, partners, term_groups):
    """PubMed query, title/abstract: the gene AND (any partner OR all the words of any term group)."""
    tiab = lambda t: f'"{t}"[tiab]'
    alts = [tiab(p) for p in partners] + [
        ' AND '.join(map(tiab, g)) if len(g) == 1 else '(' + ' AND '.join(map(tiab, g)) + ')' for g in term_groups]
    return tiab(gene) + (f" AND ({' OR '.join(alts)})" if alts else '')


def pubmed_search(query, retmax=SEARCH_POOL):
    """(total hits, PMIDs by PubMed relevance)."""
    r = requests.get(ESEARCH_URL, params=_ncbi_params(db='pubmed', term=query, retmode='json', retmax=retmax,
                                                      sort='relevance'), timeout=30)
    r.raise_for_status()
    res = r.json().get('esearchresult', {})
    return int(res.get('count') or 0), [str(p) for p in res.get('idlist') or []]


def pubmed_articles(pmids):
    """{pmid: {title, journal, year, abstract}}, in one efetch."""
    if not pmids:
        return {}
    r = requests.get(EFETCH_URL, params=_ncbi_params(db='pubmed', id=','.join(pmids), retmode='xml'), timeout=60)
    r.raise_for_status()
    out = {}
    for art in ET.fromstring(r.content).iter('PubmedArticle'):
        text = lambda path: ' '.join(''.join(e.itertext()).strip() for e in art.findall(path)).strip()
        pmid = text('MedlineCitation/PMID')
        year = text('.//Journal/JournalIssue/PubDate/Year') or text('.//Journal/JournalIssue/PubDate/MedlineDate')[:4]
        out[pmid] = {'title': text('.//Article/ArticleTitle'), 'journal': text('.//Article/Journal/Title'),
                     'year': year, 'abstract': text('.//Article/Abstract/AbstractText')}
    return out


def _mentions(text, names):
    """The names mentioned in text, as whole words, case-insensitive, in the order given."""
    return [n for n in names if re.search(rf'(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])', text, re.I)]


def supporting_sentences(abstract, gene, partners, terms, n=2):
    """Up to n abstract sentences that mention the gene together with a partner or pathway term, most of those first."""
    scored = []
    for i, sent in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z(])', abstract or '')):
        if _mentions(sent, [gene]):
            hits = len(_mentions(sent, partners)) * 2 + len(_mentions(sent, terms))
            if hits:
                scored.append((-hits, i, sent.strip()))
    return [s for _, _, s in sorted(scored)[:n]]


def pathway_evidence(gene, pathway, max_papers):
    """The literature for one predicted pathway: (papers, total PubMed hits, {pmid: abstract} of the papers listed, for
    the summary; the abstracts are not part of the result). Each paper says which of the pathway's
    partners (its FI partners in the pathway) and name terms it mentions, and quotes the sentences that tie them to the
    gene. Papers co-mentioning more partners rank first, then more terms, then PubMed relevance."""
    partners = pathway['mapped_genes']
    groups = pathway_term_groups(pathway['pathway_name'], gene, partners)
    terms = [t for g in groups for t in g]
    total, pmids = pubmed_search(evidence_query(gene, partners, groups))
    articles = pubmed_articles(pmids)
    papers = []
    for rank, pmid in enumerate(pmids):
        a = articles.get(pmid)
        if not a:
            continue
        text = f"{a['title']} {a['abstract']}"
        papers.append((rank, {'pmid': pmid, 'title': a['title'], 'journal': a['journal'], 'year': a['year'],
                              'partners_mentioned': _mentions(text, partners),
                              'terms_mentioned': _mentions(text, terms),
                              'sentences': supporting_sentences(a['abstract'], gene, partners, terms)}))
    papers.sort(key=lambda rp: (-len(rp[1]['partners_mentioned']), -len(rp[1]['terms_mentioned']),
                                -len(rp[1]['sentences']), rp[0]))
    listed = [p for _, p in papers[:max_papers]]
    return listed, total, {p['pmid']: articles[p['pmid']]['abstract'] for p in listed}


def pmc_ids(pmids):
    """{pmid: pmcid or None}. Best-effort: on failure every paper is reported as unknown (None)."""
    import PubMedFetcher as f
    try:
        return f.resolve_pmcids(pmids) if pmids else {}
    except Exception as e:
        print(f'[warn] PMC id lookup failed: {type(e).__name__}: {e}', flush=True)
        return {}


def add_evidence(gene, pathways, max_papers):
    """Fill in each pathway's `evidence`, `evidence_count` and, when its lookup failed, `evidence_error` (in place).
    Returns {pmid: abstract} over all the papers listed, for add_summaries."""
    abstracts = {}
    for p in pathways:
        progress(f"finding the literature for {p['pathway_name']}")
        try:
            p['evidence'], p['evidence_count'], found = pathway_evidence(gene, p, max_papers)
            abstracts.update(found)
        except Exception as e:
            p['evidence'], p['evidence_count'], p['evidence_error'] = [], 0, f'{type(e).__name__}: {e}'
    pmc = pmc_ids(sorted({e['pmid'] for p in pathways for e in p['evidence']}))
    for e in (e for p in pathways for e in p['evidence']):
        e['pmcid'] = pmc.get(e['pmid'])
    return abstracts


def summary_model():
    """The LangChain chat model for the summaries (role 'pathway_summary' in curator_llm/model_settings.py)."""
    from ModelConfig import create_chat_model
    return create_chat_model('pathway_summary', max_tokens=800)


def summary_prompt(gene, pathway, abstracts):
    papers = '\n\n'.join(f"[PMID:{e['pmid']}] {e['title']} ({e['journal']}, {e['year']})\n"
                          f"{abstracts.get(e['pmid']) or '(no abstract)'}" for e in pathway['evidence'])
    return f"""You help a Reactome curator decide whether the gene {gene} belongs in the Reactome pathway \
"{pathway['pathway_name']}". {gene} is not annotated in it yet; it was predicted because its functional-interaction \
partners {', '.join(pathway['mapped_genes'])} are in the pathway.

Below are the PubMed papers found that mention {gene} with those partners or with the pathway's key words. Using ONLY \
these papers, write ONE paragraph of at most 4 sentences (under 100 words) saying what they report about {gene}'s role in this pathway: the molecular \
mechanism (what {gene} binds, modifies, regulates or is regulated by) and the experimental context, when given. Cite \
every claim with the paper's tag exactly as written, e.g. [PMID:12345]. If the papers do not actually tie {gene} to \
this pathway (they only mention the words together), say so plainly in one sentence instead of inventing a link, and \
do not describe each paper. Plain text only: no headings, lists, line breaks or markdown.

{papers}"""


_CITATION = re.compile(r'\s*\[PMID:\s*(\d+)\]')


def _reply_text(content):
    """A chat reply's text: a string, or the text of a list of content blocks."""
    if isinstance(content, str):
        return content
    return ''.join(b.get('text', '') if isinstance(b, dict) else str(b) for b in content or [])


def summarize_pathway(model, gene, pathway, abstracts):
    """The summary text. Citations of papers that are not this pathway's evidence are removed: the curator can only
    check what the page lists."""
    text = _reply_text(model.invoke(summary_prompt(gene, pathway, abstracts)).content).strip()
    listed = {e['pmid'] for e in pathway['evidence']}
    text = _CITATION.sub(lambda m: f' [PMID:{m.group(1)}]' if m.group(1) in listed else '', text)
    return re.sub(r'\s+', ' ', text).strip()           # one paragraph, whatever the model sent


def add_summaries(gene, pathways, abstracts):
    """Fill in `summary` on each pathway with literature, or `summary_error` when it could not be written (in place).
    A pathway without papers gets neither: there is nothing to summarize, and the page says so already."""
    todo = [p for p in pathways if p.get('evidence')]
    if not todo:
        return
    progress(f'summarizing the literature for {len(todo)} pathway{"s" if len(todo) != 1 else ""}')
    try:
        model = summary_model()
    except Exception as e:
        for p in todo:
            p['summary_error'] = f'{type(e).__name__}: {e}'
        return

    def one(p):
        try:
            p['summary'] = summarize_pathway(model, gene, p, abstracts) or None
            if not p['summary']:
                p['summary_error'] = 'the model returned an empty summary'
        except Exception as e:
            p['summary_error'] = f'{type(e).__name__}: {e}'
    with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:
        list(pool.map(one, todo))


def search(gene, fi_cutoff=0.8, top_partners=10, fdr_cutoff=0.05, max_pathways=8, max_evidence=5, summarize=True):
    progress('looking up the pathways it is annotated in')
    annotated = annotated_pathways(gene)
    progress('ranking its functional-interaction partners')
    place, confident = placement(gene, fi_cutoff, top_partners, fdr_cutoff)
    partners = place.get('partners_used') or []
    predicted = []
    if place.get('status') == 'ok' and partners:
        progress(f'testing pathway enrichment over {len(partners)} partners')
        in_already = {a['pathway_id'] for a in annotated}
        predicted = [dict(p, annotated=False) for p in enriched_pathways([p['gene'] for p in partners], fdr_cutoff)
                     if p['pathway_id'] not in in_already][:max_pathways]
        if max_evidence > 0:
            abstracts = add_evidence(gene, predicted, max_evidence)
            if summarize:
                add_summaries(gene, predicted, abstracts)
    return {'gene': gene, 'annotated_pathways': annotated, 'status': place.get('status') or 'ok',
            'confident': bool(confident), 'predicted_pathways': predicted,
            'partners': [{'gene': p['gene'], 'score': float(p['score'])} for p in partners],
            'params': {'fi_cutoff': fi_cutoff, 'top_partners': top_partners, 'fdr_cutoff': fdr_cutoff,
                       'max_pathways': max_pathways, 'max_evidence': max_evidence, 'summarize': summarize}}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Find the Reactome pathways a gene is, or could be, annotated in.')
    ap.add_argument('gene', help='gene symbol, e.g. PINK1')
    ap.add_argument('--fi-cutoff', type=float, default=0.8, help='lowest FI score for a partner (default 0.8)')
    ap.add_argument('--top-partners', type=int, default=10, help='partners used, best FI score first (default 10)')
    ap.add_argument('--fdr-cutoff', type=float, default=0.05, help='highest FDR for a predicted pathway (default 0.05)')
    ap.add_argument('--max-pathways', type=int, default=8,
                    help='predicted pathways listed, those the gene is not in yet (default 8)')
    ap.add_argument('--max-evidence', type=int, default=5, help='papers listed per pathway, 0 for none (default 5)')
    ap.add_argument('--no-summaries', dest='summarize', action='store_false',
                    help="skip the LLM summary of each predicted pathway's literature")
    ap.add_argument('--out', help='write the result here as JSON')
    args = ap.parse_args(argv)
    result = search(args.gene.strip().upper(), args.fi_cutoff, args.top_partners, args.fdr_cutoff, args.max_pathways,
                    args.max_evidence, args.summarize)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(result, fh, indent=2)
    print(f"\n{result['gene']}: annotated in {len(result['annotated_pathways'])} pathways; prediction "
          f"{result['status']}{' (confident)' if result['confident'] else ''}", flush=True)
    for p in result['predicted_pathways']:
        print(f"  FDR {p['fdr']:.2e}  {p['overlap_count']} partners  {p.get('evidence_count', 0)} papers  "
              f"{p['pathway_name']}", flush=True)
        if p.get('summary') or p.get('summary_error'):
            print(f"      {p.get('summary') or 'summary failed: ' + p['summary_error']}", flush=True)
        for e in p.get('evidence') or []:
            print(f"      {e['pmid']:<10} {', '.join(e['partners_mentioned'] + e['terms_mentioned'])}  "
                  f"{e['title'][:70]}", flush=True)


if __name__ == '__main__':
    main()
