"""Probe: can a TypeSafe Noul stand in for the Claude pairwise duplicate judge in run_merge.py?

Builds the same same-source pairs run_merge.py stage 1b judges, asks TypeSafe whether each pair
is the same biochemical event (plus three narrower sub-judgments), and writes a CSV for review.
Nothing in the pipeline is changed.

    export TYPESAFE_API_KEY=...
    python experiments/typesafe_dup_probe.py results/pink1_extraction.json --limit 40

Pairs with an exact name match or a shared participant are sampled first, since those are the
hard ones; the rest are mostly obvious "no".
"""
import argparse
import csv
import json
import os
import sys
import time

import requests

API_URL = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-latest'

# --- questions: the part worth reviewing. Condensed from reactions_are_duplicate() in run_merge.py.
QUESTIONS = {
    'same_event': {
        'type': 'noul',
        'instructions': {
            'question': 'Do `reaction_a` and `reaction_b` describe the same underlying biochemical '
                        'event, as a Reactome curator deduplicating extracted reactions would judge it?',
            'same_despite': [
                'different wording, synonyms or level of detail',
                'a different reactionType label (blackBoxEvent vs transition)',
                'a domain or subunit vs its parent protein named as catalyst',
                'one compartment merely containing the other',
                'the same core transformation observed under two experimental conditions',
            ],
            'different_if': [
                'the inputs or outputs differ',
                'the catalyst differs, including X as catalyst in one and only as regulator in the other',
                'the same regulator has opposite regulationType',
                'the molecular action differs (e.g. phosphorylation vs ubiquitination)',
                'the compartments are genuinely different locations',
                'they merely share one participant',
            ],
        },
        'criteria': {'true': 'Same event; should be merged',
                     'false': 'Different events; must stay separate'},
    },
    'same_participants': {
        'type': 'noul',
        'instructions': 'Do `reaction_a` and `reaction_b` have the same inputs and outputs, '
                        'allowing for synonyms and differing detail?',
    },
    'same_catalyst': {
        'type': 'noul',
        'instructions': 'Do `reaction_a` and `reaction_b` name the same catalyst (or both none)? '
                        'A domain or subunit counts as its parent protein; a regulator is not a catalyst.',
    },
    'same_action': {
        'type': 'noul',
        'instructions': 'Do `reaction_a` and `reaction_b` describe the same molecular action '
                        '(e.g. both phosphorylation, both binding)?',
    },
}

# Noul values in this band are what run_merge.py would currently re-vote on.
UNSURE_LO, UNSURE_HI = 0.2, 0.8


def _norm(x):
    return (x or '').strip().lower()


def _sig(a):
    return (_norm(a.get('name')),
            tuple(sorted(_norm(i) for i in (a.get('input') or []))),
            tuple(sorted(_norm(o) for o in (a.get('output') or []))))


def _state_record(a):
    ca = a.get('catalystActivity') or {}
    return {
        'name': a.get('name'),
        'reactionType': a.get('reactionType'),
        'input': a.get('input') or [],
        'output': a.get('output') or [],
        'catalyst': ca.get('catalyst'),
        'molecularFunction': ca.get('molecularFunction'),
        'regulatedBy': a.get('regulatedBy') or [],
        'condition': a.get('condition'),
        'compartment': a.get('compartment'),
        'relationships': a.get('relationships') or [],
    }


def _hard_first(a, b):
    """Sort key: pairs sharing a participant come first."""
    pa = {_norm(x) for x in (a.get('input') or []) + (a.get('output') or [])}
    pb = {_norm(x) for x in (b.get('input') or []) + (b.get('output') or [])}
    return -len(pa & pb)


def ask(session, state):
    for attempt in range(4):
        r = session.post(API_URL, json={'state': state, 'model': MODEL, 'questions': QUESTIONS},
                         timeout=60)
        if r.status_code in (429, 529):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('extraction', help='an *_extraction.json from run_extraction.py')
    p.add_argument('--limit', type=int, default=40, help='max pairs to send (default 40)')
    p.add_argument('--out', default=None, help='CSV path (default: next to input)')
    args = p.parse_args()

    key = os.environ.get('TYPESAFE_API_KEY')
    if not key:
        sys.exit('[error] TYPESAFE_API_KEY is not set')

    seen, unique = set(), []
    for e in json.load(open(args.extraction)):
        a = e.get('annotation_result') or {}
        s = _sig(a)
        if s not in seen:
            seen.add(s)
            unique.append(e)

    pairs = [(i, j) for i in range(len(unique)) for j in range(i + 1, len(unique))
             if unique[i]['source'] == unique[j]['source']]
    pairs.sort(key=lambda ij: _hard_first(unique[ij[0]]['annotation_result'],
                                          unique[ij[1]]['annotation_result']))
    pairs = pairs[:args.limit]
    print(f'[probe] {len(unique)} unique reactions, sending {len(pairs)} pairs', flush=True)

    out = args.out or os.path.splitext(args.extraction)[0] + '_typesafe_probe.csv'
    session = requests.Session()
    session.headers['Authorization'] = f'Bearer {key}'
    rows, tokens, t0 = [], 0, time.time()
    for n, (i, j) in enumerate(pairs, 1):
        a, b = unique[i]['annotation_result'], unique[j]['annotation_result']
        resp = ask(session, {'reaction_a': _state_record(a), 'reaction_b': _state_record(b)})
        ans = resp['answers']
        tokens += resp['usage']['input_tokens'] + resp['usage']['output_tokens']
        row = {'i': i, 'j': j, 'name_a': a.get('name'), 'name_b': b.get('name')}
        row.update({q: round(ans[q]['noul'], 3) for q in QUESTIONS})
        row['unsure'] = UNSURE_LO < row['same_event'] < UNSURE_HI
        rows.append(row)
        print(f"  {n:3d}. same={row['same_event']:.2f}  {row['name_a'][:45]!r} vs {row['name_b'][:45]!r}",
              flush=True)

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ['curator_same'])
        w.writeheader()
        w.writerows(rows)

    same = sum(r['same_event'] >= 0.5 for r in rows)
    unsure = sum(r['unsure'] for r in rows)
    print(f'[probe] {same}/{len(rows)} judged same, {unsure} in the {UNSURE_LO}-{UNSURE_HI} band, '
          f'{tokens} tokens, {time.time() - t0:.0f}s -> {out}')
    print('[probe] fill the curator_same column (y/n) to score agreement.')


if __name__ == '__main__':
    main()
