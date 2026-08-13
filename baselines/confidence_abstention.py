"""Confidence-threshold abstention baseline (Section 6 / E4, Table 11):
answer only when a confidence score clears a threshold, otherwise abstain.
Compared against CARVE's Delta score as a selective risk-coverage signal
(Whitehead et al., 2022).

Two scoring protocols were used across the two benchmarks and are both
preserved here as separate score functions over the same generic sweep/
risk-coverage machinery, rather than picking one and silently discarding
the other's behavior:

  - `vote_margin_score` (LVBench): margin between the top two hard-vote
    answer counts over a self-consistency pool. Ported from
    analysis/carve_scripts/confidence_threshold_abstention.py.
  - `vanilla_logprob_score` (VideoMME): p(y_vanilla | frozen prefix), the
    planner's own token probability for the answer it actually gave.

Read-only: consumes already-saved model outputs, makes zero new model
calls, alters nothing.
"""
import json
import glob


# ---------------------------------------------------------------------------
# LVBench: vote-margin scoring
# ---------------------------------------------------------------------------
def load_vote_margin_questions(results_glob):
    files = [f for f in glob.glob(results_glob)
             if '.metadata' not in f and '.failures' not in f]
    questions = []
    for fn in files:
        data = json.load(open(fn, encoding='utf-8'))
        for q in data:
            questions.append({
                'question_id': q.get('question_id'),
                'vote_margin': q.get('vote_margin'),
                'hard_vote_winner': q.get('hard_vote_winner'),
                'hard_vote_correct': q.get('hard_vote_correct'),
            })
    return questions


def vote_margin_sweep(questions):
    """Every distinct observed margin is a candidate threshold - the full
    coverage/accuracy curve at native resolution, not an arbitrary grid.
    Does NOT select a single "final" threshold; that's a methodology
    decision applied downstream (see paper/table11_selective_risk_coverage.py)."""
    total = len(questions)
    thresholds = sorted(set(q['vote_margin'] for q in questions if q['vote_margin'] is not None))
    thresholds = [0.0] + thresholds + [max(thresholds) + 0.01] if thresholds else [0.0]

    rows = []
    for t in sorted(set(thresholds)):
        answered = [q for q in questions
                    if q['hard_vote_winner'] is not None and (q['vote_margin'] or 0.0) >= t]
        n_answered = len(answered)
        n_correct = sum(1 for q in answered if q['hard_vote_correct'])
        rows.append({
            'threshold': t,
            'coverage': n_answered / total if total else 0.0,
            'n_answered': n_answered,
            'n_total': total,
            'accuracy_when_answering': n_correct / n_answered if n_answered else None,
            'accuracy_treating_abstain_as_wrong': n_correct / total if total else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# VideoMME: vanilla-answer logprob scoring
# ---------------------------------------------------------------------------
def vanilla_logprob_score(conf_rows):
    """conf_rows: qid -> {'confidence': p(y_vanilla), 'status': 'ok'|...}.
    Returns qid -> float, omitting anything not status == 'ok'."""
    return {qid: r['confidence'] for qid, r in conf_rows.items() if r.get('status') == 'ok'}


# ---------------------------------------------------------------------------
# Shared: generic risk-coverage curve over any qid -> score mapping.
# Identical integration rule to paper/table11_selective_risk_coverage.py so
# both consumers of "risk-coverage" in this repo agree on the definition.
# ---------------------------------------------------------------------------
def risk_coverage_curve(qids, correct, score, seed, coverage_levels=(1.0, 0.9, 0.7)):
    import random
    n = len(qids)
    rnd = random.Random(seed)
    order = sorted(qids, key=lambda q: (-score.get(q, float('-inf')), rnd.random()))
    corr_seq = [correct[q] for q in order]
    accs = {}
    for cov in coverage_levels:
        k = max(1, round(cov * n))
        accs[cov] = sum(corr_seq[:k]) / k
    cum, risks = 0, []
    for i, c in enumerate(corr_seq, 1):
        cum += c
        risks.append(1 - cum / i)
    covs = [i / n for i in range(1, n + 1)]
    full_auc = sum((risks[i] + risks[i - 1]) / 2 * (covs[i] - covs[i - 1])
                   for i in range(1, len(covs)))
    return accs, full_auc, risks, covs
