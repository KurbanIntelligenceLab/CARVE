"""Judge-grounding selection baseline (Section 6, baseline #4): an external
LLM judge scores which of {vanilla answer, Look-more's candidate} is better
visually supported, without retraining (Luo et al., 2025; Uppaal et al.,
2025). Reuses Look-more's already-generated candidate via
`reuse_candidates_from` so the judge's own cost stays separable from
candidate-generation cost.

`run_judge`, `JUDGE_TEMPLATE`, and `SUPPORT_MARGIN` are extracted here
(not imported from archive/controllers/carve_v1.py) because this baseline
is live functionality, not part of the abandoned V1 controller family that
originally happened to define them.
"""
import re
import time

from ._common import gen_candidate, video_id_of, finish

# A candidate must beat vanilla by more than this on the judge's 0-5 support
# scale to replace it. Ties and sub-threshold wins retain vanilla by design.
SUPPORT_MARGIN = 1

JUDGE_MAX_TOKENS = 128  # decoding-budget correction only; see carve_defer_fix history

JUDGE_TEMPLATE = (
    "You are judging which of two candidate answers to a video question is better "
    "supported by the visual evidence quoted beneath it. You do NOT know the correct "
    "answer and must not guess it from prior knowledge - judge ONLY which candidate's "
    "quoted evidence more directly supports its own answer.\n\n"
    "QUESTION:\n{question}\n\nOPTIONS:\n{options}\n\n"
    "--- CANDIDATE A (answer: {ans_a}) ---\nEvidence:\n{ev_a}\n\n"
    "--- CANDIDATE B (answer: {ans_b}) ---\nEvidence:\n{ev_b}\n\n"
    "Rate how directly each candidate's evidence supports its own answer, 0-5:\n"
    "  0 = evidence absent or irrelevant\n"
    "  3 = evidence is related but does not settle the option\n"
    "  5 = evidence directly shows the answer is correct\n"
    "Reply in EXACTLY this format and nothing else:\n"
    "SCORE_A: <0-5>\nSCORE_B: <0-5>\nREASON: <one sentence>\n"
)


def run_judge(manager, dic, ans_a, ev_a, ans_b, ev_b, max_tokens=JUDGE_MAX_TOKENS):
    """Grounding-selection judge. Never receives the gold answer."""
    opts = dic.get('options')
    prompt = JUDGE_TEMPLATE.format(
        question=dic.get('question', ''),
        options='\n'.join(opts) if isinstance(opts, list) else str(opts),
        ans_a=ans_a, ev_a=(ev_a or '(none)')[:4000],
        ans_b=ans_b, ev_b=(ev_b or '(none)')[:4000])
    out = manager.single_text2text([{'role': 'user', 'content': prompt}],
                                   manager.ds_model_name, manager.ds_api_base,
                                   manager.ds_api_keys, max_tokens=max_tokens)
    sa = re.search(r'SCORE_A:\s*([0-5])', out or '')
    sb = re.search(r'SCORE_B:\s*([0-5])', out or '')
    rs = re.search(r'REASON:\s*(.+)', out or '')
    if not sa or not sb:
        return {'judge_valid': False, 'judge_raw': (out or '')[:600],
                'judge_raw_full': (out or ''), 'judge_max_tokens': max_tokens,
                'score_vanilla': None, 'score_candidate': None, 'judge_reason': None}
    return {'judge_valid': True, 'judge_raw_full': (out or ''),
            'judge_max_tokens': max_tokens,
            'score_vanilla': int(sa.group(1)),
            'score_candidate': int(sb.group(1)),
            'judge_reason': rs.group(1).strip()[:400] if rs else None,
            'judge_raw': (out or '')[:600]}


def run(manager, qid, frozen, dic, reuse_candidate):
    """`reuse_candidate` is Look-more's saved row for this qid (must have
    `candidate_answer` and `new_evidence_text` populated)."""
    row = {'question_id': qid, 'video_id': video_id_of(qid, frozen),
           'run_name': 'judge_compute_matched', 'mode': 'judge',
           'gold_answer': dic['answer'][0],
           'vanilla_answer': frozen.get('reference_answer'), 'status': 'ok',
           'candidate_source': 'reused_from_gated_lookmore'}
    row['vanilla_correct'] = row['vanilla_answer'] == row['gold_answer']
    t0 = time.time()

    cand = reuse_candidate.get('candidate_answer')
    new_ev = reuse_candidate.get('new_evidence_text') or ''
    if not cand:
        return finish(row, t0, row['vanilla_answer'], 'missing_candidate')

    ev_old = '\n'.join(str(t)[:800] for t in frozen.get('evidence_slots', {}).values())
    row['judge_evidence_vanilla'] = ev_old[:4000]
    row['judge_evidence_candidate'] = new_ev[:4000]
    manager._reset_stats()
    tj = time.time()
    try:
        j = run_judge(manager, dic, row['vanilla_answer'], ev_old, cand, new_ev)
    except Exception as e:
        row['judge_error'] = f'{type(e).__name__}: {e}'[:250]
        return finish(row, t0, row['vanilla_answer'], 'judge_exception')
    row['judge_seconds'] = round(time.time() - tj, 2)
    stj = getattr(manager, '_stats', {}) or {}
    row['judge_input_tokens'] = stj.get('input_tokens')
    row['judge_output_tokens'] = stj.get('output_tokens')
    row.update(j)
    row['judge_model'] = manager.ds_model_name
    if not j['judge_valid']:
        return finish(row, t0, row['vanilla_answer'], 'judge_invalid')
    margin = j['score_candidate'] - j['score_vanilla']
    row['support_margin'] = margin
    if margin < SUPPORT_MARGIN:
        return finish(row, t0, row['vanilla_answer'], 'tie_or_insufficient_margin')
    return finish(row, t0, cand, 'judge_selected_candidate')
