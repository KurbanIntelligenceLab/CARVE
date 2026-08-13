"""Look-more baseline (Section 6, baseline #2): issue an additional
evidence-gathering rollout, gated on the wrapped agent's own evidence gate
(>=1 novel tool call before it may answer), WITHOUT ever consulting Delta.
Matched to CARVE in the number of additional agent rollouts.

This is the comparison that isolates whether a re-routing gain comes from
score-targeted re-grounding (CARVE) or merely from spending more compute
(this baseline) - see Section 6, "The compute-matched look-more control
tests whether any rerouting gain arises from score-targeted re-grounding or
merely from an additional rollout."
"""
import time

from ._common import gen_candidate, video_id_of, finish, assert_neutral, LOOKMORE_INSTRUCTION


def run(manager, qid, frozen, dic, budget, max_ds_round, gate=True, max_gate_retries=1):
    row = {'question_id': qid, 'video_id': video_id_of(qid, frozen),
           'run_name': 'lookmore_compute_matched', 'mode': 'lookmore',
           'gold_answer': dic['answer'][0],
           'vanilla_answer': frozen.get('reference_answer'), 'status': 'ok'}
    row['vanilla_correct'] = row['vanilla_answer'] == row['gold_answer']
    t0 = time.time()

    cand, _ = gen_candidate(manager, frozen, dic, budget, max_ds_round, row, gate,
                             max_gate_retries)
    if row.get('status') != 'ok':
        return finish(row, t0, row['vanilla_answer'], 'continuation_failed')
    if not cand:
        return finish(row, t0, row['vanilla_answer'], 'invalid_candidate')
    return finish(row, t0, cand, 'lookmore_replaced')
