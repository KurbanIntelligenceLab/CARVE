"""Self-consistency baseline (Section 6, baseline #3): sample complete
agent trajectories and return the modal answer, with the number of
trajectories chosen to match CARVE's mean token and model-call budget
(Appendix C - k' set so total tool calls and tokens are equal in
expectation).

Reuses `carve.probe.run_job`'s SELF condition (no masking, no captioning -
"just rerun the answering phase") for the additional draws, so nothing here
can collide with or alter the frozen Delta: this baseline and the CARVE
probe read/write disjoint repetition indices under the same
`results/<qid>/SELF/<rep>.json` layout.
"""
from collections import Counter

from carve.probe import run_job


def draw_additional_self_replays(manager, frozen, qid, reps, existing_rep0=None):
    """Run SELF-condition replays for `reps` (e.g. [1, 2] to add to an
    existing rep 0), returning the list of parsed answers including
    `existing_rep0` if given."""
    answers = [existing_rep0] if existing_rep0 is not None else []
    for rep in reps:
        row = run_job(manager, frozen, 'SELF', 'self', rep, manager.planner_replica_suffix)
        answers.append(row.get('answer'))
    return answers


def majority_vote(answers, vanilla_answer):
    """Modal answer over k draws; ties resolved in favor of the vanilla
    answer (the first draw), matching the k=3 majority-vote protocol used
    for both benchmarks."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return vanilla_answer
    counts = Counter(valid)
    top = counts.most_common()
    best_count = top[0][1]
    tied = [a for a, c in top if c == best_count]
    return vanilla_answer if vanilla_answer in tied else tied[0]
