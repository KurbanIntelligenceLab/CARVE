"""Section 4: the evidence-dependence score.

    delta_hat_c(q, E) = (1/k) * sum_j 1[a_c^(j) != a_hat]          (Eq. 4)
    Delta_hat(q, E)   = delta_hat_destroy(q, E) - delta_hat_sham(q, E)   (Eq. 5)

k is the number of independently regenerated caption-and-answer replays
PER CONDITION, not the number of mask families - see Section 4's own
clarification. The paper's frozen protocol uses k=3; the VideoMME
production run (Table 7, E2) actually ran k=1 (confirmed against the raw
Phase-B replay counts: SHAM/DESTROY each had exactly 1 replay/question,
only the SELF diagnostic got k=3) - see docs/REPRODUCING.md for why this
matters and what it means for Table 9/10's stated ablation framing.

This module takes flip indicators (0/1, already computed by carve/probe.py
from a completed replay) rather than re-deriving them, so it has no model
or filesystem dependency and is trivially unit-testable.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ConditionFlips:
    """Flip indicators (1 = answer changed vs. the original a_hat, 0 = did
    not, None = invalid/unparseable replay - excluded from delta_hat, not
    counted as 0) for one condition's k replays."""
    flips: Sequence[Optional[int]] = field(default_factory=list)

    def valid(self):
        return [f for f in self.flips if f is not None]

    def delta_hat(self):
        """Eq. 4. Returns None if there are zero valid replays (undefined,
        not 0.0 - a question with no valid replay contributes no signal,
        it should never be silently treated as evidence-dependent)."""
        v = self.valid()
        return (sum(v) / len(v)) if v else None


@dataclass
class DeltaResult:
    question_id: str
    sham: ConditionFlips
    destroy: ConditionFlips
    self_: Optional[ConditionFlips] = None  # diagnostic only, never subtracted - see note below

    @property
    def delta_hat_sham(self):
        return self.sham.delta_hat()

    @property
    def delta_hat_destroy(self):
        return self.destroy.delta_hat()

    @property
    def delta(self):
        """Eq. 5. None (not a triggerable score) if either condition has
        zero valid replays."""
        s, d = self.delta_hat_sham, self.delta_hat_destroy
        return None if (s is None or d is None) else (d - s)


def compute_delta(question_id, sham_flips, destroy_flips, self_flips=None):
    """Convenience constructor matching the compact per-question record
    schema in data/ (sham_flips, destroy_flips, delta)."""
    return DeltaResult(
        question_id=question_id,
        sham=ConditionFlips(sham_flips),
        destroy=ConditionFlips(destroy_flips),
        self_=ConditionFlips(self_flips) if self_flips is not None else None,
    )


# ---------------------------------------------------------------------------
# NOTE ON THE SELF/RANDOM-MASK CONTROL DISCREPANCY (Figure 2 vs. Eq. 5):
#
# Figure 2's caption describes a three-arm drift statistic,
#   Delta = Delta_hat - max(Delta_hat_rand, Delta_hat_self),
# i.e. subtracting the larger of a random-frame-mask control and a no-mask
# (self) control from the raw destroy-vs-sham score. This is NOT what
# Definition 2/Eq. 5, the "Diagnostics" paragraph in Section 6, or this
# module implement. `self_flips` above is accepted and stored for exactly
# the diagnostic purpose Section 6 states ("these diagnostics characterize
# the pipeline but are not subtracted separately from Delta_hat, because
# the intact-frame sham already includes caption regeneration") - it is
# never subtracted here, and there is no random-mask condition in this
# codebase at all. Figure 2 needs a matching text/diagram fix before
# submission; `DeltaResult.delta` is the number that is actually reported
# in every table in the paper.
# ---------------------------------------------------------------------------


def accept(delta_value, theta=0.0, comparator='le'):
    """Section 5's decision rule: accept iff Delta_hat >= theta (comparator
    is which side of theta counts as "re-route" - the shipped controller
    (carve/controller.py, ported from carve_defer_fix.py) uses `theta=0.0,
    comparator='le'`, i.e. re-route on Delta <= 0 - matching Table 9's
    "Inclusive trigger theta>=0" row (48.06%/59.2% reroute), NOT the
    "Strict trigger theta<0" row (45.53%/13.2%) that a stricter '<'
    comparator would produce. Both comparators are implemented here so
    E3's threshold ablation can reproduce both rows from the same function."""
    if delta_value is None:
        return None  # can't compute Delta -> not eligible for the trigger set at all
    if comparator == 'le':
        return delta_value > theta       # True == accept, False == re-route
    if comparator == 'lt':
        return delta_value >= theta
    raise ValueError(f'unknown comparator {comparator!r}')
