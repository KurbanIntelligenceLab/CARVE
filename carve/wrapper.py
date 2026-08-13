"""The wrapping interface: what CARVE requires of a tool-using video agent
(Section 3's problem setup) to be usable at all.

CARVE treats the agent as a black box exposing (tau, E, a_hat) - the
trajectory, the retrieved evidence, and the final answer - plus two
callable seams: an evidence-rendering step it can re-run on masked frames
(h in Eq. 1), and a re-route affordance that continues the SAME trajectory
toward new evidence rather than restarting (Section 5, step 3's re-route
action, and the Look-more baseline's "continue the frozen Phase-A
trajectory... so the agent resumes the same conversation instead of
restarting").

This module defines that seam as a Protocol. It intentionally does NOT
reimplement `_run_reroute_with_evidence_gate` or the agent loop itself -
those live inside the ~4,000-line vendored harness
(integrations/videoexplorer/vendor/eval.py's `SingleSampleProcessor`)
because they are genuinely coupled to VideoExplorer's specific tool-call
tag format, prompt structure, and retry logic. Re-deriving an
agent-agnostic implementation of that loop was out of scope for this
reorganization; what this module guarantees instead is that
`carve/controller.py`, `carve/probe.py`, and `baselines/` only ever call
through this seam, so swapping in a different agent means writing a new
`AgentWrapper` adapter, not touching the method code.
"""
from typing import Any, Dict, List, Optional, Protocol, Tuple


class AgentWrapper(Protocol):
    """What CARVE needs from a wrapped tool-using video agent."""

    def run(self, question: str, video: str) -> Tuple[List[Dict], List[int], str]:
        """Returns (trajectory_messages, evidence_indices, answer) - a
        fresh rollout from scratch. Section 3's f(q, E)."""
        ...

    def rerender_evidence(self, evidence_item: Dict, transform) -> str:
        """h in Eq. (1): re-render one retrieved evidence item to text
        after applying `transform` (identity for sham, phase-randomization
        for destroy - see carve/mask.py). Returns the regenerated caption."""
        ...

    def reanswer(self, context: List[Dict], valid_options) -> Dict:
        """g in Eq. (1): the final answering step only, given a rebuilt
        context. Must NOT issue new tool calls - this is what makes a
        probe replay cheap (Section 5: "without issuing new planning or
        retrieval tool calls")."""
        ...

    def continue_with_evidence_gate(self, messages: List[Dict], question_dict: Dict,
                                    remaining_turns: int, evidence_indices: List[int],
                                    start_time: float, max_gate_retries: int = 1) -> Dict:
        """Section 5's re-route action: continue the SAME trajectory,
        supervised so the agent may not answer without at least one
        successfully executed tool call that is novel relative to every
        prior turn (Look-more and CARVE's re-route both go through this)."""
        ...


class VideoExplorerWrapper:
    """Concrete adapter over the vendored VideoExplorer/VideoDeepResearch
    harness (integrations/videoexplorer/vendor/eval.py's
    `SingleSampleProcessor`). Thin by design - every method below is a
    direct pass-through to the vendored manager object; this class exists
    so `carve/` and `baselines/` never import `eval_mod` directly, only
    this file does.
    """

    def __init__(self, manager):
        self._m = manager

    def run(self, question, video):
        raise NotImplementedError(
            "Fresh rollouts are produced by the vendored harness's own CLI drivers "
            "(experiments/e*/), not by this adapter - CARVE only ever probes or "
            "re-routes an ALREADY-COMPLETED trajectory (Section 5: 'CARVE never "
            "changes this original trajectory during probing').")

    def rerender_evidence(self, evidence_item, transform):
        return self._m.batch_video2text([evidence_item])[0]

    def reanswer(self, context, valid_options):
        return self._m._answer_with_retry(context, valid_options)

    def continue_with_evidence_gate(self, messages, question_dict, remaining_turns,
                                    evidence_indices, start_time, max_gate_retries=1):
        return self._m._run_reroute_with_evidence_gate(
            messages, question_dict, remaining_turns, evidence_indices, start_time,
            max_gate_retries=max_gate_retries)
