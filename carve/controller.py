"""Section 5: CARVE, the re-grounding controller.

This is the deployed method - ported from carve_defer_fix.py, the end of
the abandoned-iteration sequence documented in archive/README.md. Given a
question whose stored Delta triggers re-routing (`carve.delta.accept`
returns False), it defers to an INDEPENDENT second reasoner over the same
evidence the original agent inspected:

  - independent, not stronger: same 7B scale as the planner, already used
    as the captioner elsewhere in the pipeline. Nothing here is a
    capability upgrade; the only variable is which model looks at the
    evidence.
  - it does NOT receive the original agent's answer, reasoning, captions,
    or the gold label - the second answer is formed independently, not by
    agreeing or disagreeing with an anchor.
  - it sees the SAME evidence intervals the original agent already
    inspected (capped at MAX_CLIPS), not new footage - this isolates
    "does a second, independent reasoner over the same evidence disagree"
    from "does different/better retrieval change the answer" (conflating
    the two is what the abandoned V1 controller got wrong).
  - no judge: the deferral answer unconditionally replaces the original
    when the question was triggered. This measures only whether the
    independent candidate creates more repair opportunities than harm
    opportunities (Table 8).

Compute-bounded per Proposition 5: at most one extra rollout (this
deferral pass) plus the probe replays that produced the stored Delta.
"""
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

MAX_CLIPS = 3        # cap on clips shown to the deferral model, bounds cost/question
MIN_SPAN_S = 10.0    # minimum evidence span; matches the benchmark clip length

ASK = (
    "Answer the multiple-choice question using ONLY what you can see in this "
    "footage.\n\nQUESTION: {question}\n\nOPTIONS:\n{options}\n\n"
    "Reply with the single letter of the best option and nothing else. "
    "If the footage does not show enough, still give your best single letter.\n"
    "Answer:"
)


def parse_letter(text, valid):
    if not text:
        return None
    m = re.search(r'\b([A-H])\b', str(text))
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(r'([A-H])[.):]', str(text))
    return m.group(1).upper() if m and m.group(1).upper() in valid else None


def widen_point_interval(a, b, min_span=MIN_SPAN_S):
    """A frozen trajectory sometimes records a point timestamp (e.g.
    [1199.0, 1199.0]), which yields a zero-frame clip that the video
    processor rejects outright - the single largest source of deferral
    failures before this fix. Widen to one min_span window centred on the
    point instead of dropping the question."""
    if b - a >= min_span:
        return a, b, False
    mid = (a + b) / 2.0
    return max(0.0, mid - min_span / 2.0), mid + min_span / 2.0, True


@dataclass
class DeferralResult:
    question_id: str
    vanilla_answer: Optional[str]
    gold_answer: str
    deferral_answer: Optional[str]
    clips_shown: int
    intervals_shown: List[Tuple[float, float]]
    n_widened_intervals: int
    status: str

    @property
    def vanilla_correct(self):
        return self.vanilla_answer == self.gold_answer

    @property
    def deferral_correct(self):
        return (self.deferral_answer == self.gold_answer) if self.deferral_answer else None

    @property
    def repair_opportunity(self):
        return self.vanilla_correct is False and self.deferral_correct is True

    @property
    def harm_opportunity(self):
        return self.vanilla_correct is True and self.deferral_correct is False

    @property
    def final_answer(self):
        """No judge: unconditional replacement when this question was
        triggered and the deferral produced a valid, parseable answer."""
        return self.deferral_answer if self.deferral_answer else self.vanilla_answer


def run_deferral(manager, question_id, evidence_intervals, video_path, question, options,
                  gold_answer, vanilla_answer, valid_option_letters,
                  clip_resolver: Callable, batch_video2text: Callable):
    """`clip_resolver(video_path, a, b, fps) -> (clip_paths, used_interval)`
    and `batch_video2text([(prompt, clips, timestamps)]) -> [raw_text]` are
    integration-specific (they know how to turn a time interval into model
    input) - see integrations/videoexplorer/vendor/."""
    clips, used, widened = [], [], 0
    for iv in evidence_intervals[:MAX_CLIPS]:
        a, b = float(iv[0]), float(iv[1])
        a, b, was_widened = widen_point_interval(a, b)
        widened += int(was_widened)
        try:
            paths, ts = clip_resolver(video_path, a, b, fps=2.0)
        except Exception:
            continue
        if paths:
            clips.extend(paths if isinstance(paths, list) else [paths])
            used.append((a, b))

    if not clips:
        return DeferralResult(question_id, vanilla_answer, gold_answer, None, 0, [], widened,
                              status='no_clips')

    prompt = ASK.format(question=question, options='\n'.join(options) if isinstance(options, list) else str(options))
    try:
        out = batch_video2text([(prompt, clips, used[0] if used else None)])
        raw = out[0] if out else None
    except Exception:
        return DeferralResult(question_id, vanilla_answer, gold_answer, None, len(clips), used,
                              widened, status='deferral_exception')

    ans = parse_letter(raw, valid_option_letters)
    return DeferralResult(question_id, vanilla_answer, gold_answer, ans, len(clips), used, widened,
                          status='ok')
