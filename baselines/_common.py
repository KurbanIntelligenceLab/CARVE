"""Shared machinery for the compute-matched baselines (Section 6): Look-more
and Judge-grounding selection both continue the FROZEN Phase-A trajectory via
the wrapped agent's own loop, so the agent resumes the same conversation
instead of restarting from scratch. Neither ever reads Phase-B, Delta, or a
completed CARVE output for routing, and neither writes to them.

Ported verbatim from analysis/carve_scripts/baselines_cm.py (LVBench) and
analysis/scripts/baselines_cm_vmme.py (VideoMME); the two differed only in
paths and a qid-parsing fix (VideoMME qids are "601-2", not LVBench's
"<video>_<n>", so `qid.rsplit('_',1)[0]` silently mis-parses them). That
fix is folded in below via `video_id_of`.
"""
import re
import time
import copy
import random

# Terms that must never reach the model in a baseline continuation - a
# baseline is only a valid comparison point if it carries none of CARVE's
# own framing.
BANNED = ('delta', 'evidence dependence', 'evidence-dependence', 'drift',
          'failed grounding', 'semantic destruction', 'carve', 'destroy',
          'sham', 'did not depend')

LOOKMORE_INSTRUCTION = (
    "Gather additional video evidence before reconsidering the answer."
)

TOOL_TAGS = ('temporal_grounding_agent', 'video_reader', 'video_browser')
THINK_RE = re.compile(r'<think>.*?</think>', re.S)
VR_PAIR_RE = re.compile(
    r'<video_reader>.*?</video_reader>\s*<video_reader_question>.*?</video_reader_question>', re.S)
TG_RE = re.compile(r'<temporal_grounding_agent>.*?</temporal_grounding_agent>', re.S)
VB_RE = re.compile(r'<video_browser>.*?</video_browser>', re.S)


def assert_neutral(text):
    low = str(text).lower()
    hits = [b for b in BANNED if b in low]
    if hits:
        raise AssertionError(f'Delta/CARVE leakage in continuation prompt: {hits}')


def video_id_of(qid, frozen=None):
    """VideoMME qids ("601-2") carry no video-id-derived split; LVBench qids
    ("<video>_<n>") do. Prefer the frozen record's own video_id field
    (always correct); fall back to the LVBench convention only when it's
    absent, never silently on VideoMME data."""
    if frozen and frozen.get('video_id'):
        return frozen['video_id']
    return qid.rsplit('_', 1)[0]


def count_tools(msgs):
    """Explicit tool invocations, same strict definition used for the
    Table 7 tool-call recount."""
    per = {k: 0 for k in TOOL_TAGS}
    turns = 0
    for m in msgs:
        if m.get('role') != 'assistant':
            continue
        turns += 1
        c = THINK_RE.sub(' ', str(m.get('content')))
        per['video_reader'] += len(VR_PAIR_RE.findall(c))
        per['temporal_grounding_agent'] += len(TG_RE.findall(c))
        per['video_browser'] += len(VB_RE.findall(c))
    per['_total'] = sum(per[k] for k in TOOL_TAGS)
    per['_planner_turns'] = turns
    return per


def select_subset(m_target, seed, eligible, vidmap=None):
    """Video-stratified round-robin over a seeded shuffle. Uses only qid +
    video_id - never Delta, correctness, gold evidence, or Phase-B outcomes.
    `vidmap` (qid -> video_id) is required for VideoMME; LVBench falls back
    to the rsplit convention when it's omitted."""
    by_v = {}
    for q in sorted(eligible):
        vid = vidmap[q] if vidmap else q.rsplit('_', 1)[0]
        by_v.setdefault(vid, []).append(q)
    rnd = random.Random(seed)
    vids = sorted(by_v)
    rnd.shuffle(vids)
    for v in vids:
        rnd.shuffle(by_v[v])
    out, i = [], 0
    while len(out) < m_target:
        added = False
        for v in vids:
            if i < len(by_v[v]):
                out.append(by_v[v][i])
                added = True
                if len(out) >= m_target:
                    break
        if not added:
            break
        i += 1
    return out


def gen_candidate(manager, frozen, dic, budget, max_ds_round, row, gate=False,
                   max_gate_retries=1):
    """Neutral continuation of the frozen Phase-A trajectory.

    With gate=True the wrapped agent's evidence gate supervises the
    continuation: if it answers without a successfully executed tool call
    that is novel relative to every prior assistant turn (including the
    whole frozen trajectory), it is asked once to use a tool.
    """
    assert_neutral(LOOKMORE_INSTRUCTION)
    messages = copy.deepcopy(frozen['messages'])
    evidence_indices = list(frozen['evidence_indices'])
    base_len = len(messages)
    messages.append({'role': 'user', 'content': LOOKMORE_INSTRUCTION})
    row['continuation_instruction'] = LOOKMORE_INSTRUCTION
    row['budget_turns'] = budget
    row['evidence_gate'] = bool(gate)

    manager._reset_stats()
    t0 = time.time()
    try:
        if gate:
            res = manager._run_reroute_with_evidence_gate(
                messages, dic, max_ds_round - budget, evidence_indices, t0,
                max_gate_retries=max_gate_retries)
            row['gate_fired'] = res.get('gate_fired')
            row['gate_retries_used'] = res.get('gate_retries_used')
            row['gate_final_status'] = res.get('gate_final_status')
        else:
            res = manager._run_agent_loop(messages, dic, max_ds_round - budget,
                                          evidence_indices, t0)
    except Exception as e:
        row['status'] = 'continuation_exception'
        row['error'] = f'{type(e).__name__}: {e}'[:300]
        row['added_seconds'] = round(time.time() - t0, 2)
        return None, None
    row['added_seconds'] = round(time.time() - t0, 2)
    st = getattr(manager, '_stats', {}) or {}
    row['added_input_tokens'] = st.get('input_tokens')
    row['added_output_tokens'] = st.get('output_tokens')

    new_msgs = res['messages'][base_len:]
    row['continuation_trajectory'] = [
        {'role': m.get('role'), 'content': str(m.get('content'))[:4000]} for m in new_msgs]
    tc = count_tools(new_msgs)
    row['added_tool_calls'] = tc['_total']
    row['added_tool_calls_by_type'] = {k: tc[k] for k in TOOL_TAGS}
    row['added_planner_turns'] = tc['_planner_turns']
    row['hit_max_turns'] = not isinstance(res.get('score'), bool)

    ans = manager.extract_final_answer(str(new_msgs[-1].get('content')) if new_msgs else '')
    valid = manager._get_valid_option_letters(dic)
    if ans and manager._is_valid_answer_for_question(ans, valid):
        row['candidate_answer'] = ans
    else:
        row['candidate_answer'] = None
        row['candidate_invalid_raw'] = str(new_msgs[-1].get('content'))[:400] if new_msgs else ''

    new_ev = [str(res['messages'][i].get('content'))[:1200]
              for i in res['evidence_indices'] if i >= base_len and i < len(res['messages'])]
    row['n_new_evidence_turns'] = len(new_ev)
    row['new_evidence_text'] = '\n'.join(new_ev)[:8000]
    return row['candidate_answer'], '\n'.join(new_ev)


def finish(row, t0, final, reason):
    row['selection_reason'] = reason
    row['final_answer'] = final
    row['final_correct'] = (final == row['gold_answer']) if final else None
    b, a = row.get('vanilla_correct'), row['final_correct']
    row['repair'] = bool(b is False and a is True)
    row['harm'] = bool(b is True and a is False)
    row['both_right'] = bool(b is True and a is True)
    row['both_wrong'] = bool(b is False and a is False)
    row['fell_back_to_vanilla'] = final == row['vanilla_answer']
    row['total_seconds'] = round(time.time() - t0, 2)
    return row
