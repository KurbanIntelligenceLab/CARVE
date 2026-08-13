"""
Faithful k'=2 recomputation of the 3 soft_sc signals (vote_frequency,
evidence_count, evidence_overlap), reusing the EXACT formulas from
run_e1_pilot.py's compute_candidate_signal_values + soft_vote_with_signal,
restricted to each of the 3 possible pairs of the saved 3 rollouts.
"""
import json
import glob
import os
import ast
import itertools

TOOL_TAGS = [
    'temporal_grounding_agent', 'video_reader_question',
    'video_segment_retriever_textual_query', 'video_segment_retriever_image_query',
    'subtitle_retriever', 'subtitle_extractor', 'video_browser',
]
SEGMENTS_MARKER = 'related segments in the video:'


def content_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join((p.get('text', '') if isinstance(p, dict) else str(p)) for p in c)
    return str(c or '')


def extract_tool_call_sequence(messages):
    seq = []
    for m in messages:
        if m.get('role') != 'assistant':
            continue
        txt = content_to_text(m.get('content'))
        for tag in TOOL_TAGS:
            if tag in txt:
                seq.append(tag)
    return seq


def _ast_pos_to_offset(text, lineno, col_offset):
    lines = text.split('\n')
    return sum(len(l) + 1 for l in lines[:lineno - 1]) + col_offset


def _parse_tuple_spans(list_text):
    try:
        tree = ast.parse(list_text, mode='eval')
        elts = tree.body.elts
    except (SyntaxError, ValueError, AttributeError):
        return []
    spans = []
    for elt in elts:
        start = _ast_pos_to_offset(list_text, elt.lineno, elt.col_offset)
        end = _ast_pos_to_offset(list_text, elt.end_lineno, elt.end_col_offset)
        try:
            value = ast.literal_eval(list_text[start:end])
            start_ts = value[0] if len(value) > 0 else None
            end_ts = value[1] if len(value) > 1 else None
        except Exception:
            start_ts = end_ts = None
        spans.append((start_ts, end_ts))
    return spans


def extract_segment_spans_from_text(text):
    spans = []
    idx = text.find(SEGMENTS_MARKER)
    while idx != -1:
        start_bracket = text.find('[', idx)
        if start_bracket == -1:
            break
        depth = 0
        end_bracket = None
        for i in range(start_bracket, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    end_bracket = i
                    break
        if end_bracket is None:
            break
        list_text = text[start_bracket:end_bracket + 1]
        for start_ts, end_ts in _parse_tuple_spans(list_text):
            if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float)) and end_ts > start_ts:
                spans.append((start_ts, end_ts))
        idx = text.find(SEGMENTS_MARKER, end_bracket)
    return spans


def interval_overlap_ratio(spans_a, spans_b):
    if not spans_a or not spans_b:
        return 0.0
    overlap = 0.0
    for (a0, a1) in spans_a:
        for (b0, b1) in spans_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                overlap += (hi - lo)
    len_a = sum(max(0.0, e - s) for s, e in spans_a) or 1.0
    len_b = sum(max(0.0, e - s) for s, e in spans_b) or 1.0
    denom = min(len_a, len_b)
    return min(1.0, overlap / denom) if denom > 0 else 0.0


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_candidate_signal_values(rollouts):
    """EXACT reimplementation of run_e1_pilot.py's function, operating on
    whatever subset of rollouts is passed in (here: pairs)."""
    max_evidence = max((len(r['evidence_indices']) for r in rollouts), default=0) or 1
    spans_by_idx = {r['rollout_index']: extract_segment_spans_from_text(r.get('evidence_text', '') or '')
                     for r in rollouts}
    any_spans = any(spans_by_idx.values())
    tool_seq_by_idx = {r['rollout_index']: set(r.get('tool_call_sequence') or extract_tool_call_sequence(r['messages']))
                        for r in rollouts}

    vote_counts = {}
    for r in rollouts:
        if r['valid_answer']:
            vote_counts[r['pred_answer']] = vote_counts.get(r['pred_answer'], 0) + 1
    n_valid = sum(vote_counts.values()) or 1

    values = {}
    for r in rollouts:
        idx = r['rollout_index']
        evidence_count = len(r['evidence_indices']) / max_evidence
        vote_frequency = (vote_counts.get(r['pred_answer'], 0) / n_valid) if r['valid_answer'] else 0.0

        peers = [o for o in rollouts
                 if o['rollout_index'] != idx and o['valid_answer'] and o['pred_answer'] == r['pred_answer']]
        if not peers:
            evidence_overlap = 0.0
        elif any_spans:
            evidence_overlap = sum(
                interval_overlap_ratio(spans_by_idx[idx], spans_by_idx[p['rollout_index']]) for p in peers
            ) / len(peers)
        else:
            evidence_overlap = sum(
                jaccard(tool_seq_by_idx[idx], tool_seq_by_idx[p['rollout_index']]) for p in peers
            ) / len(peers)

        values[idx] = {
            'vote_frequency': vote_frequency,
            'evidence_count': evidence_count,
            'evidence_overlap': evidence_overlap,
        }
    return values


def soft_vote_with_signal(rollouts, signal_values, signal_name):
    scores = {}
    for r in rollouts:
        if r['valid_answer']:
            scores[r['pred_answer']] = scores.get(r['pred_answer'], 0.0) + signal_values[r['rollout_index']][signal_name]
    winner = max(scores, key=scores.get) if scores else None
    return winner


# ---------------------------------------------------------------------
RESULT_FILES = [f for f in glob.glob('e1_full900/results/*.json')
                 if '.metadata' not in f and '.failures' not in f
                 and 'diversity_report' not in f and 'signal_evaluation_report' not in f]
e1_by_qid = {}
for f in sorted(RESULT_FILES, key=os.path.getmtime):
    d = json.load(open(f, encoding='utf-8'))
    for row in d:
        e1_by_qid[row['question_id']] = row

frozen = {}
for f in sorted(glob.glob('eval_result/e0_full900_validated.part*.json.part*')):
    d = json.load(open(f, encoding='utf-8'))
    for r in d['processed_results']:
        qid = r['raw_data']['id']
        if qid in frozen:
            continue
        frozen[qid] = r

common_ids = sorted(set(e1_by_qid.keys()) & set(frozen.keys()))
gold = {qid: frozen[qid]['vanilla']['raw_data']['answer'][0] for qid in common_ids}
N = len(common_ids)
print(f'N = {N}')

for signal in ['vote_frequency', 'evidence_count', 'evidence_overlap']:
    pair_accs = []
    for i, j in itertools.combinations(range(3), 2):
        correct = 0
        n_have = 0
        for qid in common_ids:
            rollouts_all = e1_by_qid[qid].get('rollouts', [])
            if i >= len(rollouts_all) or j >= len(rollouts_all):
                continue
            pair = [rollouts_all[i], rollouts_all[j]]
            n_have += 1
            sig_vals = compute_candidate_signal_values(pair)
            winner = soft_vote_with_signal(pair, sig_vals, signal)
            if winner == gold[qid]:
                correct += 1
        acc = correct / n_have if n_have else 0
        pair_accs.append(acc)
        print(f'  {signal}, pair ({i},{j}): n={n_have}, correct={correct}, accuracy={100*acc:.4f}%')
    avg = sum(pair_accs) / len(pair_accs)
    print(f'  {signal}, AVERAGE across 3 pairs: accuracy={100*avg:.4f}%')
    print()
