"""Eq. (1) and Assumption 1: the matched sham/destroy/self replay engine.

Replays ONLY the final answering step from a frozen trajectory - the
planner, temporal grounder, retriever, and backbone agent loop are never
invoked during a probe. No new planning or retrieval calls are made
(Section 5, step 2).

Global flat work queue over (question_id, condition, repetition), not
nine-at-a-time per question, so every caption-model replica stays
saturated under many parallel workers.

Ported from analysis/carve_scripts/phase_b_worker.py (LVBench) and
analysis/scripts/phase_b_worker_vm.py (VideoMME), which were byte-identical
except for which harness module supplied the masking/context helpers - see
`integrations/videoexplorer/vendor/` for the VideoExplorer-specific
implementations of the four functions this module expects to be handed via
`HarnessHooks`, rather than importing them itself. That indirection is the
actual boundary between "the CARVE method" and "one agent's integration
of it."
"""
import os
import re
import time
import json
import glob
import hashlib
from dataclasses import dataclass
from typing import Callable, Optional

CONDITIONS = [('SELF', 'self'), ('SHAM', 'sham'), ('DESTROY', 'phase')]
STALE_CLAIM_GRACE_SECONDS = 600  # far above observed max job time; see release_stale_claims


@dataclass
class HarnessHooks:
    """The four integration-specific operations `run_job` needs. Concrete
    implementations for VideoExplorer live in
    integrations/videoexplorer/vendor/ (ported from run_lvbench_full_k1.py /
    run_videomme_phaseA.py, which supplied these byte-identically across
    both benchmarks)."""
    make_destroy_clip: Callable    # (clip, cache_dir, tag, seed) -> masked_paths ; T_destroy(E)
    make_sham_clip: Callable       # (clip, cache_dir, tag)       -> masked_paths ; T_sham(E) = E
    build_clean_context: Callable  # (raw, messages, evidence_indices, text_by_idx) -> messages
    query_formatted_for: Callable  # (query) -> formatted evidence-rendering prompt
    content_text: Callable         # (message_content) -> str, for the prompt hash
    mask_frame_dir: str


def job_key(qid, cond, rep):
    return f'{qid}__{cond}__{rep}'.replace('/', '_')


def build_job_list(frozen_dir, k):
    qids = sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{frozen_dir}/*.json'))
    return [(qid, label, rep) for qid in qids for label, _ in CONDITIONS for rep in range(k)]


def claim(queue_dir, key):
    try:
        os.mkdir(os.path.join(queue_dir, 'claims', key))
        return True
    except FileExistsError:
        return False


def release_stale_claims(queue_dir, result_path_fn):
    """A claim held with no corresponding result means the owning worker
    died mid-job (crash/kill/OOM). Without release, that job blocks forever
    and the run can never reach 100%."""
    claims_dir = os.path.join(queue_dir, 'claims')
    now = time.time()
    released = 0
    for entry in os.listdir(claims_dir):
        try:
            qid, cond, rep = entry.rsplit('__', 2)
        except ValueError:
            continue
        _, rpath = result_path_fn(queue_dir, qid, cond, int(rep))
        if os.path.exists(rpath):
            continue
        cpath = os.path.join(claims_dir, entry)
        try:
            if now - os.path.getmtime(cpath) > STALE_CLAIM_GRACE_SECONDS:
                os.rmdir(cpath)
                released += 1
        except OSError:
            pass
    return released


def result_path(queue_dir, qid, cond, rep):
    d = os.path.join(queue_dir, 'results', qid.replace('/', '_'), cond)
    return d, os.path.join(d, f'{rep}.json')


def run_job(manager, frozen, label, mode, rep, replica_id, hooks: Optional[HarnessHooks] = None):
    """One probe draw. `mode` in {'self', 'sham', 'phase'}. `mode='self'`
    holds the textual evidence fixed and reruns only the final answerer
    (Section 6's "Answer-self" diagnostic); `mode in {'sham','phase'}`
    regenerates a fresh caption via the harness's own captioner before
    reanswering (Assumption 1: sham and destroy differ ONLY in whether the
    regenerated caption came from intact or phase-randomized frames).
    Returns the full record for atomic write.
    """
    t_job = time.time()
    qid = frozen['question_id']
    raw = frozen['raw_data']
    messages = frozen['messages']
    evidence_indices = frozen['evidence_indices']
    ref_answer = frozen['reference_answer']
    prepped = frozen['prepped']
    original_text_by_idx = {int(i): t for i, t in frozen['evidence_slots'].items()}

    valid_options = manager._get_valid_option_letters(raw)
    text_by_idx = dict(original_text_by_idx)
    captions, seeds = {}, {}
    mask_seconds = caption_seconds = 0.0
    n_caption_tasks = 0

    if mode != 'self':
        if hooks is None:
            raise ValueError("mode != 'self' requires HarnessHooks (masking is integration-specific)")
        t0 = time.time()
        tasks, metas = [], []
        for p in prepped:
            idx = p['idx']
            tag = f'{qid}_{idx}_{label}_{rep}'.replace('/', '_')
            if mode == 'phase':
                seed = f'{qid}_phase_{idx}_{rep}'
                masked_paths = hooks.make_destroy_clip(
                    p['clip'], os.path.join(hooks.mask_frame_dir, qid), tag, seed)
            else:
                seed = None
                masked_paths = hooks.make_sham_clip(
                    p['clip'], os.path.join(hooks.mask_frame_dir, qid), tag)
            seeds[str(idx)] = seed
            tasks.append((hooks.query_formatted_for(p['query']), masked_paths, p['timestamps']))
            metas.append(idx)
        mask_seconds = time.time() - t0
        n_caption_tasks = len(tasks)
        t1 = time.time()
        caps = manager.batch_video2text(tasks) if tasks else []
        caption_seconds = time.time() - t1
        for idx, cap in zip(metas, caps):
            wm = re.match(r'^(.*?is )', original_text_by_idx[idx], re.DOTALL)
            prefix = wm.group(1) if wm else ''
            text_by_idx[idx] = prefix + cap
            captions[str(idx)] = cap

    if hooks is not None:
        ctx = hooks.build_clean_context(raw, messages, evidence_indices, text_by_idx)
        ctx_text = '\n'.join(hooks.content_text(m.get('content')) for m in ctx)
    else:
        # mode == 'self': no context rebuild needed since text_by_idx is unchanged
        # from the frozen trajectory; a hooks-free caller may pass its own ctx builder.
        raise ValueError('HarnessHooks required to rebuild the answering context')
    prompt_sha256 = hashlib.sha256(ctx_text.encode('utf-8')).hexdigest()

    t2 = time.time()
    ans_draw = manager._answer_with_retry(ctx, valid_options)
    answer_seconds = time.time() - t2

    valid = bool(ans_draw.get('valid'))
    ans = ans_draw.get('answer') if valid else None
    flip = (1 if (ans is not None and ans != ref_answer) else (0 if ans is not None else None))
    st = getattr(manager, '_stats', {}) or {}

    return {
        'question_id': qid, 'condition': label, 'repetition': rep,
        'mode': mode, 'replica_id': replica_id,
        'reference_answer': ref_answer,
        'answer': ans, 'parsed_answer': ans_draw.get('answer'),
        'valid': valid, 'flip': flip,
        'retry_count': ans_draw.get('retry_count'),
        'status': ans_draw.get('status'),
        'finish_reason': st.get('last_call_finish_reason'),
        'input_tokens': st.get('last_call_input_tokens'),
        'output_tokens': st.get('last_call_output_tokens'),
        'prompt_sha256': prompt_sha256,
        'seeds': seeds, 'captions': captions,
        'n_evidence': frozen['n_evidence'], 'n_prepped_items': frozen['n_prepped_items'],
        'mask_seconds': round(mask_seconds, 3),
        'caption_seconds': round(caption_seconds, 3),
        'n_caption_tasks': n_caption_tasks,
        'answer_seconds': round(answer_seconds, 3),
        'job_seconds': round(time.time() - t_job, 3),
        'completed_at': time.time(),
    }
