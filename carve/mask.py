"""Definition 1 (Evidence-destruction operator).

A visual transformation m(.) replaces retrieved frames E with a
format-preserving counterpart m(E) that keeps the frame count, resolution,
ordering, and temporal positions while attempting to destroy semantic
content. Two candidate transformations:

  - `phase_randomize` (primary, frozen operator): Fourier phase-scrambling.
    Because a visually altered frame is not automatically uninformative,
    this is what gets validated against the intact-frame sham in E0 -
    see `sham()` below and `experiments/e0_intervention_validation/`.
  - `mean_frame` (weak-mask ablation, E3 only): each frame replaced by its
    own flat per-channel mean color. Kept for the E3 mean-vs-phase ablation,
    not used in any primary result.

The actual Fourier phase-randomization implementation (`_fft_destroy.py`,
histogram-matched to avoid the clipping artifact described in its own
docstring) is unchanged from the original `carve_delta/fft_destroy.py`.
"""
import os

import numpy as np
from PIL import Image

from ._fft_destroy import (  # noqa: F401  (re-exported for callers that want the internals)
    seed_from_key, destroy_frame, destroy_frames,
)

__all__ = ['phase_randomize_frame', 'phase_randomize_frames', 'mean_frame_destroy',
           'mean_frame_destroy_frames', 'sham']


def phase_randomize_frame(src_path, dst_path, seed_key):
    """The primary, frozen destroy operator (T_destroy in Eq. 1)."""
    destroy_frame(src_path, dst_path, seed_key)


def phase_randomize_frames(frame_paths, dst_dir, seed_key_prefix):
    return destroy_frames(frame_paths, dst_dir, seed_key_prefix)


def mean_frame_destroy(src_path, dst_path):
    """Weak-mask ablation (E3 only, never a primary result): replace the
    frame with a flat image at its own per-channel mean color. Format-
    preserving (same resolution/mode), semantically emptier than
    phase-randomization since no spatial structure survives at all -
    that's exactly why it's the *weak* mask, per Definition 1."""
    with Image.open(src_path) as img:
        img = img.convert('RGB')
        arr = np.asarray(img, dtype=np.float64)
    mean_color = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    out = np.broadcast_to(mean_color, arr.shape).astype(np.uint8)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    Image.fromarray(out, mode='RGB').save(dst_path)


def mean_frame_destroy_frames(frame_paths, dst_dir):
    out_paths = []
    for i, src_path in enumerate(frame_paths):
        dst_path = os.path.join(dst_dir, f'{i:04d}_{os.path.basename(src_path)}')
        mean_frame_destroy(src_path, dst_path)
        out_paths.append(dst_path)
    return out_paths


def sham(frame_paths):
    """T_sham(E) = E (Eq. 1's sham condition is the identity transform on
    frame content - the same source frames are re-decoded and re-captioned
    through the identical downstream pipeline as the destroy condition, so
    the ONLY thing this function does is document that the sham path never
    touches pixel content). Returns frame_paths unchanged."""
    return list(frame_paths)
