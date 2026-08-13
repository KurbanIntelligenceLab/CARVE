"""FFT phase-randomization for the DESTROY condition.

Standard phase-scrambling technique: replace an image's phase spectrum with
the phase spectrum of independent real-valued noise (magnitude spectrum is
untouched). Because the noise array is real, its FFT is automatically
Hermitian-symmetric, so the magnitude/replaced-phase recombination inverse-
transforms back to a real-valued image with no discarded imaginary residual.
The raw inverse-FFT result routinely swings far outside [0, 255] though, so
it's histogram-matched back to the original frame's pixel value distribution
(see _match_histogram) rather than clipped - clipping is nonlinear and was
measured, during this module's own tests, to reintroduce enough high-
frequency energy to distort the magnitude spectrum by 70-100%, which would
have defeated the entire point.

Caveat worth knowing before trusting DESTROY results on a specific dataset:
histogram matching preserves low-level stats well on natural/photographic
frames (measured ~20-30% magnitude-spectrum drift on a smooth test image),
but degrades sharply on frames with a near-binary/flat-color histogram
(measured ~250%+ drift on a synthetic two-tone test square) - rank-matching
a smooth scrambled signal back onto only 2-3 distinct pixel values is an
extreme quantization. Flat-color animated/cartoon content is the realistic
case this could bite - if this dataset includes that domain, spot-check a
few DESTROY frames from it before trusting Delta there.
"""
import hashlib
import os

import numpy as np
from PIL import Image


def seed_from_key(key):
    h = hashlib.sha256(key.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') % (2**32 - 1)


def _phase_randomize_channel(channel, rng):
    magnitude = np.abs(np.fft.fft2(channel))
    noise = rng.standard_normal(channel.shape)
    random_phase = np.angle(np.fft.fft2(noise))
    scrambled = magnitude * np.exp(1j * random_phase)
    return np.real(np.fft.ifft2(scrambled))


def _match_histogram(scrambled, reference):
    """Reorder scrambled's pixel values (rank-preserving) so its histogram
    exactly matches reference's.

    The raw phase-randomized array routinely swings far outside [0, 255]
    (a sharp-edged 64x64 test square overshoots to roughly -400..+150) -
    naively np.clip()-ing that range is a nonlinear operation that
    reintroduces high-frequency energy and was measured (see this module's
    tests) to distort the magnitude spectrum by 70-100%, defeating the
    entire point of phase-only randomization. Histogram matching is the
    standard fix from the phase-scrambling literature (e.g. the SHINE
    toolbox): it's still a nonlinear pointwise remap so it isn't a perfect
    100.00% spectral match either, but it guarantees the exact same pixel
    value distribution (same brightness/contrast/color balance - the
    low-level statistic that actually matters here) with no clipping.
    """
    shape = scrambled.shape
    reference_sorted = np.sort(reference.ravel())
    order = np.argsort(scrambled.ravel())
    matched = np.empty_like(order, dtype=reference_sorted.dtype)
    matched[order] = reference_sorted
    return matched.reshape(shape)


def destroy_frame(src_path, dst_path, seed_key):
    """Phase-randomize one frame image, deterministically seeded by seed_key."""
    rng = np.random.default_rng(seed_from_key(seed_key))
    with Image.open(src_path) as img:
        img = img.convert('RGB')
        arr = np.asarray(img, dtype=np.float64)

    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        scrambled = _phase_randomize_channel(arr[:, :, c], rng)
        out[:, :, c] = _match_histogram(scrambled, arr[:, :, c])

    out = np.clip(out, 0, 255).astype(np.uint8)  # histogram-matched to a uint8 reference, so this is a no-op safety net, not the real clamp
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    Image.fromarray(out, mode='RGB').save(dst_path)


def destroy_frames(frame_paths, dst_dir, seed_key_prefix):
    """Phase-randomize a list of frames (one video_reader/video_browser call's
    worth). Each frame gets its own draw off seed_key_prefix + its index, so
    the scrambled clip doesn't just repeat one static noise pattern.
    Returns the list of new frame paths, same order as frame_paths.
    """
    out_paths = []
    for i, src_path in enumerate(frame_paths):
        dst_path = os.path.join(dst_dir, f'{i:04d}_{os.path.basename(src_path)}')
        destroy_frame(src_path, dst_path, f'{seed_key_prefix}_frame_{i}')
        out_paths.append(dst_path)
    return out_paths
