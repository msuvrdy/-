"""Compute global Mel z-score and temperature min/max from train-segment batches."""
import argparse
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BATCH_DIR = os.path.join(SCRIPT_DIR, "D_processed_data", "D_experiment_mel_32", "batches")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "D_processed_data", "D_experiment_mel_32", "norm_stats.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        default=DEFAULT_BATCH_DIR,
        help=f"Directory with batch_XX.npz files (default: {DEFAULT_BATCH_DIR})",
    )
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output path for norm_stats.npz (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    out = args.out

    mel_chunks, temp_chunks = [], []
    for b in range(args.num_batches):
        path = os.path.join(args.batch_dir, f"batch_{b:02d}.npz")
        data = np.load(path)
        mel_chunks.append(data["tst"])
        temp_chunks.append(data["tvt"])

    mel = np.concatenate(mel_chunks, axis=0)  # (N, 4, F, T)
    temp = np.concatenate(temp_chunks, axis=0)  # (N,)

    stft_mean = mel.mean(axis=(0, 3), keepdims=True).astype(np.float32)  # (1, 4, F, 1)
    stft_std = mel.std(axis=(0, 3), keepdims=True).astype(np.float32) + 1e-8
    temp_min = np.float32(temp.min())
    temp_max = np.float32(temp.max())

    np.savez(
        out,
        stft_mean=stft_mean.squeeze(),
        stft_std=stft_std.squeeze(),
        temp_min=temp_min,
        temp_max=temp_max,
    )
    n_freq = stft_mean.shape[2]
    print(f"Saved {out}")
    print(f"  stft_mean/std shape: (4, {n_freq}) from {len(mel)} train windows")
    print(f"  temp_min={temp_min:.4f}, temp_max={temp_max:.4f}")


if __name__ == "__main__":
    main()
