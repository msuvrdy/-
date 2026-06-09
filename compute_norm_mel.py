"""Compute global Mel z-score stats from experiment batch npz files (train segment only)."""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_dir', required=True, help='e.g. processed_data/experiment_mel32/batches')
    p.add_argument('--num_batches', type=int, default=8)
    p.add_argument('--out', default=None, help='norm_stats.npz path (default: batch_dir/../norm_stats.npz)')
    args = p.parse_args()
    out = args.out or f"{args.batch_dir.rstrip('/')}/../norm_stats.npz"

    chunks = []
    for b in range(args.num_batches):
        d = np.load(f"{args.batch_dir}/batch_{b:02d}.npz")
        chunks.append(d['tst'])
    data = np.concatenate(chunks, axis=0)  # (N, 4, F, T)
    mean_s = data.mean(axis=(0, 3), keepdims=True).astype(np.float32)  # (4, F, 1)
    std_s = data.std(axis=(0, 3), keepdims=True).astype(np.float32) + 1e-8
    np.savez(out, stft_mean=mean_s.squeeze(), stft_std=std_s.squeeze())
    print(f"Saved {out}: shape (4, {mean_s.shape[1]}, 1) from {len(data)} train windows")


if __name__ == '__main__':
    main()
