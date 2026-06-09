"""Preprocess UOEMD-VAFCVS CSV files into log-Mel spectrogram batches (experiment_mel_32)."""
import csv
import os
import sys

import numpy as np

FS = 42000
WINDOW = 4096
STEP = 2048
N_FFT = 256
N_MELS = 32
NPERSEG = 256
NOVERLAP = 128
F_MIN = 0
F_MAX = 21000

TRAIN_END = int(6.5 * FS)
VAL_END = int(8.0 * FS)

FAULT2ID = {"H_H": 0, "R_U": 1, "R_M": 2, "S_W": 3, "V_U": 4, "B_R": 5, "K_A": 6, "F_B": 7}
FILES_PER_BATCH = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_BASE_DIR = os.path.join(PROJECT_ROOT, "2_CSV_Data_Files")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "D_processed_data", "D_experiment_mel_32", "batches")


def make_mel_filterbank(n_fft, n_mels, fs, f_min=0, f_max=None):
    """Build triangular Mel filterbank covering [f_min, f_max] Hz."""
    if f_max is None:
        f_max = fs / 2
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, ctr):
            fbank[m - 1, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            fbank[m - 1, k] = (hi - k) / max(hi - ctr, 1)
    return fbank


FBM = make_mel_filterbank(N_FFT, N_MELS, FS, f_min=F_MIN, f_max=F_MAX)


def load_csv(fp):
    with open(fp, "r", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 5]
    return np.array([[float(x) for x in r[:5]] for r in rows[1:]], dtype=np.float32)


def detrend(sig):
    """Remove DC offset and least-squares linear trend."""
    sig = sig - np.mean(sig)
    n = len(sig)
    x = np.arange(n, dtype=np.float64)
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, sig, rcond=None)[0]
    return (sig - (slope * x + intercept)).astype(np.float32)


def sliding_windows(data_5ch, start, end):
    seg = data_5ch[start:end]
    nw = max(0, (len(seg) - WINDOW) // STEP + 1)
    if nw == 0:
        return np.empty((0, 5, WINDOW), dtype=np.float32)
    w = np.zeros((nw, 5, WINDOW), dtype=np.float32)
    for i in range(nw):
        w[i] = seg[i * STEP : i * STEP + WINDOW].T
    return w


def compute_mel(windows_4ch):
    """STFT + Mel filterbank + dB conversion for channels 0-3."""
    hop = NPERSEG - NOVERLAP
    nfr = (WINDOW - NPERSEG) // hop + 1
    n = len(windows_4ch)
    if n == 0:
        return np.empty((0, 4, N_MELS, nfr), dtype=np.float32)

    result = np.zeros((n, 4, N_MELS, nfr), dtype=np.float32)
    wfn = np.hanning(NPERSEG).astype(np.float32)
    for i in range(n):
        for c in range(4):
            sig = windows_4ch[i, c]
            for frm in range(nfr):
                s = frm * hop
                frame = sig[s : s + NPERSEG] * wfn
                mag = np.abs(np.fft.rfft(frame, n=N_FFT))
                mel = np.dot(FBM, mag)
                db = 20.0 * np.log10(np.maximum(mel, 1e-10))
                result[i, c, :, frm] = np.clip(db, -80, None)
    return result


def collect_files(base_dir):
    """Sorted file list: unloaded (load=0) then loaded (load=1), constant speed sp<=4 only."""
    all_files = []
    for cond_dir, load_id in [("1_Unloaded_Condition", 0), ("2_Loaded_Condition", 1)]:
        folder = os.path.join(base_dir, cond_dir)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            parts = fn.replace(".csv", "").split("_")
            sp = int(parts[2])
            if sp > 4:
                continue
            fault_key = f"{parts[0]}_{parts[1]}"
            all_files.append(
                {
                    "path": os.path.join(folder, fn),
                    "fault_id": FAULT2ID[fault_key],
                    "fname": fn,
                    "speed": sp,
                    "load": load_id,
                }
            )
    return all_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python D_process_mel.py <BATCH_IDX> [BASE_DIR] [OUT_DIR]")
        print(f"  Default BASE_DIR: {DEFAULT_BASE_DIR}")
        print(f"  Default OUT_DIR:  {DEFAULT_OUT_DIR}")
        sys.exit(1)

    if len(sys.argv) == 2:
        base_dir = DEFAULT_BASE_DIR
        out_dir = DEFAULT_OUT_DIR
        batch_idx = int(sys.argv[1])
    else:
        base_dir = sys.argv[1]
        out_dir = sys.argv[2]
        batch_idx = int(sys.argv[3])

    all_files = collect_files(base_dir)
    total = len(all_files)
    start = batch_idx * FILES_PER_BATCH
    end = min(start + FILES_PER_BATCH, total)
    batch = all_files[start:end]

    print(f"Batch {batch_idx}: files {start + 1}-{end} of {total} (N_MELS={N_MELS}, sp<=4)")

    tst, tvt, tlt = [], [], []
    vst, vvt, vlt = [], [], []
    est, evt, elt = [], [], []

    for fi in batch:
        raw = load_csv(fi["path"])
        for ch in range(4):
            raw[:, ch] = detrend(raw[:, ch])

        wt = sliding_windows(raw, 0, TRAIN_END)
        wv = sliding_windows(raw, TRAIN_END, VAL_END)
        wte = sliding_windows(raw, VAL_END, len(raw))

        st = compute_mel(wt[:, :4, :])
        sv = compute_mel(wv[:, :4, :])
        ste = compute_mel(wte[:, :4, :])

        tt = np.mean(wt[:, 4, :], axis=1)
        tv = np.mean(wv[:, 4, :], axis=1)
        tte = np.mean(wte[:, 4, :], axis=1)

        label = fi["fault_id"]
        tst.append(st)
        tvt.append(tt)
        tlt.extend([label] * len(st))
        vst.append(sv)
        vvt.append(tv)
        vlt.extend([label] * len(sv))
        est.append(ste)
        evt.append(tte)
        elt.extend([label] * len(ste))
        print(f"  {fi['fname']}: train={len(st)} val={len(sv)} test={len(ste)}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"batch_{batch_idx:02d}.npz")
    np.savez(
        out_path,
        tst=np.concatenate(tst),
        tvt=np.concatenate(tvt),
        tlt=np.array(tlt, dtype=np.int32),
        vst=np.concatenate(vst),
        vvt=np.concatenate(vvt),
        vlt=np.array(vlt, dtype=np.int32),
        est=np.concatenate(est),
        evt=np.concatenate(evt),
        elt=np.array(elt, dtype=np.int32),
    )
    print(f"Saved {out_path}: T={len(tlt)} V={len(vlt)} E={len(elt)}")


if __name__ == "__main__":
    main()
