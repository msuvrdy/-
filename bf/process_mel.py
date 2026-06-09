"""Preprocess UOEMD data with log-Mel spectrograms (speed-robust features)."""
import numpy as np, csv, os, sys

FS = 42000; WINDOW = 4096; STEP = 2048
N_FFT = 256; N_MELS = 64; NPERSEG = 256; NOVERLAP = 128
TRAIN_END = int(6.5 * FS); VAL_END = int(8.0 * FS)
BASE = sys.argv[1]; OUT = sys.argv[2]; BATCH = int(sys.argv[3])
FAULT2ID = {'H_H':0,'R_U':1,'R_M':2,'S_W':3,'V_U':4,'B_R':5,'K_A':6,'F_B':7}
FILES_PER_BATCH = 8

# ------------------------------------------------------------
# Mel filterbank
# ------------------------------------------------------------
def make_mel_filterbank(n_fft, n_mels, fs, f_min=0, f_max=None):
    if f_max is None: f_max = fs / 2
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, ctr): fbank[m-1, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi): fbank[m-1, k] = (hi - k) / max(hi - ctr, 1)
    return fbank

FBM = make_mel_filterbank(N_FFT, N_MELS, FS)

# ------------------------------------------------------------
# Signal processing
# ------------------------------------------------------------
def load_csv(fp):
    with open(fp,'r',encoding='utf-8-sig') as f:
        rows = [r for r in csv.reader(f) if len(r)>=5]
    return np.array([[float(x) for x in r[:5]] for r in rows[1:]], dtype=np.float32)

def detrend(sig):
    sig = sig - np.mean(sig)
    n = len(sig); x = np.arange(n, dtype=np.float64)
    A = np.vstack([x, np.ones(n)]).T
    slope, intc = np.linalg.lstsq(A, sig, rcond=None)[0]
    return (sig - (slope*x + intc)).astype(np.float32)

def windows(data_5ch, start, end):
    seg = data_5ch[start:end]
    nw = max(0, (len(seg)-WINDOW)//STEP+1)
    if nw==0: return np.empty((0,5,WINDOW), dtype=np.float32)
    w = np.zeros((nw,5,WINDOW), dtype=np.float32)
    for i in range(nw): w[i] = seg[i*STEP:i*STEP+WINDOW].T
    return w

def compute_mel(windows_4ch):
    N = len(windows_4ch)
    hop = NPERSEG - NOVERLAP
    nfr = (WINDOW - NPERSEG) // hop + 1
    if N == 0: return np.empty((0, 4, N_MELS, nfr), dtype=np.float32)
    result = np.zeros((N, 4, N_MELS, nfr), dtype=np.float32)
    wfn = np.hanning(NPERSEG).astype(np.float32)
    for i in range(N):
        for c in range(4):
            sig = windows_4ch[i, c]
            for frm in range(nfr):
                s = frm * hop
                frame = sig[s:s+NPERSEG] * wfn
                mag = np.abs(np.fft.rfft(frame))
                mel = np.dot(FBM, mag)
                db = 20 * np.log10(np.maximum(mel, 1e-10))
                result[i, c, :, frm] = np.clip(db, -80, None)
    return result

# ------------------------------------------------------------
# Collect files
# ------------------------------------------------------------
all_files = []
for cond_dir, ld in [('1_Unloaded_Condition','0'),('2_Loaded_Condition','1')]:
    folder = os.path.join(BASE, cond_dir)
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.csv'): continue
        parts = fn.replace('.csv','').split('_')
        sp = int(parts[2])
        if sp > 4: continue  # constant speed only
        all_files.append({
            'path': os.path.join(folder, fn),
            'fault_id': FAULT2ID[f'{parts[0]}_{parts[1]}'],
            'fname': fn, 'speed': sp, 'load': ld
        })

total = len(all_files)
start = BATCH * FILES_PER_BATCH; end = min(start + FILES_PER_BATCH, total)
batch = all_files[start:end]
print(f"Batch {BATCH}: files {start+1}-{end} of {total} (Mel spectrogram, constant speed)")

tst, tvt, tlt = [], [], []
vst, vvt, vlt = [], [], []
est, evt, elt = [], [], []

for fi in batch:
    raw = load_csv(fi['path'])
    for ch in range(4): raw[:, ch] = detrend(raw[:, ch])
    wt = windows(raw, 0, TRAIN_END)
    wv = windows(raw, TRAIN_END, VAL_END)
    wte = windows(raw, VAL_END, len(raw))
    st = compute_mel(wt[:,:4,:]); sv = compute_mel(wv[:,:4,:]); ste = compute_mel(wte[:,:4,:])
    tt = np.mean(wt[:,4,:], axis=1)
    tv = np.mean(wv[:,4,:], axis=1)
    tte = np.mean(wte[:,4,:], axis=1)
    l = fi['fault_id']
    tst.append(st); tvt.append(tt); tlt.extend([l]*len(st))
    vst.append(sv); vvt.append(tv); vlt.extend([l]*len(sv))
    est.append(ste); evt.append(tte); elt.extend([l]*len(ste))
    print(f"  {fi['fname']}: t{len(st)} v{len(sv)} e{len(ste)}")

os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/batch_{BATCH:02d}.npz",
    tst=np.concatenate(tst), tvt=np.concatenate(tvt), tlt=np.array(tlt, dtype=np.int32),
    vst=np.concatenate(vst), vvt=np.concatenate(vvt), vlt=np.array(vlt, dtype=np.int32),
    est=np.concatenate(est), evt=np.concatenate(evt), elt=np.array(elt, dtype=np.int32))
print(f"Saved: T={len(tlt)} V={len(vlt)} E={len(elt)}")
