#!/usr/bin/env python3
# Transparent batch master for AI mixes (memory-safe)
# - Loudness band:  -13.5 .. -10.0 LUFS (prefers top if safe)
# - True-peak:      -0.6 dBTP (SoundOn)
# - Oversampling:   8x (memory-friendly)
# - EQ focus: wide dips in mid/presence + ultra-high control (12–18 kHz)
# - Parallel comp:  <= 3% (default 2%)
# - No saturation; limiter mainly catches kicks, not melodies

import os, glob, math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

# ---------------------------- Targets ----------------------------
LUFS_MIN            = -13.5
LUFS_MAX            = -10.0
TP_CEILING_DBTP     = -0.6

GR_MAX_DB           = 2.0        # max instantaneous limiter GR we allow
GAIN_TOL_LU         = 0.10       # loudness trim tolerance
POST_LIMIT_PAD_DB   = 0.10

# Oversampling for TP measure/limit (memory-conscious)
OS_FACTOR           = 8
FIR_TAPS            = 127
KAISER_BETA         = 9.5

# ---------------------------- Filters / EQ ----------------------------
HPF_HZ              = 20.0

# Mid dip (broad, subtractive only)
MID1_CENTER_HZ      = 1800.0
MID1_Q              = 0.8
MID1_DB_MAX_CUT     = 2.0

# Presence dip (broad)
PRES_CENTER_HZ      = 3500.0
PRES_Q              = 1.0
PRES_DB_MAX_CUT     = 1.6

# High shelf cut (tames 9 kHz+ if harsh)
HIGH_SHELF_HZ       = 9000.0
HIGH_SHELF_MAX_CUT  = 1.0

# Ultra-high shelf cut (controls 15–16 kHz area)
UHIGH_SHELF_HZ      = 16000.0
UHIGH_SHELF_MAX_CUT = 1.2

# Dynamic de-ess (sibilance)
DEESS_CENTER_HZ     = 7000.0
DEESS_Q             = 2.0
DEESS_MAX_DB        = 3.0
DEESS_ATTACK_MS     = 3.0
DEESS_RELEASE_MS    = 60.0

# Dynamic air clamp (12–18 kHz, wide)
AIR_TAME_CENTER_HZ  = 13500.0
AIR_TAME_Q          = 0.9
AIR_TAME_MAX_DB     = 2.5
AIR_ATTACK_MS       = 2.0
AIR_RELEASE_MS      = 80.0

# Micro parallel comp (kept tiny to preserve dynamics)
COMP_RATIO          = 1.35
COMP_ATTACK_MS      = 30.0
COMP_RELEASE_MS     = 220.0
COMP_KNEE_DB        = 6.0
COMP_THRESH_DB      = -18.0
COMP_WET_DEFAULT    = 0.02    # 2%
COMP_WET_MAX        = 0.03    # hard cap

# CPU friendliness
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

# ---------------------------- Utils ----------------------------
def db_to_lin(db): return 10.0 ** (db / 20.0)
def lin_to_db(x):  return 20.0 * math.log10(max(float(x), 1e-12))

def unique_wavs():
    seen = set(); out = []
    for pat in ("*.wav", "*.WAV"):
        for p in glob.glob(pat):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key); out.append(p)
    return sorted(out)

# ---------------------------- Spectrum helpers ----------------------------
def band_energy_db(mono, sr, flo, fhi):
    n = 1
    while (1 << n) < min(len(mono), 65536): n += 1
    n_fft = 1 << n
    hop = n_fft // 2
    win = np.hanning(n_fft)
    acc = 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    band_ix = np.logical_and(freqs >= flo, freqs < fhi)

    if len(mono) < n_fft:
        pad = np.zeros(n_fft); pad[:len(mono)] = mono
        spec = np.fft.rfft(pad * win)
        mag2 = (spec.real**2 + spec.imag**2)
        acc = float(np.sum(mag2[band_ix])) + 1e-18
        return 10.0 * np.log10(acc)

    for start in range(0, len(mono) - n_fft + 1, hop):
        frame = mono[start:start+n_fft] * win
        spec = np.fft.rfft(frame)
        mag2 = (spec.real**2 + spec.imag**2)
        acc += float(np.sum(mag2[band_ix]))
    acc += 1e-18
    return 10.0 * np.log10(acc)

def analyze_tone(mono, sr):
    # low (60–160), mid (1–3k), presence (3–5k), high (6–12k), air (12–18k)
    low = band_energy_db(mono, sr, 60.0, 160.0)
    mid = band_energy_db(mono, sr, 1000.0, 3000.0)
    prs = band_energy_db(mono, sr, 3000.0, 5000.0)
    hig = band_energy_db(mono, sr, 6000.0, 12000.0)
    air = band_energy_db(mono, sr, 12000.0, 18000.0)
    return low, mid, prs, hig, air

def needs_deess(mono, sr):
    sib  = band_energy_db(mono, sr, 5000.0, 9000.0)
    upm  = band_energy_db(mono, sr, 2500.0, 5000.0)
    return (sib - upm) > 2.0

def needs_air_clamp(mono, sr):
    air  = band_energy_db(mono, sr, 12000.0, 18000.0)
    prs  = band_energy_db(mono, sr, 3000.0, 5000.0)
    return (air - prs) > 1.0

# ---------------------------- Biquads ----------------------------
def normalize_biquad(coeffs):
    b0,b1,b2,a0,a1,a2 = coeffs
    b0/=a0; b1/=a0; b2/=a0; a1/=a0; a2/=a0
    return (b0,b1,b2,a1,a2)

def biquad_highpass(sr, f0, Q=0.707):
    w0 = 2*math.pi*f0/sr; c = math.cos(w0); s = math.sin(w0)
    alpha = s/(2*Q)
    b0 =  (1 + c)/2; b1 = -(1 + c); b2 =  (1 + c)/2
    a0 =   1 + alpha; a1 =  -2*c;    a2 =   1 - alpha
    return normalize_biquad((b0,b1,b2,a0,a1,a2))

def biquad_high_shelf(sr, f0, gain_db, S=1.0):
    A = 10**(gain_db/40.0)
    w0 = 2*math.pi*f0/sr; c = math.cos(w0); s = math.sin(w0)
    alpha = s/2 * math.sqrt((A + 1/A)*(1/S - 1) + 2)
    b0 =    A*((A+1) + (A-1)*c + 2*math.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*c)
    b2 =    A*((A+1) + (A-1)*c - 2*math.sqrt(A)*alpha)
    a0 =       (A+1) - (A-1)*c + 2*math.sqrt(A)*alpha
    a1 =   2*((A-1) - (A+1)*c)
    a2 =       (A+1) - (A-1)*c - 2*math.sqrt(A)*alpha
    return normalize_biquad((b0,b1,b2,a0,a1,a2))

def biquad_peak(sr, f0, Q, gain_db):
    A = 10**(gain_db/40.0)
    w0 = 2*math.pi*f0/sr; c = math.cos(w0); s = math.sin(w0)
    alpha = s/(2*Q)
    b0 = 1 + alpha*A
    b1 = -2*c
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*c
    a2 = 1 - alpha/A
    return normalize_biquad((b0,b1,b2,a0,a1,a2))

def biquad_bandpass_peak(sr, f0, Q=1.0):
    w0 = 2*math.pi*f0/sr; c = math.cos(w0); s = math.sin(w0)
    alpha = s/(2*Q)
    b0 = alpha; b1 = 0.0; b2 = -alpha
    a0 = 1 + alpha; a1 = -2*c; a2 = 1 - alpha
    return normalize_biquad((b0,b1,b2,a0,a1,a2))

def biquad_process_stereo(x, coeffs):
    if x.ndim == 1: x = x[:,None]
    N,C = x.shape
    b0,b1,b2,a1,a2 = coeffs
    y = np.zeros_like(x, dtype=np.float64)
    for ch in range(C):
        x1=x2=0.0; y1=y2=0.0
        xc = x[:,ch]; yc = np.empty_like(xc)
        for n in range(N):
            xn = xc[n]
            yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
            yc[n] = yn
            x2,x1 = x1,xn; y2,y1 = y1,yn
        y[:,ch] = yc
    return y

# ---------------------------- Dynamics helpers ----------------------------
def envelope_abs(x, sr, attack_ms, release_ms):
    atk = math.exp(-1.0/(0.001*attack_ms*sr))
    rel = math.exp(-1.0/(0.001*release_ms*sr))
    x_m = np.mean(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    env = np.empty_like(x_m); e = 0.0
    for i,v in enumerate(x_m):
        e = atk*e + (1.0-atk)*v if v>e else rel*e + (1.0-rel)*v
        env[i] = e
    return env

def compressor_gain_curve(level_db, thr_db, ratio, knee_db):
    over = level_db - thr_db
    if knee_db <= 1e-9:
        return np.where(over>0.0, thr_db + over/ratio - (thr_db + over), 0.0)
    half = knee_db/2.0
    gain_red = np.zeros_like(level_db)
    below = over <= -half
    above = over >=  half
    mid   = (~below) & (~above)
    gain_red[below] = 0.0
    gain_red[above] = (thr_db + (over[above]) - (thr_db + over[above]/ratio)) - over[above]
    t = (over[mid] + half) / (2*half)
    comp_in = thr_db + (over[mid] - half*t*(1-t)*knee_db)
    comp_out = thr_db + (over[mid]/ratio)
    gain_red[mid] = comp_out - comp_in
    return gain_red

def compress_parallel(x, sr, thr_db=COMP_THRESH_DB, ratio=COMP_RATIO, knee_db=COMP_KNEE_DB,
                      attack_ms=COMP_ATTACK_MS, release_ms=COMP_RELEASE_MS, wet=COMP_WET_DEFAULT):
    wet = float(np.clip(wet, 0.0, COMP_WET_MAX))
    if wet <= 0.0:
        return x, 0.0, wet
    env = envelope_abs(x, sr, attack_ms, release_ms)
    level_db = 20.0*np.log10(np.maximum(env, 1e-12))
    gr_db = compressor_gain_curve(level_db, thr_db, ratio, knee_db)  # negative values
    gain = 10.0 ** (gr_db/20.0)
    gain = gain if x.ndim==1 else gain[:,None]
    y_comp = x * gain
    y = (1.0 - wet)*x + wet*y_comp
    return y, float(np.max(-gr_db)), wet

def deesser_dynamic(x, sr, enable=True):
    if not enable:
        return x, 0.0
    bp = biquad_bandpass_peak(sr, DEESS_CENTER_HZ, Q=DEESS_Q)
    band = biquad_process_stereo(x, bp)
    env = envelope_abs(band, sr, DEESS_ATTACK_MS, DEESS_RELEASE_MS)
    med = float(np.median(env) + 1e-12)
    thr = med * 2.0
    over = np.maximum(env - thr, 0.0) / (thr + 1e-12)
    drive = over / (1.0 + over)
    red_db = -DEESS_MAX_DB * drive
    gain = 10.0 ** (red_db/20.0)
    gain = gain if x.ndim==1 else gain[:,None]
    y = x + (gain - 1.0) * band
    return y, float(-np.min(red_db))

def air_tamer_dynamic(x, sr, enable=True):
    if not enable:
        return x, 0.0
    bp = biquad_bandpass_peak(sr, AIR_TAME_CENTER_HZ, Q=AIR_TAME_Q)
    band = biquad_process_stereo(x, bp)
    env = envelope_abs(band, sr, AIR_ATTACK_MS, AIR_RELEASE_MS)
    # Use a higher percentile so only bright peaks trigger
    p80 = float(np.percentile(env, 80)) + 1e-12
    thr = p80
    over = np.maximum(env - thr, 0.0) / (thr + 1e-12)
    drive = over / (1.0 + over)
    red_db = -AIR_TAME_MAX_DB * drive
    gain = 10.0 ** (red_db/20.0)
    gain = gain if x.ndim==1 else gain[:,None]
    y = x + (gain - 1.0) * band
    return y, float(-np.min(red_db))

# ---------------------------- Oversampled true-peak limiter ----------------------------
def design_kaiser_lowpass(fc, numtaps, beta, gain=1.0):
    n = np.arange(numtaps) - (numtaps - 1)//2
    h = 2*fc * np.sinc(2*fc*n)
    h *= np.kaiser(numtaps, beta)
    h /= np.sum(h)
    h *= gain
    return h

def conv_same_1d(x, h):
    y = np.convolve(x, h, mode='full')
    d = (len(h) - 1)//2
    return y[d:d+len(x)]

_OS_FILTERS = None
def get_os_filters(L=OS_FACTOR, taps=FIR_TAPS, beta=KAISER_BETA):
    global _OS_FILTERS
    if _OS_FILTERS is None:
        fc = 0.5/L * 0.95
        h_up = design_kaiser_lowpass(fc, taps, beta, gain=L)
        h_dn = design_kaiser_lowpass(fc, taps, beta, gain=1.0)
        _OS_FILTERS = (h_up, h_dn)
    return _OS_FILTERS

def upsample_kaiser(x, L, h_up):
    if x.ndim == 1: x = x[:,None]
    N,C = x.shape
    z = np.zeros((N*L, C), dtype=np.float64)
    z[::L,:] = x
    y = np.empty_like(z)
    for c in range(C): y[:,c] = conv_same_1d(z[:,c], h_up)
    return y

def downsample_kaiser(y, L, h_dn):
    if y.ndim == 1: y = y[:,None]
    N,C = y.shape
    yf = np.empty_like(y)
    for c in range(C): yf[:,c] = conv_same_1d(y[:,c], h_dn)
    return yf[::L,:]

def truepeak_lin(x, L=OS_FACTOR, taps=FIR_TAPS, beta=KAISER_BETA):
    if x.ndim == 1: x = x[:,None]
    h_up, _ = get_os_filters(L, taps, beta)
    y_os = upsample_kaiser(x, L, h_up)
    return float(np.max(np.abs(y_os)))

def limit_oversampled(x, ceiling_db=TP_CEILING_DBTP, L=OS_FACTOR, taps=FIR_TAPS, beta=KAISER_BETA):
    if x.ndim == 1: x = x[:,None]
    ceil = db_to_lin(ceiling_db)
    h_up, h_dn = get_os_filters(L, taps, beta)
    y_os = upsample_kaiser(x, L, h_up)
    absmax = np.max(np.abs(y_os), axis=1)
    gain = np.minimum(1.0, ceil/np.maximum(absmax, 1e-12))
    y_os *= gain[:,None]
    y = downsample_kaiser(y_os, L, h_dn)
    return y

def limit_truepeak_strict(x, tp_limit_db=TP_CEILING_DBTP, base_ceiling_db=TP_CEILING_DBTP, max_iter=8):
    target = base_ceiling_db
    y = x
    for _ in range(max_iter):
        y = limit_oversampled(y, target)
        tp_db = lin_to_db(truepeak_lin(y))
        sp_db = lin_to_db(float(np.max(np.abs(y))))
        if tp_db <= tp_limit_db + 1e-6 and sp_db <= tp_limit_db + 1e-6:
            return y
        overshoot = max(tp_db - tp_limit_db, sp_db - tp_limit_db, 0.0)
        target -= overshoot + POST_LIMIT_PAD_DB
    return y

# ---------------------------- Loudness ----------------------------
def measure_lufs(audio, sr):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mono))

# ---------------------------- Per-file processing ----------------------------
def process_file(path):
    try:
        info = sf.info(path)
        data, sr = sf.read(path, always_2d=True, dtype='float32')
        x = data.astype(np.float64, copy=False)

        mono = x.mean(axis=1)
        l_in  = measure_lufs(x, sr)
        tp_in = lin_to_db(truepeak_lin(x))
        low_db, mid_db, prs_db, hig_db, air_db = analyze_tone(mono, sr)
        deess_flag = needs_deess(mono, sr)
        air_flag   = needs_air_clamp(mono, sr)

        # --- EQ decisions (subtractive only) ---
        # Mid dip vs avg of low & presence
        mid_ref   = 0.5 * (low_db + prs_db)
        mid_excess = max(0.0, mid_db - mid_ref)
        mid_cut   = -float(np.clip(0.6 * mid_excess, 0.0, MID1_DB_MAX_CUT))

        # Presence dip vs mid
        prs_excess = max(0.0, prs_db - mid_db)
        prs_cut   = -float(np.clip(0.6 * prs_excess, 0.0, PRES_DB_MAX_CUT))

        # High shelf cut vs mid
        hig_excess = max(0.0, hig_db - mid_db)
        high_cut  = -float(np.clip(0.4 * hig_excess, 0.0, HIGH_SHELF_MAX_CUT))

        # Ultra-high shelf cut vs presence (targets 12–18k “air hash”)
        air_excess = max(0.0, air_db - prs_db)
        uhigh_cut = -float(np.clip(0.5 * air_excess, 0.0, UHIGH_SHELF_MAX_CUT))

        # --- EQ processing ---
        y = biquad_process_stereo(x, biquad_highpass(sr, HPF_HZ, Q=0.707))
        if mid_cut < -0.1:
            y = biquad_process_stereo(y, biquad_peak(sr, MID1_CENTER_HZ, Q=MID1_Q, gain_db=mid_cut))
        if prs_cut < -0.1:
            y = biquad_process_stereo(y, biquad_peak(sr, PRES_CENTER_HZ, Q=PRES_Q, gain_db=prs_cut))
        if high_cut < -0.1:
            y = biquad_process_stereo(y, biquad_high_shelf(sr, HIGH_SHELF_HZ, gain_db=high_cut))
        if uhigh_cut < -0.1:
            y = biquad_process_stereo(y, biquad_high_shelf(sr, UHIGH_SHELF_HZ, gain_db=uhigh_cut))

        # Dynamic guards
        y, deess_gr = deesser_dynamic(y, sr, enable=deess_flag)
        y, air_gr   = air_tamer_dynamic(y, sr, enable=air_flag)

        # Micro parallel comp (<=3%)
        y, comp_gr_max, comp_wet_used = compress_parallel(y, sr, wet=COMP_WET_DEFAULT)

        # --- Gain plan: aim near LUFS_MAX without exceeding TP or GR caps ---
        l_pre = measure_lufs(y, sr)
        tp_pre = lin_to_db(truepeak_lin(y))
        headroom = TP_CEILING_DBTP - tp_pre
        allowed_gain = headroom + GR_MAX_DB

        gain_to_top = LUFS_MAX - l_pre
        G = min(gain_to_top, allowed_gain)

        desired_L = l_pre + G
        if desired_L < LUFS_MIN:
            G = min(LUFS_MIN - l_pre, allowed_gain + 1.0)
            desired_L = l_pre + G

        # Pre-gain then strict TP limit
        y = y * db_to_lin(G)
        y = limit_truepeak_strict(y, tp_limit_db=TP_CEILING_DBTP, base_ceiling_db=TP_CEILING_DBTP)

        # Tiny trims to settle within band (re-limit after trim)
        attempts = 0
        while attempts < 2:
            out_lufs = measure_lufs(y, sr)
            target_delta = 0.0
            if out_lufs > LUFS_MAX + GAIN_TOL_LU:
                target_delta = LUFS_MAX - out_lufs
            elif out_lufs < LUFS_MIN - 0.2:
                target_delta = min(LUFS_MIN - out_lufs, 0.8)
            if abs(target_delta) <= GAIN_TOL_LU:
                break
            y = limit_truepeak_strict(y * db_to_lin(target_delta),
                                      tp_limit_db=TP_CEILING_DBTP,
                                      base_ceiling_db=TP_CEILING_DBTP)
            attempts += 1

        # Final safety
        y = np.clip(y, -1.0, 1.0)

        # Metrics
        out_sp_db = lin_to_db(np.max(np.abs(y)))
        out_tp_db = lin_to_db(truepeak_lin(y))
        out_lufs  = measure_lufs(y, sr)

        # Write
        out = path[:-4] + "_master.wav"
        subtype = info.subtype or 'PCM_24'
        sf.write(out, y.astype(np.float32), sr, format='WAV', subtype=subtype)

        return (
            f"[OK] {os.path.basename(path)} -> {os.path.basename(out)}\n"
            f"     SR {sr} Hz | Ch {x.shape[1]}\n"
            f"     EQ cuts: mid {mid_cut:+.2f} dB @ {MID1_CENTER_HZ:.0f} Hz | "
            f"presence {prs_cut:+.2f} dB @ {PRES_CENTER_HZ:.0f} Hz | "
            f"high {high_cut:+.2f} dB @ {HIGH_SHELF_HZ/1000:.1f} kHz | "
            f"ultra-high {uhigh_cut:+.2f} dB @ {UHIGH_SHELF_HZ/1000:.1f} kHz\n"
            f"     De-ess: {'ON' if deess_flag else 'OFF'} | Max red ≈ {deess_gr:.1f} dB\n"
            f"     Air clamp: {'ON' if air_flag else 'OFF'} | Max red ≈ {air_gr:.1f} dB\n"
            f"     Parallel comp: wet {comp_wet_used*100:.0f}% | Max GR ≈ {comp_gr_max:.1f} dB\n"
            f"     Loudness: in {l_in:+.2f} LUFS → out {out_lufs:+.2f} LUFS (band {LUFS_MIN:+.1f}…{LUFS_MAX:+.1f})\n"
            f"     Peaks: in TP {tp_in:6.2f} dBTP → out TP {out_tp_db:6.2f} dBTP (ceil {TP_CEILING_DBTP:.1f}, {OS_FACTOR}× OS) | SP {out_sp_db:6.2f} dBFS"
        )
    except Exception as e:
        return f"[ERROR] {os.path.basename(path)}: {e}"

# ---------------------------- Main ----------------------------
def main():
    files = unique_wavs()
    if not files:
        print("No .wav files found.")
        return
    workers = max(1, 4)
    print(f"Found {len(files)} WAV file(s). Using {workers} worker(s). "
          f"Target {LUFS_MIN:.1f}…{LUFS_MAX:.1f} LUFS, TP≤{TP_CEILING_DBTP:.1f} dBTP, {OS_FACTOR}× OS.\n")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_file, f) for f in files]
        for fut in as_completed(futures):
            print(fut.result())

if __name__ == "__main__":
    main()
