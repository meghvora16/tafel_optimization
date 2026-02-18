"""
Tafel Fitting Tool — v4: Comprehensive Global Electrochemical Model
====================================================================
Handles ALL electrochemically possible polarization curve types:
  • Active only (simple BV)
  • Active + diffusion-limited cathodic
  • Active + passive (with active dissolution peak / nose)
  • Active + passive + transpassive
  • Active + passive + transpassive + secondary passivity
  • Active + passive + pitting (sharp breakdown)
  • Any combination with cathodic diffusion limitation

Physics-based model with film-coverage approach for passivation.
Multi-stage optimization: DE → Basin-hopping → L-BFGS-B → Nelder-Mead.
Automatic data quality diagnostics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import (curve_fit, differential_evolution, minimize,
                             dual_annealing)
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from itertools import groupby
import warnings, io, re, time

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Tafel v4", page_icon="⚡", layout="wide")

st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{background:#0f0f1a;color:#cdd6f4}
[data-testid="stSidebar"]{background:#1a1a2e}
section[data-testid="stFileUploadDropzone"]{background:#1e1e2e!important;
  border:2px dashed #45475a!important;border-radius:12px!important}
.pcard{background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:14px 16px;margin:4px 0}
.plabel{color:#a6adc8;font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.pval{font-size:21px;font-weight:700;margin:1px 0}
.punit{color:#585b70;font-size:11px}
.sechead{color:#89b4fa;font-size:12px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;border-bottom:1px solid #313244;padding-bottom:5px;margin:14px 0 6px}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;margin:1px 2px}
.bg{background:#1c3a2f;color:#a6e3a1;border:1px solid #a6e3a1}
.bb{background:#1a2a3f;color:#89b4fa;border:1px solid #89b4fa}
.by{background:#3a3020;color:#f9e2af;border:1px solid #f9e2af}
.br{background:#3a1a20;color:#f38ba8;border:1px solid #f38ba8}
.bp{background:#2a1a3a;color:#cba6f7;border:1px solid #cba6f7}
.ok-box{background:#1e1e2e;border-left:4px solid #a6e3a1;border-radius:0 8px 8px 0;
  padding:8px 14px;margin:6px 0;font-size:12px;color:#cdd6f4}
.warn-box{background:#1e1e2e;border-left:4px solid #f9e2af;border-radius:0 8px 8px 0;
  padding:8px 14px;margin:6px 0;font-size:12px;color:#f9e2af}
.err-box{background:#1e1e2e;border-left:4px solid #f38ba8;border-radius:0 8px 8px 0;
  padding:8px 14px;margin:6px 0;font-size:12px;color:#f38ba8}
.type-box{background:linear-gradient(135deg,#1e1e2e,#232336);
  border:1px solid #45475a;border-radius:10px;padding:14px 18px;margin:8px 0}
.type-title{font-size:16px;font-weight:700;margin-bottom:4px}
.type-desc{font-size:12px;color:#a6adc8}
.diag-box{background:#1e1e2e;border:1px solid #313244;border-radius:10px;
  padding:14px 18px;margin:6px 0}
.diag-title{font-size:13px;font-weight:700;margin-bottom:6px}
.diag-item{font-size:12px;color:#a6adc8;padding:3px 0}
</style>""", unsafe_allow_html=True)

C = dict(
    data="#89b4fa", anodic="#f9e2af", cathodic="#cba6f7", fit="#a6e3a1",
    passive="rgba(166,227,161,0.10)", limiting="rgba(137,220,235,0.10)",
    transpassive="rgba(243,188,168,0.08)", sec_passive="rgba(203,166,247,0.08)",
    ecorr="#f38ba8", grid="#313244", bg="#1e1e2e", paper="#131320", text="#cdd6f4",
)

# ═══════════════════════════════════════════════════════════════
# FILE I/O  (same as v3, compact)
# ═══════════════════════════════════════════════════════════════
COL_SIG = [
    (r"we.*potential", r"we.*current", "A"), (r"ewe", r"i/ma", "mA"),
    (r"ewe", r"<i>/ma", "mA"), (r"^vf$", r"^im$", "A"),
    (r"potential/v", r"current/a", "A"), (r"e/v", r"i/a", "A"),
    (r"potential|volt|^e$|e \(v\)|e_v", r"current|amps|^i$|i \(a\)|i_a", "A"),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]
UHINT = {r"\(a\)|_a$|/a$":1.0, r"\(ma\)|_ma$|/ma$":1e-3,
         r"\(µa\)|_ua$|/ua$":1e-6, r"a/cm":1.0, r"ma/cm":1e-3}


def auto_detect_columns(df):
    cl = {c: c.lower().strip() for c in df.columns}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for Ep, Ip, u in COL_SIG:
        em = [c for c, v in cl.items() if re.search(Ep, v) and c in num]
        im = [c for c, v in cl.items() if re.search(Ip, v) and c in num and c not in em]
        if em and im:
            ec = sorted(em, key=lambda c: 0 if "we" in c.lower() else 1)[0]
            ic = im[0]
            f = 1e-3 if u == "mA" else 1.0
            for p, fv in UHINT.items():
                if re.search(p, cl[ic]): f = fv; break
            return ec, ic, f
    if len(num) >= 2: return num[0], num[1], 1.0
    raise ValueError("Could not detect columns.")


def load_any_file(f):
    nm = f.name.lower()
    ext = next((e for e in (".xlsx",".xls",".csv",".txt") if nm.endswith(e)), ".csv")
    raw = f.read(); f.seek(0)
    if ext in (".xlsx",".xls"): return pd.read_excel(io.BytesIO(raw))
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    skip = 0
    for idx, line in enumerate(lines):
        pts = re.split(r"[,;\t ]+", line.strip())
        if sum(1 for p in pts if re.match(r"^-?[\d.eE+]+$", p)) >= 2:
            skip = max(0, idx - 1) if idx > 0 and not any(
                re.match(r"^-?[\d.eE+]+$", p)
                for p in re.split(r"[,;\t ]+", lines[idx-1].strip())) else idx
            break
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5:
                return df.dropna(axis=1, how="all")
        except: pass
    raise ValueError(f"Cannot parse {f.name}")


# ═══════════════════════════════════════════════════════════════
# MATH HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_log(x): return np.log10(np.maximum(np.abs(x), 1e-20))

def smooth(y, w=11, p=3):
    n = len(y); w = min(w, n if n%2==1 else n-1)
    return savgol_filter(y, w, min(p, w-1)) if w >= 5 else y.copy()

def _r2(yt, yp):
    sr = np.sum((yt-yp)**2); st_ = np.sum((yt-yt.mean())**2)
    return float(max(0, 1-sr/st_)) if st_ > 0 else 0.0

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -50, 50)))


# ═══════════════════════════════════════════════════════════════
# GLOBAL ELECTROCHEMICAL MODEL  — v4
# ═══════════════════════════════════════════════════════════════
#
# Complete physics-based model using film-coverage approach:
#
#   i_net = i_anodic_total − i_cathodic
#
# ── CATHODIC ──────────────────────────────────────────────────
#   i_c_kinetic = icorr · exp(−2.303·η/bc)
#   i_cathodic  = i_c_kinetic / (1 + i_c_kinetic/iL)
#     → Tafel at low |η|, saturates at iL for mass-transport limit
#
# ── ANODIC  ───────────────────────────────────────────────────
#   Uses film-coverage model:
#
#   i_active = icorr · exp(2.303·η/ba)        ← activation kinetics
#
#   θ₁(E) = sigmoid(k₁·(E − Epp))           ← primary passive film coverage
#   i_after_pass1 = i_active·(1−θ₁) + ipass·θ₁
#     → Below Epp: pure Tafel (active dissolution)
#     → At Epp: peak (the "nose" — i_active rises but θ₁ turns on)
#     → Above Epp: drops to ipass (passive plateau)
#
#   i_transpassive = a_tp · exp(b_tp·(E−Eb)) · sigmoid(k_tp·(E−Eb))
#     → Exponential rise above breakdown Eb
#
#   θ₂(E) = sigmoid(k₂·(E − Esp))           ← secondary passive film
#   i_anodic_total = (i_after_pass1 + i_tp)·(1−θ₂) + ipass2·θ₂
#     → Above Esp: drops to ipass2 (secondary passivity)
#
# ── PARAMETER REDUCTION ──────────────────────────────────────
#   Active only:     k₁→0, a_tp→0, k₂→0     (12→4 effective params)
#   + diffusion:     add iL                    (12→5)
#   + passive:       add k₁, Epp, ipass        (12→7-8)
#   + transpassive:  add Eb, a_tp, b_tp        (12→10-11)
#   + secondary:     add Esp, ipass2, k₂        (full 12+)
# ═══════════════════════════════════════════════════════════════

# Fixed parameter order (14 params):
# [Ecorr, icorr, ba, bc, iL, Epp, k_pass, ipass, Eb, a_tp, b_tp, Esp, k_sp, ipass2]
PARAM_NAMES = ["Ecorr", "icorr", "ba", "bc", "iL",
               "Epp", "k_pass", "ipass",
               "Eb", "a_tp", "b_tp",
               "Esp", "k_sp", "ipass2"]
N_PARAMS = len(PARAM_NAMES)


def global_model(E, p):
    """
    Full global polarization curve model.
    p = [Ecorr, icorr, ba, bc, iL,
         Epp, k_pass, ipass,
         Eb, a_tp, b_tp,
         Esp, k_sp, ipass2]
    """
    Ecorr, icorr, ba, bc, iL = p[0], p[1], p[2], p[3], p[4]
    Epp, k_pass, ipass = p[5], p[6], p[7]
    Eb, a_tp, b_tp = p[8], p[9], p[10]
    Esp, k_sp, ipass2 = p[11], p[12], p[13]

    eta = E - Ecorr

    # ── Cathodic: Tafel → diffusion limited ──────────────
    i_c_kin = icorr * np.exp(-2.303 * eta / bc)
    i_cathodic = i_c_kin / (1.0 + i_c_kin / iL)

    # ── Anodic: active dissolution kinetics ──────────────
    i_active = icorr * np.exp(2.303 * eta / ba)

    # ── Primary passivation (film coverage θ₁) ──────────
    theta1 = sigmoid(E - Epp, k_pass)
    i_after_pass1 = i_active * (1.0 - theta1) + ipass * theta1

    # ── Transpassive dissolution ─────────────────────────
    i_tp = a_tp * np.exp(np.clip(b_tp * (E - Eb), -50, 50)) * sigmoid(E - Eb, 40.0)

    # ── Secondary passivation (film coverage θ₂) ────────
    theta2 = sigmoid(E - Esp, k_sp)

    # ── Combine anodic ──────────────────────────────────
    i_anodic_total = (i_after_pass1 + i_tp) * (1.0 - theta2) + ipass2 * theta2

    return i_anodic_total - i_cathodic


def global_model_components(E, p):
    """Return individual components for visualization."""
    Ecorr, icorr, ba, bc, iL = p[0:5]
    Epp, k_pass, ipass = p[5:8]
    Eb, a_tp, b_tp = p[8:11]
    Esp, k_sp, ipass2 = p[11:14]
    eta = E - Ecorr

    i_c_kin = icorr * np.exp(-2.303 * eta / bc)
    i_cathodic = i_c_kin / (1.0 + i_c_kin / iL)
    i_active = icorr * np.exp(2.303 * eta / ba)
    theta1 = sigmoid(E - Epp, k_pass)
    i_after_pass1 = i_active * (1.0 - theta1) + ipass * theta1
    i_tp = a_tp * np.exp(np.clip(b_tp * (E - Eb), -50, 50)) * sigmoid(E - Eb, 40.0)
    theta2 = sigmoid(E - Esp, k_sp)
    i_anodic_total = (i_after_pass1 + i_tp) * (1.0 - theta2) + ipass2 * theta2

    return dict(
        i_cathodic=i_cathodic, i_active=i_active,
        i_after_pass1=i_after_pass1, i_tp=i_tp,
        i_anodic_total=i_anodic_total,
        theta1=theta1, theta2=theta2,
        i_total=i_anodic_total - i_cathodic,
    )


# ═══════════════════════════════════════════════════════════════
# CURVE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

class CT:
    ACTIVE = "active"
    ACTIVE_DIFF = "active_diff"
    PASSIVE = "passive"
    PASSIVE_DIFF = "passive_diff"
    PASSIVE_TP = "passive_tp"
    PASSIVE_TP_SP = "passive_tp_sp"      # with secondary passivity
    PASSIVE_PITTING = "passive_pitting"   # sharp breakdown
    FULL = "full"                          # everything

    DESC = {
        "active":          ("⚡ Active Only",
                            "Pure activation kinetics on both branches. Standard Butler-Volmer."),
        "active_diff":     ("⚡🌊 Active + Diffusion-Limited",
                            "Active kinetics with cathodic mass-transport limitation."),
        "passive":         ("🛡️ Active–Passive",
                            "Active dissolution peak (nose) → passive plateau."),
        "passive_diff":    ("🛡️🌊 Active–Passive + Diffusion",
                            "Passivating system with cathodic diffusion limitation."),
        "passive_tp":      ("🛡️💥 Active–Passive–Transpassive",
                            "Passive plateau with transpassive dissolution above Eb."),
        "passive_tp_sp":   ("🛡️💥🟣 Full: Passive–Transpassive–Secondary Passive",
                            "Complete multi-region: active → passive → transpassive → secondary passivity."),
        "passive_pitting": ("🛡️⚡ Passive with Pitting Breakdown",
                            "Passive region terminated by sharp pitting at Epit."),
        "full":            ("🔀 Full Multi-Region",
                            "All regions detected: active, passive, transpassive, limiting, secondary."),
    }


# ═══════════════════════════════════════════════════════════════
# REGION DETECTION (comprehensive)
# ═══════════════════════════════════════════════════════════════

def detect_regions(E, i):
    """Comprehensive region detection with peak finding for active dissolution nose."""
    reg = {}; n = len(E); abs_i = np.abs(i)

    # ── Ecorr ───────────────────────────────────────────
    sc = np.where(np.diff(np.sign(i)))[0]
    if len(sc) > 0:
        k = sc[0]; d = i[k+1] - i[k]
        reg["Ecorr"] = float(E[k] - i[k]*(E[k+1]-E[k])/d) if abs(d) > 0 else float(E[k])
        reg["ecorr_idx"] = k
    else:
        k = int(np.argmin(abs_i))
        reg["Ecorr"] = float(E[k]); reg["ecorr_idx"] = k
    Ec = reg["Ecorr"]

    # ── Cathodic limiting ───────────────────────────────
    ci = np.where(E < Ec)[0]
    if len(ci) >= 8:
        lc = smooth(safe_log(abs_i[ci]), min(11, (len(ci)//2)*2-1 or 5))
        dl = np.abs(np.gradient(lc, E[ci]))
        thr = np.percentile(dl, 20)
        flat = dl < max(thr, 0.5)
        runs = [(k2, list(g)) for k2, g in groupby(enumerate(flat), key=lambda x: x[1]) if k2]
        if runs:
            br = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in br[1]]
            er = abs(E[ci[idxs[-1]]] - E[ci[idxs[0]]])
            far = abs(E[ci[idxs[0]]] - Ec) > 0.06
            if len(idxs) >= 4 and er > 0.03 and far:
                reg.update(iL=float(np.median(abs_i[ci[idxs]])),
                           E_lim_s=float(E[ci[idxs[0]]]),
                           E_lim_e=float(E[ci[idxs[-1]]]))

    # ── Anodic analysis ─────────────────────────────────
    ai = np.where(E > Ec)[0]
    if len(ai) < 6:
        reg["curve_type"] = CT.ACTIVE_DIFF if "iL" in reg else CT.ACTIVE
        return reg

    log_a = smooth(safe_log(abs_i[ai]), min(15, (len(ai)//2)*2-1 or 5))
    E_a = E[ai]
    dlog = np.gradient(log_a, E_a)

    # ── Find anodic peaks (active dissolution "nose") ───
    # A peak in |i| on the anodic side indicates active dissolution peak
    # Use smoothed current to find local maxima
    i_a_smooth = smooth(abs_i[ai], min(21, (len(ai)//2)*2-1 or 5))
    log_i_smooth = safe_log(i_a_smooth)

    # Find local maxima in log|i| on anodic side
    peaks = []
    if len(ai) > 20:
        order = max(3, len(ai)//20)
        peak_idx = argrelextrema(log_i_smooth, np.greater, order=order)[0]
        # Filter: peak must be significantly above the minimum after it
        for pk in peak_idx:
            # Look for a minimum after this peak
            rest = log_i_smooth[pk:]
            if len(rest) > 5:
                min_after = np.min(rest[3:])
                prominence = log_i_smooth[pk] - min_after
                if prominence > 0.3:  # at least 0.3 decades drop
                    peaks.append({
                        "idx": ai[pk], "E": float(E_a[pk]),
                        "log_i": float(log_i_smooth[pk]),
                        "prominence": float(prominence),
                    })

    reg["anodic_peaks"] = peaks

    # ── Passive region (flat sections in anodic log|i|) ──
    abs_dlog = np.abs(dlog)
    thr_p = np.percentile(abs_dlog, 25)
    flat = abs_dlog < max(thr_p, 0.8)
    runs = [(k2, list(g)) for k2, g in groupby(enumerate(flat), key=lambda x: x[1]) if k2]

    passive_regions = []
    if runs:
        for _, run_items in runs:
            idxs = [s[0] for s in run_items]
            er = abs(E_a[idxs[-1]] - E_a[idxs[0]])
            if len(idxs) >= 4 and er > 0.03:
                ps_g, pe_g = ai[idxs[0]], ai[idxs[-1]]
                i_med = float(np.median(abs_i[ps_g:pe_g+1]))
                passive_regions.append({
                    "ps": ps_g, "pe": pe_g,
                    "E_s": float(E[ps_g]), "E_e": float(E[pe_g]),
                    "ipass": i_med, "range": er,
                })

    # Validate passive regions: current should be lower than preceding peak
    valid_passive = []
    for pr in passive_regions:
        # Check if there's a peak before this passive region
        pre = np.where((E > Ec) & (E < pr["E_s"]))[0]
        if len(pre) > 2:
            i_peak = np.max(abs_i[pre])
            if pr["ipass"] < i_peak * 0.5:
                valid_passive.append(pr)
        elif pr["range"] > 0.08:  # wide enough to be passive even without clear peak
            valid_passive.append(pr)

    # ── Assign primary and secondary passive regions ─────
    if len(valid_passive) >= 1:
        reg["pass1"] = valid_passive[0]
        reg["Epp"] = valid_passive[0]["E_s"]
        reg["ipass"] = valid_passive[0]["ipass"]

    if len(valid_passive) >= 2:
        reg["pass2"] = valid_passive[1]  # secondary passivity
        reg["Esp"] = valid_passive[1]["E_s"]
        reg["ipass2"] = valid_passive[1]["ipass"]

    # ── Transpassive / breakdown ─────────────────────────
    if "pass1" in reg:
        pe = reg["pass1"]["pe"]
        if pe + 5 < n:
            d_after = np.gradient(safe_log(abs_i[pe:]), E[pe:])
            # Sharp rise = breakdown/transpassive
            thr_b = np.percentile(np.abs(d_after), 75)
            jump = np.where(np.abs(d_after) > max(thr_b, 2.0))[0]
            if len(jump):
                eb_idx = pe + jump[0]
                reg["Eb"] = float(E[eb_idx])
                reg["Eb_idx"] = eb_idx

                # Is it pitting (very sharp) or transpassive (gradual)?
                if len(jump) > 3:
                    slope_at_break = np.mean(np.abs(d_after[jump[:5]]))
                    reg["is_pitting"] = slope_at_break > 10  # >10 dec/V = likely pitting
                else:
                    reg["is_pitting"] = False

    # ── Classify curve type ─────────────────────────────
    has_diff = "iL" in reg
    has_pass = "pass1" in reg
    has_tp = "Eb" in reg
    has_sp = "pass2" in reg
    is_pit = reg.get("is_pitting", False)

    if has_sp:
        ct = CT.PASSIVE_TP_SP if has_tp else CT.FULL
    elif has_tp and is_pit:
        ct = CT.PASSIVE_PITTING
    elif has_tp:
        ct = CT.PASSIVE_TP if not has_diff else CT.FULL
    elif has_pass:
        ct = CT.PASSIVE_DIFF if has_diff else CT.PASSIVE
    elif has_diff:
        ct = CT.ACTIVE_DIFF
    else:
        ct = CT.ACTIVE

    reg["curve_type"] = ct
    return reg


# ═══════════════════════════════════════════════════════════════
# INITIAL GUESS BUILDER
# ═══════════════════════════════════════════════════════════════

def build_initial_guess(E, i, reg):
    """Extract initial parameter guesses from local analysis."""
    Ec = reg["Ecorr"]
    abs_i = np.abs(i)
    ct = reg["curve_type"]
    log_abs = safe_log(i)

    # ── Local Tafel slopes ──────────────────────────────
    ba0, bc0, icorr0 = 0.060, 0.120, max(abs_i[reg["ecorr_idx"]], 1e-10)

    # Anodic Tafel window search
    Epp = reg.get("Epp", Ec + 0.50)
    E_limit = min(Epp - 0.005, Ec + 0.25)
    best_an = None
    for lo in np.arange(0.005, 0.05, 0.005):
        for hi in np.arange(lo+0.015, min(lo+0.18, E_limit-Ec), 0.005):
            m = (E > Ec+lo) & (E < Ec+hi)
            if m.sum() < 4: continue
            s, b, r, *_ = linregress(E[m], log_abs[m])
            if s > 0 and 20 < (1/s)*1000 < 500 and r**2 > 0.88:
                if best_an is None or r**2 > best_an[2]:
                    best_an = (1/s, b, r**2, s)
    if best_an:
        ba0 = best_an[0]

    # Cathodic Tafel window search
    E_lim = reg.get("E_lim_e")
    best_ca = None
    for lo in np.arange(0.005, 0.08, 0.005):
        for hi in np.arange(lo+0.015, lo+0.25, 0.005):
            m = (E < Ec-lo) & (E > Ec-hi)
            if E_lim is not None: m = m & (E > E_lim + 0.005)
            if m.sum() < 4: continue
            s, b, r, *_ = linregress(E[m], log_abs[m])
            if s < 0 and 20 < (-1/s)*1000 < 500 and r**2 > 0.88:
                if best_ca is None or r**2 > best_ca[2]:
                    best_ca = (-1/s, b, r**2, s)
    if best_ca:
        bc0 = best_ca[0]

    # Refine icorr from Tafel intersection
    if best_an and best_ca:
        ds = best_an[3] - best_ca[3]
        if abs(ds) > 1e-10:
            Ei = (best_ca[1] - best_an[1]) / ds  # b_ca - b_an on y = s*E + b
            # Actually: best_an = (ba, intercept_an, r2, slope_an), best_ca = (bc, intercept_ca, r2, slope_ca)
            Ei = (best_ca[1] - best_an[1]) / (best_an[3] - best_ca[3])
            li = best_an[3] * Ei + best_an[1]
            icorr0 = 10**li

    # ── Build p0 array [14 params] ──────────────────────
    iL0 = reg.get("iL", 1e2)
    Epp0 = reg.get("Epp", Ec + 0.30)
    ipass0 = reg.get("ipass", 1e1)
    k_pass0 = 40.0  # ~25mV transition width

    Eb0 = reg.get("Eb", E[-1])
    a_tp0 = ipass0 if reg.get("Eb") else 1e-12
    b_tp0 = 8.0

    Esp0 = reg.get("Esp", E[-1] + 0.5)
    ipass2_0 = reg.get("ipass2", ipass0)
    k_sp0 = 30.0

    # Estimate transpassive params from data after Eb
    if reg.get("Eb") and not reg.get("pass2"):
        tp_mask = E > reg["Eb"]
        if tp_mask.sum() > 5:
            try:
                s, inter, *_ = linregress(E[tp_mask], safe_log(abs_i[tp_mask]))
                b_tp0 = max(abs(s) * 2.303, 1.0)
                a_tp0 = max(10**(inter - b_tp0/2.303 * reg["Eb"]), 1e-12)
            except: pass

    # For active-only systems, disable unused components
    if ct == CT.ACTIVE:
        k_pass0 = 0.01; ipass0 = 1e3; iL0 = 1e3
        a_tp0 = 1e-15; k_sp0 = 0.01; ipass2_0 = 1e3
    elif ct == CT.ACTIVE_DIFF:
        k_pass0 = 0.01; ipass0 = 1e3
        a_tp0 = 1e-15; k_sp0 = 0.01; ipass2_0 = 1e3

    p0 = np.array([Ec, icorr0, ba0, bc0, iL0,
                    Epp0, k_pass0, ipass0,
                    Eb0, a_tp0, b_tp0,
                    Esp0, k_sp0, ipass2_0])
    return p0


def build_bounds(E, i, reg, p0):
    """Build bounds for each parameter based on curve type."""
    ct = reg["curve_type"]
    Ec = p0[0]; ic = p0[1]
    has_pass = ct not in (CT.ACTIVE, CT.ACTIVE_DIFF)
    has_diff = "iL" in reg
    has_tp = "Eb" in reg
    has_sp = "pass2" in reg

    lo = [
        Ec - 0.15,                                          # Ecorr
        max(ic * 1e-4, 1e-14),                              # icorr
        0.010,                                               # ba
        0.010,                                               # bc
        reg.get("iL", 1e-4)*0.01 if has_diff else 1e0,     # iL
        p0[5] - 0.15 if has_pass else E[-1] - 0.1,         # Epp
        5.0 if has_pass else 0.001,                          # k_pass
        max(p0[7]*0.01, 1e-10) if has_pass else 1e0,        # ipass
        p0[8] - 0.20 if has_tp else E[-1] - 0.1,            # Eb
        max(p0[9]*1e-4, 1e-15),                              # a_tp
        0.5,                                                  # b_tp
        p0[11] - 0.20 if has_sp else E[-1],                  # Esp
        5.0 if has_sp else 0.001,                             # k_sp
        max(p0[13]*0.01, 1e-10) if has_sp else 1e0,          # ipass2
    ]
    hi = [
        Ec + 0.15,
        min(ic * 1e4, 1e0),
        0.500,
        0.500,
        reg.get("iL", 1e0)*100 if has_diff else 1e5,
        p0[5] + 0.15 if has_pass else E[-1] + 0.5,
        200.0 if has_pass else 1.0,
        p0[7]*100 if has_pass else 1e5,
        p0[8] + 0.20 if has_tp else E[-1] + 0.5,
        max(p0[9]*1e4, 1e-2),
        40.0,
        p0[11] + 0.20 if has_sp else E[-1] + 1.0,
        200.0 if has_sp else 1.0,
        p0[13]*100 if has_sp else 1e5,
    ]
    # Ensure lo < hi
    for idx in range(len(lo)):
        if lo[idx] >= hi[idx]:
            mid = (lo[idx] + hi[idx]) / 2
            lo[idx] = mid - abs(mid)*0.5 - 1e-6
            hi[idx] = mid + abs(mid)*0.5 + 1e-6
    return lo, hi


# ═══════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════

def _objective_log(p, E, log_data):
    """Log-space objective (main): treats all decades equally."""
    try:
        pred = global_model(E, p)
        log_pred = safe_log(pred)
        return float(np.sum((log_data - log_pred)**2))
    except:
        return 1e30


def _objective_combined(p, E, i_data, log_data):
    """Combined objective: log-space + weighted linear."""
    try:
        pred = global_model(E, p)
        log_pred = safe_log(pred)
        res_log = np.sum((log_data - log_pred)**2)
        res_rel = np.sum(((i_data - pred) / (np.abs(i_data) + 1e-12))**2) * 0.03
        return float(res_log + res_rel)
    except:
        return 1e30


class Optimizer:
    """Multi-stage global optimization engine."""

    def __init__(self, E, i, reg, p0, progress_callback=None):
        self.E = E
        self.i = i
        self.log_data = safe_log(i)
        self.reg = reg
        self.p0 = p0
        self.lo, self.hi = build_bounds(E, i, reg, p0)
        self.log = []
        self.best_p = None
        self.best_score = 1e30
        self.progress = progress_callback

    def _update_best(self, p, tag=""):
        score = _objective_log(p, self.E, self.log_data)
        r2 = _r2(self.log_data, safe_log(global_model(self.E, p)))
        if score < self.best_score:
            self.best_p = p.copy()
            self.best_score = score
            self.log.append(f"✅ {tag}: R²(log) = {r2:.6f}")
        else:
            self.log.append(f"ℹ️ {tag}: R²(log) = {r2:.6f} (no improvement)")
        return r2

    def _make_de_bounds(self):
        """Convert to log-space bounds for better DE exploration."""
        # Work in log-space for current-like parameters
        b = []
        for idx in range(N_PARAMS):
            if idx in (1, 4, 7, 9, 13):  # icorr, iL, ipass, a_tp, ipass2
                b.append((np.log10(max(self.lo[idx], 1e-15)),
                          np.log10(max(self.hi[idx], 1e-14))))
            else:
                b.append((self.lo[idx], self.hi[idx]))
        return b

    def _de_to_real(self, x):
        """Convert DE log-space params back to real."""
        p = x.copy()
        for idx in (1, 4, 7, 9, 13):
            p[idx] = 10**x[idx]
        return p

    def _real_to_de(self, p):
        """Convert real params to DE log-space."""
        x = p.copy()
        for idx in (1, 4, 7, 9, 13):
            x[idx] = np.log10(max(p[idx], 1e-15))
        return x

    def run_de(self, maxiter=4000, popsize=40):
        """Stage 1: Differential Evolution (global search)."""
        self.log.append("🔧 **Stage 1: Differential Evolution** (global search)")
        bounds = self._make_de_bounds()
        E, ld = self.E, self.log_data

        def obj(x):
            p = self._de_to_real(x)
            return _objective_combined(p, E, self.i, ld)

        try:
            t0 = time.time()
            res = differential_evolution(
                obj, bounds, seed=42, maxiter=maxiter, tol=1e-13,
                popsize=popsize, workers=1,
                mutation=(0.5, 1.8), recombination=0.9, polish=False)
            p = self._de_to_real(res.x)
            self._update_best(p, f"DE ({time.time()-t0:.1f}s, {res.nit} iter)")
        except Exception as ex:
            self.log.append(f"⚠️ DE failed: {ex}")

    def run_dual_annealing(self, maxiter=2000):
        """Stage 2: Dual Annealing (escape local minima)."""
        self.log.append("🔧 **Stage 2: Dual Annealing** (escape local minima)")
        bounds = self._make_de_bounds()
        E, ld = self.E, self.log_data
        x0 = self._real_to_de(self.best_p) if self.best_p is not None else self._real_to_de(self.p0)

        def obj(x):
            return _objective_log(self._de_to_real(x), E, ld)

        try:
            t0 = time.time()
            res = dual_annealing(
                obj, bounds, x0=x0, maxiter=maxiter, seed=42,
                initial_temp=5230, visit=2.62, restart_temp_ratio=2e-5)
            p = self._de_to_real(res.x)
            self._update_best(p, f"Dual Annealing ({time.time()-t0:.1f}s)")
        except Exception as ex:
            self.log.append(f"⚠️ Dual Annealing failed: {ex}")

    def run_lbfgsb(self):
        """Stage 3: L-BFGS-B gradient refinement."""
        self.log.append("🔧 **Stage 3: L-BFGS-B** (gradient refinement)")
        if self.best_p is None: return
        bounds = list(zip(self.lo, self.hi))
        E, ld = self.E, self.log_data

        try:
            t0 = time.time()
            res = minimize(lambda p: _objective_log(p, E, ld), self.best_p,
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12})
            self._update_best(res.x, f"L-BFGS-B ({time.time()-t0:.1f}s)")
        except Exception as ex:
            self.log.append(f"ℹ️ L-BFGS-B: {ex}")

    def run_nelder_mead(self):
        """Stage 4: Nelder-Mead (derivative-free polish)."""
        self.log.append("🔧 **Stage 4: Nelder-Mead** (final polish)")
        if self.best_p is None: return
        x0 = self._real_to_de(self.best_p)
        E, ld = self.E, self.log_data

        try:
            t0 = time.time()
            res = minimize(lambda x: _objective_log(self._de_to_real(x), E, ld), x0,
                          method="Nelder-Mead",
                          options={"maxiter": 100000, "xatol": 1e-14,
                                   "fatol": 1e-16, "adaptive": True})
            p = self._de_to_real(res.x)
            self._update_best(p, f"Nelder-Mead ({time.time()-t0:.1f}s)")
        except Exception as ex:
            self.log.append(f"ℹ️ NM: {ex}")

    def run_all(self):
        """Run full optimization pipeline."""
        self.best_p = self.p0.copy()
        self.best_score = _objective_log(self.p0, self.E, self.log_data)

        self.run_de()
        self.run_dual_annealing()
        self.run_lbfgsb()
        self.run_nelder_mead()

        if self.best_p is not None:
            r2 = _r2(self.log_data, safe_log(global_model(self.E, self.best_p)))
            self.log.append("─" * 50)
            if r2 >= 0.995:
                self.log.append(f"🎯 **Excellent global fit** — R²(log) = {r2:.6f}")
            elif r2 >= 0.970:
                self.log.append(f"✅ **Good global fit** — R²(log) = {r2:.6f}")
            elif r2 >= 0.90:
                self.log.append(f"⚠️ **Acceptable fit** — R²(log) = {r2:.6f}")
            else:
                self.log.append(f"❌ **Poor fit** — R²(log) = {r2:.6f}")

        return self.best_p


# ═══════════════════════════════════════════════════════════════
# DATA QUALITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def diagnose_data(E, i, reg, best_p, r2_log):
    """Analyze data quality and explain discrepancies."""
    issues = []
    abs_i = np.abs(i)
    log_i = safe_log(i)
    n = len(E)

    # 1. Noise level assessment
    if n > 20:
        log_sm = smooth(log_i, min(21, (n//4)*2-1 or 5))
        noise = np.std(log_i - log_sm)
        if noise > 0.5:
            issues.append(("🔴 High noise", f"Noise σ = {noise:.2f} decades. "
                          "Data is very noisy — consider using slower scan rate or "
                          "averaging multiple scans. High noise degrades fit quality."))
        elif noise > 0.15:
            issues.append(("🟡 Moderate noise", f"Noise σ = {noise:.2f} decades. "
                          "Acceptable but could be improved with slower scan rate."))
        else:
            issues.append(("🟢 Low noise", f"Noise σ = {noise:.3f} decades. Excellent signal quality."))

    # 2. Scan rate / IR drop check
    # Large Tafel slopes (>200 mV/dec) can indicate IR drop
    if best_p is not None:
        ba_mV = best_p[2] * 1000
        bc_mV = best_p[3] * 1000
        if ba_mV > 200:
            issues.append(("🟡 Large anodic Tafel slope",
                          f"βa = {ba_mV:.0f} mV/dec is unusually high. Possible causes: "
                          "uncompensated IR drop, multi-step reaction mechanism, "
                          "or mixed potential effects. Consider IR compensation."))
        if bc_mV > 200:
            issues.append(("🟡 Large cathodic Tafel slope",
                          f"βc = {bc_mV:.0f} mV/dec is unusually high. "
                          "May indicate diffusion contribution in Tafel region, "
                          "IR drop, or reaction involving multiple electrons."))

    # 3. Data range assessment
    E_range = E[-1] - E[0]
    if E_range < 0.3:
        issues.append(("🟡 Narrow potential range",
                      f"Only {E_range*1000:.0f} mV scanned. May not capture all "
                      "regions (passive, transpassive). Consider wider scan."))

    # 4. Data density
    pts_per_V = n / max(E_range, 0.01)
    if pts_per_V < 50:
        issues.append(("🟡 Low data density",
                      f"{pts_per_V:.0f} points/V. Denser sampling improves "
                      "Tafel region identification. Use ≥100 pts/V."))

    # 5. Ecorr region quality
    ec = reg["Ecorr"]
    near_ec = np.abs(E - ec) < 0.05
    if near_ec.sum() < 5:
        issues.append(("🟡 Sparse data near Ecorr",
                      f"Only {near_ec.sum()} points within ±50 mV of Ecorr. "
                      "Denser sampling near Ecorr improves icorr accuracy."))

    # 6. Asymmetry check
    cat_range = ec - E[0]
    an_range = E[-1] - ec
    if cat_range > 0 and an_range > 0:
        ratio = an_range / cat_range
        if ratio > 3 or ratio < 0.33:
            issues.append(("🟡 Asymmetric scan range",
                          f"Cathodic: {cat_range*1000:.0f} mV, Anodic: {an_range*1000:.0f} mV. "
                          "Highly asymmetric — one branch may be poorly characterized."))

    # 7. Residual pattern analysis (systematic errors)
    if best_p is not None:
        pred = global_model(E, best_p)
        residuals = log_i - safe_log(pred)
        # Check for systematic patterns (autocorrelation)
        if len(residuals) > 10:
            # Run-length test: too many same-sign runs → systematic misfit
            signs = np.sign(residuals)
            n_runs = np.sum(np.abs(np.diff(signs)) > 0) + 1
            expected_runs = n / 2
            if n_runs < expected_runs * 0.5:
                issues.append(("🟡 Systematic residual pattern",
                              f"Only {n_runs} sign changes in residuals (expected ~{expected_runs:.0f}). "
                              "Indicates model may be missing a physical feature in the data "
                              "(e.g., an additional reaction, adsorption peak, or "
                              "intermediate passivation). Inspect the residual plot."))

            # Check for large residuals in specific regions
            for region_name, mask_fn in [
                ("near Ecorr", lambda: np.abs(E - ec) < 0.05),
                ("cathodic", lambda: E < ec - 0.1),
                ("passive", lambda: (E > reg.get("Epp", 99)) &
                 (E < reg.get("Eb", E[-1]))),
            ]:
                m = mask_fn()
                if m.sum() > 3:
                    rmse_region = np.sqrt(np.mean(residuals[m]**2))
                    if rmse_region > 0.5:
                        issues.append(("🟡 Poor fit in " + region_name,
                                      f"RMSE = {rmse_region:.3f} decades in {region_name} region. "
                                      "Model may not capture the exact behavior here."))

    # 8. Physically unreasonable parameters
    if best_p is not None:
        icorr = best_p[1]
        if icorr > 1e-1:
            issues.append(("🔴 Very high icorr",
                          f"icorr = {icorr:.2e} A/cm² is extremely high. "
                          "Check electrode area or unit conversion."))
        if icorr < 1e-12:
            issues.append(("🟡 Very low icorr",
                          f"icorr = {icorr:.2e} A/cm² is unusually low. "
                          "Check if the data spans Ecorr properly."))

    # 9. Fit quality summary
    if r2_log is not None:
        if r2_log < 0.90:
            issues.append(("🔴 Poor overall fit",
                          f"R²(log) = {r2_log:.4f}. The data may contain features "
                          "not captured by standard electrochemical models: "
                          "hydrogen evolution/oxidation coupling, adsorption pseudocapacitance, "
                          "surface roughness effects, or experimental artifacts."))

    return issues


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_main(E, i, best_p, reg, ct):
    log_i = safe_log(i)
    fig = go.Figure()

    # Region fills
    if "pass1" in reg:
        p = reg["pass1"]
        fig.add_vrect(x0=p["E_s"], x1=p["E_e"], fillcolor=C["passive"],
                      layer="below", line_width=0,
                      annotation=dict(text="Passive", font=dict(color="#a6e3a1", size=11),
                                      yanchor="top"))
    if "pass2" in reg:
        p = reg["pass2"]
        fig.add_vrect(x0=p["E_s"], x1=p["E_e"],
                      fillcolor=C["sec_passive"], layer="below", line_width=0,
                      annotation=dict(text="2nd Passive", font=dict(color="#cba6f7", size=10),
                                      yanchor="top"))
    if "E_lim_s" in reg:
        fig.add_vrect(x0=reg["E_lim_s"], x1=reg["E_lim_e"],
                      fillcolor=C["limiting"], layer="below", line_width=0,
                      annotation=dict(text="Limiting", font=dict(color="#89dceb", size=11),
                                      yanchor="top"))
    if reg.get("Eb"):
        tp_end = reg.get("Esp", E[-1])
        fig.add_vrect(x0=reg["Eb"], x1=tp_end,
                      fillcolor=C["transpassive"], layer="below", line_width=0,
                      annotation=dict(text="Transpassive", font=dict(color="#fab387", size=10),
                                      yanchor="top"))

    # Key lines
    Ec = reg["Ecorr"]
    fig.add_vline(x=Ec, line=dict(color=C["ecorr"], width=1.5, dash="dot"),
                  annotation=dict(text="Ecorr", font=dict(color=C["ecorr"], size=10)))
    if reg.get("Epp"):
        fig.add_vline(x=reg["Epp"], line=dict(color="#a6e3a1", width=1, dash="dot"),
                      annotation=dict(text="Epp", font=dict(color="#a6e3a1", size=10)))
    if reg.get("Eb"):
        fig.add_vline(x=reg["Eb"], line=dict(color="#f38ba8", width=1, dash="dash"),
                      annotation=dict(text="Eb", font=dict(color="#f38ba8", size=10)))

    # Measured data
    fig.add_trace(go.Scatter(x=E, y=log_i, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5)))

    # Global fit
    if best_p is not None:
        E_m = np.linspace(E.min(), E.max(), 800)
        try:
            i_m = global_model(E_m, best_p)
            r2 = _r2(log_i, safe_log(global_model(E, best_p)))
            fig.add_trace(go.Scatter(
                x=E_m, y=safe_log(i_m), mode="lines",
                name=f"Global Fit  R²(log)={r2:.4f}",
                line=dict(color=C["fit"], width=3)))
        except: pass

        # icorr marker
        fig.add_trace(go.Scatter(
            x=[best_p[0]], y=[np.log10(max(best_p[1], 1e-20))],
            mode="markers", name=f"icorr = {best_p[1]:.3e} A/cm²",
            marker=dict(symbol="x-thin", size=18, color=C["ecorr"],
                        line=dict(width=4, color=C["ecorr"]))))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Potentiodynamic Polarization — Global Fit",
                   font=dict(size=17, color=C["text"])),
        xaxis=dict(title="Potential (V vs Ref)", gridcolor=C["grid"], color=C["text"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)", gridcolor=C["grid"], color=C["text"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11), x=0.01, y=0.01),
        height=540, margin=dict(l=70, r=20, t=50, b=60), hovermode="x unified")
    return fig


def plot_components(E, best_p, reg, ct):
    if best_p is None: return None
    E_m = np.linspace(E.min(), E.max(), 800)
    comp = global_model_components(E_m, best_p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E_m, y=safe_log(comp["i_cathodic"]),
                             mode="lines", name="Cathodic",
                             line=dict(color=C["cathodic"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=E_m, y=safe_log(comp["i_active"]),
                             mode="lines", name="Active dissolution (Tafel)",
                             line=dict(color=C["anodic"], width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=E_m, y=safe_log(comp["i_after_pass1"]),
                             mode="lines", name="After primary passivation",
                             line=dict(color="#a6e3a1", width=1.5, dash="dot")))
    if ct not in (CT.ACTIVE, CT.ACTIVE_DIFF):
        fig.add_trace(go.Scatter(x=E_m, y=safe_log(comp["i_tp"]),
                                 mode="lines", name="Transpassive",
                                 line=dict(color="#fab387", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=E_m, y=safe_log(comp["i_total"]),
                             mode="lines", name="Total (model)",
                             line=dict(color=C["fit"], width=3)))

    # Film coverage
    fig.add_trace(go.Scatter(x=E_m, y=comp["theta1"]*3 - 12,
                             mode="lines", name="θ₁ (primary film)",
                             line=dict(color="#94e2d5", width=1, dash="dash"),
                             yaxis="y2"))
    if ct in (CT.PASSIVE_TP_SP, CT.FULL):
        fig.add_trace(go.Scatter(x=E_m, y=comp["theta2"]*3 - 12,
                                 mode="lines", name="θ₂ (secondary film)",
                                 line=dict(color="#cba6f7", width=1, dash="dash"),
                                 yaxis="y2"))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Model Component Breakdown", font=dict(size=15, color=C["text"])),
        xaxis=dict(title="Potential (V)", gridcolor=C["grid"], color=C["text"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)", gridcolor=C["grid"], color=C["text"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=10)),
        height=420, margin=dict(l=70, r=20, t=50, b=60))
    return fig


def plot_residuals(E, i, best_p):
    if best_p is None: return None
    log_d = safe_log(i)
    pred = global_model(E, best_p)
    log_p = safe_log(pred)
    res = log_d - log_p
    rmse = np.sqrt(np.mean(res**2))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.06,
                        subplot_titles=("Data vs Global Fit", "Residuals (log-space)"))

    fig.add_trace(go.Scatter(x=E, y=log_d, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2)), row=1, col=1)
    E_m = np.linspace(E.min(), E.max(), 600)
    fig.add_trace(go.Scatter(x=E_m, y=safe_log(global_model(E_m, best_p)),
                             mode="lines", name="Global Fit",
                             line=dict(color=C["fit"], width=2.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=E, y=res, mode="lines", name="Residual",
                             line=dict(color="#fab387", width=1),
                             fill="tozeroy", fillcolor="rgba(250,179,135,0.06)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#585b70", width=1, dash="dot"), row=2)

    fig.add_annotation(text=f"RMSE(log) = {rmse:.4f}",
                       xref="paper", yref="paper", x=0.98, y=0.38,
                       showarrow=False, font=dict(color="#a6adc8", size=11),
                       bgcolor="rgba(19,19,32,0.8)", bordercolor=C["grid"])

    fig.update_layout(template="plotly_dark", plot_bgcolor=C["bg"],
                      paper_bgcolor=C["paper"], height=440,
                      margin=dict(l=70, r=20, t=40, b=60), showlegend=True,
                      legend=dict(bgcolor="rgba(19,19,32,0.9)", bordercolor=C["grid"],
                                  font=dict(color=C["text"], size=11)))
    fig.update_yaxes(gridcolor=C["grid"], color=C["text"])
    fig.update_xaxes(gridcolor=C["grid"], color=C["text"], title_text="Potential (V)", row=2)
    return fig


# ═══════════════════════════════════════════════════════════════
# PARAMETER DISPLAY
# ═══════════════════════════════════════════════════════════════

def pcard(label, val, unit="", color="#cdd6f4"):
    if val is None: disp = "—"
    elif isinstance(val, str): disp = val
    elif abs(val) < 0.001 or abs(val) > 9999: disp = f"{val:.4e}"
    else: disp = f"{val:.4f}"
    st.markdown(f'<div class="pcard"><div class="plabel">{label}</div>'
                f'<div class="pval" style="color:{color}">{disp}</div>'
                f'<div class="punit">{unit}</div></div>', unsafe_allow_html=True)


def show_params(best_p, reg, ew, rho, ct, r2_log):
    if best_p is None: return
    p = dict(zip(PARAM_NAMES, best_p))

    # Derived quantities
    ba, bc = p["ba"], p["bc"]
    B = (ba*bc) / (2.303*(ba+bc)) if ba > 0 and bc > 0 else None
    ic = p["icorr"]
    CR = ic * 3.27 * ew / rho if ic else None

    # Badges
    badges = []
    if "iL" in reg: badges.append('<span class="badge bb">Limiting</span>')
    if "pass1" in reg: badges.append('<span class="badge bg">Passive</span>')
    if reg.get("Eb"): badges.append('<span class="badge br">Breakdown</span>')
    if "pass2" in reg: badges.append('<span class="badge bp">2nd Passive</span>')
    if reg.get("anodic_peaks"):
        badges.append(f'<span class="badge by">{len(reg["anodic_peaks"])} anodic peak(s)</span>')
    if badges:
        st.markdown("**Detected:** " + "".join(badges), unsafe_allow_html=True)

    title, desc = CT.DESC.get(ct, ("Unknown", ""))
    clr = {"active":"#f9e2af","active_diff":"#89dceb","passive":"#a6e3a1",
           "passive_diff":"#94e2d5","passive_tp":"#fab387","passive_tp_sp":"#cba6f7",
           "passive_pitting":"#f38ba8","full":"#f5c2e7"}.get(ct, "#cdd6f4")
    st.markdown(f'<div class="type-box"><div class="type-title" style="color:{clr}">{title}</div>'
                f'<div class="type-desc">{desc}</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="sechead">⚡ Corrosion</div>', unsafe_allow_html=True)
        pcard("Ecorr", p["Ecorr"], "V vs Ref", "#f38ba8")
        pcard("icorr", p["icorr"], "A cm⁻²", "#fab387")
        if CR: pcard("Corrosion rate", CR, "mm yr⁻¹", "#eba0ac")
        if B: pcard("B (Stern-Geary)", B*1000, "mV", "#89dceb")

    with c2:
        st.markdown('<div class="sechead">📐 Kinetics</div>', unsafe_allow_html=True)
        pcard("βa anodic", ba*1000, "mV dec⁻¹", "#a6e3a1")
        pcard("βc cathodic", bc*1000, "mV dec⁻¹", "#94e2d5")
        cls = "ok-box" if r2_log and r2_log >= 0.97 else "warn-box"
        pfx = "✅" if r2_log and r2_log >= 0.97 else "⚠️"
        st.markdown(f'<div class="{cls}">{pfx} R²(log) = {r2_log:.5f}</div>',
                    unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="sechead">🛡️ Passivity</div>', unsafe_allow_html=True)
        if ct not in (CT.ACTIVE, CT.ACTIVE_DIFF):
            pcard("Epp", p["Epp"], "V", "#a6e3a1")
            pcard("ipass", p["ipass"], "A cm⁻²", "#94e2d5")
            pcard("k_pass (sharpness)", p["k_pass"], "V⁻¹", "#a6adc8")
            if ct in (CT.PASSIVE_TP_SP, CT.FULL):
                pcard("Esp (2nd passive)", p["Esp"], "V", "#cba6f7")
                pcard("ipass₂", p["ipass2"], "A cm⁻²", "#cba6f7")
        else:
            st.info("No passivation detected.")

    with c4:
        st.markdown('<div class="sechead">💥 Breakdown & Diffusion</div>', unsafe_allow_html=True)
        if reg.get("Eb"):
            pcard("Eb breakdown", p["Eb"], "V", "#f38ba8")
            pcard("Eb−Ecorr", (p["Eb"]-p["Ecorr"])*1000, "mV", "#f38ba8")
            pcard("a_tp", p["a_tp"], "A cm⁻²", "#fab387")
            pcard("b_tp", p["b_tp"], "V⁻¹", "#fab387")
        if ct in (CT.ACTIVE_DIFF, CT.PASSIVE_DIFF, CT.FULL, CT.PASSIVE_TP_SP):
            pcard("iL (fitted)", p["iL"], "A cm⁻²", "#89dceb")
            if reg.get("iL"):
                pcard("iL (detected)", reg["iL"], "A cm⁻²", "#89b4fa")


# ═══════════════════════════════════════════════════════════════
# MATERIALS
# ═══════════════════════════════════════════════════════════════
MATERIALS = {
    "Carbon Steel / Iron": (27.92, 7.87), "304 Stainless Steel": (25.10, 7.90),
    "316 Stainless Steel": (25.56, 8.00), "Copper": (31.77, 8.96),
    "Aluminum": (8.99, 2.70), "Nickel": (29.36, 8.91),
    "Titanium": (11.99, 4.51), "Zinc": (32.69, 7.14), "Custom": (27.92, 7.87),
}


# ═══════════════════════════════════════════════════════════════
# DEMO DATA (using the global model itself for perfect test cases)
# ═══════════════════════════════════════════════════════════════

def make_demo(choice):
    np.random.seed(42)
    if "secondary" in choice.lower() or "full" in choice.lower():
        # Full: active peak + passive + transpassive + secondary passive
        E = np.linspace(-0.65, 1.0, 600)
        p = [-0.38, 3e-6, 0.060, 0.110, 5e-4,
             -0.10, 50.0, 4e-6,
             0.50, 2e-6, 10.0,
             0.80, 40.0, 3e-6]
        i = global_model(E, p)
    elif "pitting" in choice.lower():
        E = np.linspace(-0.55, 0.55, 400)
        p = [-0.35, 2e-6, 0.055, 0.120, 1e3,
             -0.05, 60.0, 3e-6,
             0.35, 1e-4, 25.0,
             1.5, 0.01, 1e3]
        i = global_model(E, p)
    elif "passive" in choice.lower() and "trans" in choice.lower():
        E = np.linspace(-0.55, 0.75, 450)
        p = [-0.38, 2e-6, 0.065, 0.110, 1e3,
             -0.10, 45.0, 5e-6,
             0.55, 3e-6, 8.0,
             1.5, 0.01, 1e3]
        i = global_model(E, p)
    elif "passive" in choice.lower():
        E = np.linspace(-0.55, 0.50, 400)
        p = [-0.40, 2e-6, 0.065, 0.110, 1e3,
             -0.15, 50.0, 4e-6,
             0.8, 1e-15, 1.0,
             1.5, 0.01, 1e3]
        i = global_model(E, p)
    elif "diffusion" in choice.lower():
        E = np.linspace(-0.80, 0.30, 400)
        p = [-0.40, 3e-6, 0.070, 0.120, 2e-4,
             0.8, 0.01, 1e3,
             1.5, 1e-15, 1.0,
             2.0, 0.01, 1e3]
        i = global_model(E, p)
    else:  # active only
        E = np.linspace(-0.65, 0.40, 350)
        p = [-0.45, 5e-6, 0.060, 0.120, 1e3,
             0.8, 0.01, 1e3,
             1.5, 1e-15, 1.0,
             2.0, 0.01, 1e3]
        i = global_model(E, p)

    noise = np.random.normal(0, np.abs(i)*0.03 + 3e-9, len(i))
    return E, i + noise


# ═══════════════════════════════════════════════════════════════
# SUMMARY CSV
# ═══════════════════════════════════════════════════════════════

def build_csv(best_p, reg, ew, rho, ct, r2_log, diags):
    p = dict(zip(PARAM_NAMES, best_p)) if best_p is not None else {}
    ba, bc = p.get("ba",0), p.get("bc",0)
    B = (ba*bc)/(2.303*(ba+bc)) if ba > 0 and bc > 0 else None
    ic = p.get("icorr")
    CR = ic*3.27*ew/rho if ic else None

    rows = [("Curve type", CT.DESC.get(ct,("?",""))[0]),
            ("Ecorr (V)", p.get("Ecorr")),
            ("icorr (A/cm²)", ic),
            ("βa (mV/dec)", ba*1000 if ba else None),
            ("βc (mV/dec)", bc*1000 if bc else None),
            ("B (mV)", B*1000 if B else None),
            ("CR (mm/yr)", CR),
            ("ipass (A/cm²)", p.get("ipass")),
            ("Epp (V)", p.get("Epp")),
            ("k_pass (V⁻¹)", p.get("k_pass")),
            ("iL (A/cm²)", p.get("iL")),
            ("Eb (V)", p.get("Eb")),
            ("a_tp", p.get("a_tp")),
            ("b_tp (V⁻¹)", p.get("b_tp")),
            ("Esp (V)", p.get("Esp")),
            ("ipass2 (A/cm²)", p.get("ipass2")),
            ("k_sp (V⁻¹)", p.get("k_sp")),
            ("R²(log)", r2_log)]
    for sev, msg in diags:
        rows.append((sev, msg))
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_data(E, i_dens, area, ew, rho):
    # ── Detect ──────────────────────────────────────────
    with st.spinner("Detecting regions & classifying curve…"):
        reg = detect_regions(E, i_dens)
    ct = reg["curve_type"]

    # ── Initial guess ───────────────────────────────────
    with st.spinner("Building initial parameter estimates…"):
        p0 = build_initial_guess(E, i_dens, reg)

    # ── Optimize ────────────────────────────────────────
    with st.spinner("Running multi-stage global optimization…"):
        opt = Optimizer(E, i_dens, reg, p0)
        best_p = opt.run_all()

    r2_log = _r2(safe_log(i_dens), safe_log(global_model(E, best_p))) if best_p is not None else None

    # ── Diagnostics ─────────────────────────────────────
    diags = diagnose_data(E, i_dens, reg, best_p, r2_log)

    # ═══════════════════════════════════════════════════
    # DISPLAY
    # ═══════════════════════════════════════════════════
    st.markdown("---")

    # Main plot
    st.plotly_chart(plot_main(E, i_dens, best_p, reg, ct), use_container_width=True)

    # Two-column: components + residuals
    c1, c2 = st.columns(2)
    with c1:
        fig_comp = plot_components(E, best_p, reg, ct)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
    with c2:
        fig_res = plot_residuals(E, i_dens, best_p)
        if fig_res:
            st.plotly_chart(fig_res, use_container_width=True)

    # Parameters
    st.markdown("---")
    show_params(best_p, reg, ew, rho, ct, r2_log)

    # ── Data Quality Diagnostics ────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Data Quality Diagnostics")
    for severity, message in diags:
        if "🔴" in severity:
            st.markdown(f'<div class="err-box"><b>{severity}</b><br>{message}</div>',
                        unsafe_allow_html=True)
        elif "🟡" in severity:
            st.markdown(f'<div class="warn-box"><b>{severity}</b><br>{message}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box"><b>{severity}</b><br>{message}</div>',
                        unsafe_allow_html=True)

    # ── Model equation ──────────────────────────────────
    if best_p is not None:
        with st.expander("📐 Global model equation & physics"):
            p = dict(zip(PARAM_NAMES, best_p))
            st.markdown(f"""
**Film-Coverage Global Polarization Model:**

The model uses a film-coverage approach where passive films progressively
block active dissolution sites:

**Cathodic** (activation → diffusion limited):
```
i_c_kinetic = icorr · exp(−2.303·η / βc)
i_cathodic  = i_c_kinetic / (1 + i_c_kinetic / iL)
```
iL = **{p['iL']:.3e}** A/cm²

**Anodic** (active dissolution):
```
i_active = icorr · exp(2.303·η / βa)
```
This produces the exponential rise from Ecorr (the "active region").

**Primary passivation** (film coverage θ₁):
```
θ₁(E) = sigmoid(k₁ · (E − Epp))
i_after_pass = i_active · (1−θ₁) + ipass · θ₁
```
- Below Epp: θ₁ ≈ 0 → pure Tafel kinetics
- At Epp: θ₁ transitions → produces the **active dissolution peak** ("nose")
- Above Epp: θ₁ ≈ 1 → current drops to ipass = **{p['ipass']:.3e}** A/cm²
- k₁ = **{p['k_pass']:.1f}** V⁻¹ controls sharpness of the peak

**Transpassive dissolution** (above Eb):
```
i_tp = a_tp · exp(b_tp · (E − Eb)) · sigmoid(E − Eb)
```
Eb = **{p['Eb']:.3f}** V

**Secondary passivation** (film coverage θ₂):
```
θ₂(E) = sigmoid(k₂ · (E − Esp))
i_total_anodic = (i_after_pass + i_tp) · (1−θ₂) + ipass₂ · θ₂
```
Esp = **{p['Esp']:.3f}** V, ipass₂ = **{p['ipass2']:.3e}** A/cm²

**Net current:** `i_net = i_total_anodic − i_cathodic`
""")

    # ── Fitting log ─────────────────────────────────────
    with st.expander("🪵 Optimization log"):
        for msg in opt.log:
            st.markdown(f"- {msg}")

    # ── Downloads ───────────────────────────────────────
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        df_csv = build_csv(best_p, reg, ew, rho, ct, r2_log, diags)
        st.download_button("⬇️ Results + Diagnostics (CSV)",
                           df_csv.to_csv(index=False).encode(),
                           "tafel_results.csv", "text/csv", use_container_width=True)
    with c_dl2:
        df_d = pd.DataFrame({"E_V": E, "i_Acm2": i_dens, "log_abs_i": safe_log(i_dens)})
        st.download_button("⬇️ Processed data (CSV)",
                           df_d.to_csv(index=False).encode(),
                           "tafel_data.csv", "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# APP ENTRY
# ═══════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1e2e,#131320);
                border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px">⚡ Tafel Fitting Tool v4</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px">
        Global multi-region model · Film-coverage physics · All curve types · Data diagnostics
      </p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        area = st.number_input("Electrode area (cm²)", 0.001, 100.0, 1.0, 0.01)
        mat = st.selectbox("Material", list(MATERIALS.keys()))
        ew0, rho0 = MATERIALS[mat]
        ew, rho = (st.number_input("EW", 1.0, 300.0, ew0),
                   st.number_input("ρ", 0.5, 25.0, rho0)) if mat == "Custom" else (ew0, rho0)

        st.divider()
        st.markdown("""<div style="font-size:11px;color:#a6adc8;line-height:2.0">
        <b style="color:#89b4fa">Curve types auto-detected:</b><br>
        🟡 Active only<br>🔵 Active + diffusion-limited<br>
        🟢 Active → passive (with peak)<br>🟠 + transpassive<br>
        🟣 + secondary passivity<br>🔴 Pitting breakdown<br>
        🔀 Full multi-region
        </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload polarization data", type=["csv","txt","xlsx","xls"])

    if uploaded is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            demo = st.selectbox("Or try a demo", [
                "Active only", "Diffusion-limited cathodic",
                "Active → Passive", "Passive + Transpassive",
                "Passive + Transpassive + Pitting",
                "Full: Passive + Transpassive + Secondary Passivity"])
            if st.button("▶  Run demo", use_container_width=True, type="primary"):
                st.session_state["demo"] = demo
            st.markdown("""<br>
            <div style="background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:16px 20px">
            <b style="color:#89b4fa">v4 — What's new</b><br><br>
            <span style="color:#a6adc8;font-size:13px">
            ✓ <b>Film-coverage model</b> — produces the active dissolution peak (nose)<br>
            ✓ <b>Secondary passivity</b> support (θ₂ term)<br>
            ✓ <b>Pitting detection</b> (sharp vs. gradual breakdown)<br>
            ✓ <b>4-stage optimization</b>: DE → Dual Annealing → L-BFGS-B → Nelder-Mead<br>
            ✓ <b>Data quality diagnostics</b> — explains WHY fit may be poor<br>
            ✓ All params extracted for the specific curve type detected<br>
            ✓ Component breakdown with film coverage visualization
            </span></div>""", unsafe_allow_html=True)
        return

    with st.spinner("Reading…"):
        try: df = load_any_file(uploaded)
        except Exception as ex: st.error(f"Read error: {ex}"); return
    with st.spinner("Detecting columns…"):
        try: e_col, i_col, i_factor = auto_detect_columns(df)
        except Exception as ex: st.error(f"Column error: {ex}"); return

    with st.expander(f"📋 Detected: **{e_col}** · **{i_col}**", expanded=False):
        st.dataframe(df[[e_col, i_col]].head(10), use_container_width=True)

    E = df[e_col].values.astype(float)
    ir = df[i_col].values.astype(float) * i_factor
    ok = np.isfinite(E) & np.isfinite(ir)
    E, ir = E[ok], ir[ok]
    idx = np.argsort(E); E, ir = E[idx], ir[idx]
    process_data(E, ir / area, area, ew, rho)


if __name__ == "__main__":
    if "demo" in st.session_state:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e1e2e,#131320);
                    border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">
          <h1 style="margin:0;color:#cdd6f4;font-size:26px">⚡ Tafel v4</h1>
          <p style="margin:4px 0 0;color:#6c7086;font-size:13px">Demo</p>
        </div>""", unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("### ⚙️")
            area = st.number_input("Area (cm²)", 0.001, 100.0, 1.0)
            mat = st.selectbox("Material", list(MATERIALS.keys()))
            ew, rho = MATERIALS[mat]
            if st.button("← Back"): del st.session_state["demo"]; st.rerun()

        E_d, i_d = make_demo(st.session_state["demo"])
        st.info(f"🧪 **{st.session_state['demo']}**")
        process_data(E_d, i_d / area, area, ew, rho)
    else:
        main()
