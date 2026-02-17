"""
Tafel Fitting Tool — v3: Global Multi-Region Electrochemical Model
===================================================================
Auto-detects curve type → extracts initial guesses from local analysis →
fits a GLOBAL model across the ENTIRE polarization curve (active + passive
+ transpassive + diffusion-limited) → single fit line overlays all data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.signal import savgol_filter
from scipy.stats import linregress
from itertools import groupby
import warnings, io, re

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Tafel Fitting v3", page_icon="⚡", layout="wide")

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"]  { background:#13131f; color:#cdd6f4; }
[data-testid="stSidebar"]                 { background:#1a1a2e; }
section[data-testid="stFileUploadDropzone"] { background:#1e1e2e !important;
    border:2px dashed #45475a !important; border-radius:12px !important; }
.pcard { background:#1e1e2e; border:1px solid #313244; border-radius:10px;
         padding:14px 16px; margin:4px 0; }
.plabel { color:#a6adc8; font-size:10px; font-weight:700; letter-spacing:.8px;
          text-transform:uppercase; }
.pval   { font-size:21px; font-weight:700; margin:1px 0; }
.punit  { color:#585b70; font-size:11px; }
.sechead { color:#89b4fa; font-size:12px; font-weight:700; letter-spacing:1px;
           text-transform:uppercase; border-bottom:1px solid #313244;
           padding-bottom:5px; margin:14px 0 6px; }
.badge  { display:inline-block; padding:2px 9px; border-radius:20px;
          font-size:10px; font-weight:700; margin:1px 2px; }
.bg  { background:#1c3a2f; color:#a6e3a1; border:1px solid #a6e3a1; }
.bb  { background:#1a2a3f; color:#89b4fa; border:1px solid #89b4fa; }
.by  { background:#3a3020; color:#f9e2af; border:1px solid #f9e2af; }
.br  { background:#3a1a20; color:#f38ba8; border:1px solid #f38ba8; }
.bp  { background:#2a1a3a; color:#cba6f7; border:1px solid #cba6f7; }
.ok-box  { background:#1e1e2e; border-left:4px solid #a6e3a1; border-radius:0 8px 8px 0;
           padding:8px 14px; margin:6px 0; font-size:12px; color:#cdd6f4; }
.warn-box{ background:#1e1e2e; border-left:4px solid #f9e2af; border-radius:0 8px 8px 0;
           padding:8px 14px; margin:6px 0; font-size:12px; color:#f9e2af; }
.type-box{ background:linear-gradient(135deg,#1e1e2e,#232336);
           border:1px solid #45475a; border-radius:10px;
           padding:14px 18px; margin:8px 0; }
.type-title{ font-size:16px; font-weight:700; margin-bottom:4px; }
.type-desc { font-size:12px; color:#a6adc8; }
</style>
""", unsafe_allow_html=True)

C = dict(
    data="#89b4fa", anodic="#f9e2af", cathodic="#cba6f7", bv="#a6e3a1",
    global_fit="#a6e3a1",
    passive="rgba(166,227,161,0.12)", limiting="rgba(137,220,235,0.12)",
    transpassive="rgba(243,188,168,0.10)", ecorr="#f38ba8",
    grid="#313244", bg="#1e1e2e", paper="#181825", text="#cdd6f4",
    fit_band="rgba(166,227,161,0.10)",
)


# ═══════════════════════════════════════════════════════════════
# COLUMN AUTO-DETECTION & FILE LOADING
# ═══════════════════════════════════════════════════════════════
COLUMN_SIGNATURES = [
    (r"we.*potential", r"we.*current", "A"),
    (r"ewe",           r"i/ma",        "mA"),
    (r"ewe",           r"<i>/ma",      "mA"),
    (r"^vf$",          r"^im$",        "A"),
    (r"potential/v",   r"current/a",   "A"),
    (r"e/v",           r"i/a",         "A"),
    (r"potential|volt|^e$|e \(v\)|e_v",
     r"current|amps|^i$|i \(a\)|i_a",  "A"),
    (r"potential|volt|^e$",
     r"current.*ma|ima",               "mA"),
]
UNIT_HINTS = {
    r"\(a\)|_a$|/a$": 1.0, r"\(ma\)|_ma$|/ma$": 1e-3,
    r"\(µa\)|_ua$|/ua$": 1e-6, r"a/cm": 1.0, r"ma/cm": 1e-3,
}


def auto_detect_columns(df):
    cols_lower = {c: c.lower().strip() for c in df.columns}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for E_pat, I_pat, unit in COLUMN_SIGNATURES:
        e_match = [c for c, cl in cols_lower.items()
                   if re.search(E_pat, cl) and c in numeric]
        i_match = [c for c, cl in cols_lower.items()
                   if re.search(I_pat, cl) and c in numeric and c not in e_match]
        if e_match and i_match:
            e_col = sorted(e_match,
                           key=lambda c: 0 if "we" in c.lower() or "meas" in c.lower() else 1)[0]
            i_col = i_match[0]
            factor = 1e-3 if unit == "mA" else 1.0
            for pat, f in UNIT_HINTS.items():
                if re.search(pat, cols_lower[i_col]):
                    factor = f; break
            return e_col, i_col, factor
    if len(numeric) >= 2:
        return numeric[0], numeric[1], 1.0
    raise ValueError("Could not find potential and current columns.")


def detect_file_skiprows(raw_bytes, ext):
    if ext in (".xlsx", ".xls"):
        return 0
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        parts = re.split(r"[,;\t ]+", line.strip())
        nc = sum(1 for p in parts if re.match(r"^-?[\d.eE+]+$", p))
        if nc >= 2:
            return max(0, i - 1) if i > 0 and not any(
                re.match(r"^-?[\d.eE+]+$", p)
                for p in re.split(r"[,;\t ]+", lines[i-1].strip())) else i
    return 0


def load_any_file(f):
    name = f.name.lower()
    ext = next((e for e in (".xlsx", ".xls", ".csv", ".txt") if name.endswith(e)), ".csv")
    raw = f.read(); f.seek(0)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(raw))
    skip = detect_file_skiprows(raw, ext)
    text = raw.decode("utf-8", errors="replace")
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5:
                return df.dropna(axis=1, how="all")
        except Exception:
            pass
    raise ValueError(f"Unable to parse {f.name}")


# ═══════════════════════════════════════════════════════════════
# MATH HELPERS
# ═══════════════════════════════════════════════════════════════

def safe_log10(x):
    return np.log10(np.maximum(np.abs(x), 1e-20))

def smooth(y, window=11, poly=3):
    n = len(y)
    w = min(window, n if n % 2 == 1 else n - 1)
    return savgol_filter(y, w, min(poly, w - 1)) if w >= 5 else y.copy()

def _r2(y_true, y_pred):
    ss_r = np.sum((y_true - y_pred)**2)
    ss_t = np.sum((y_true - y_true.mean())**2)
    return float(max(0, 1 - ss_r / ss_t)) if ss_t > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# GLOBAL ELECTROCHEMICAL MODEL
# ═══════════════════════════════════════════════════════════════
#
# Physics-based model covering all regions of a polarization curve:
#
#   i_total = i_anodic − i_cathodic + i_transpassive
#
# Where:
#   Anodic:  i_a_kinetic = icorr · exp(2.303·η/ba)
#            i_anodic = i_a_kinetic · ipass / (i_a_kinetic + ipass)
#            → At low η: pure Tafel (i_a_kinetic << ipass)
#            → At high η: saturates at ipass (passive film limits current)
#
#   Cathodic: i_c_kinetic = icorr · exp(−2.303·η/bc)
#             i_cathodic = i_c_kinetic / (1 + i_c_kinetic / iL)
#             → At low |η|: pure Tafel
#             → At high |η|: saturates at iL (mass-transport limited)
#
#   Transpassive: i_tp = a_tp · exp(b_tp · (E − Eb)) · sigmoid(E − Eb)
#             → Zero below Eb, exponential rise above
#
# This model naturally reduces to simpler cases:
#   • ipass → ∞:  no passivation (active-only system)
#   • iL → ∞:     no diffusion limitation
#   • a_tp → 0:   no transpassive region
# ═══════════════════════════════════════════════════════════════

def global_model(E, Ecorr, icorr, ba, bc, ipass, iL, Eb, a_tp, b_tp):
    """
    Global multi-region polarization curve model.

    Parameters:
        Ecorr  : corrosion potential (V)
        icorr  : corrosion current density (A/cm²)
        ba     : anodic Tafel slope (V/decade)
        bc     : cathodic Tafel slope (V/decade)
        ipass  : passive current density (A/cm²)
        iL     : cathodic limiting current density (A/cm²)
        Eb     : breakdown / transpassive potential (V)
        a_tp   : transpassive pre-exponential (A/cm²)
        b_tp   : transpassive exponential factor (V⁻¹)
    """
    eta = E - Ecorr

    # ── Anodic: activation kinetics → passive-limited ────
    i_a_kin = icorr * np.exp(2.303 * eta / ba)
    # Harmonic transition: naturally saturates at ipass
    i_anodic = i_a_kin * ipass / (i_a_kin + ipass)

    # ── Cathodic: activation kinetics → diffusion-limited ─
    i_c_kin = icorr * np.exp(-2.303 * eta / bc)
    i_cathodic = i_c_kin / (1.0 + i_c_kin / iL)

    # ── Transpassive: exponential rise above Eb ──────────
    # Smooth sigmoid turn-on at Eb (width ~30 mV)
    sigmoid_tp = 1.0 / (1.0 + np.exp(-40.0 * (E - Eb)))
    i_tp = a_tp * np.exp(np.clip(b_tp * (E - Eb), -50, 50)) * sigmoid_tp

    return i_anodic - i_cathodic + i_tp


def global_model_log(E, Ecorr, icorr_log, ba, bc, ipass_log, iL_log,
                     Eb, a_tp_log, b_tp):
    """Same model but with log-scaled current parameters for better optimization."""
    return global_model(E, Ecorr, 10**icorr_log, ba, bc,
                        10**ipass_log, 10**iL_log, Eb, 10**a_tp_log, b_tp)


def butler_volmer(E, Ecorr, icorr, ba, bc):
    """Standard BV for local active-region fitting."""
    eta = E - Ecorr
    return icorr * (np.exp(2.303 * eta / ba) - np.exp(-2.303 * eta / bc))


# ═══════════════════════════════════════════════════════════════
# CURVE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

class CurveType:
    ACTIVE_ONLY = "active_only"
    PASSIVE = "passive"
    DIFFUSION_LIMITED = "diffusion_limited"
    MIXED = "mixed"

    DESCRIPTIONS = {
        ACTIVE_ONLY: ("⚡ Active / Standard Tafel",
                      "Pure activation-controlled. Global model with ipass→∞, iL→∞."),
        PASSIVE: ("🛡️ Passive System",
                  "Anodic passivation detected. Global model includes active→passive "
                  "transition and transpassive rise."),
        DIFFUSION_LIMITED: ("🌊 Diffusion-Limited Cathodic",
                            "Cathodic mass-transport limitation. Global model includes "
                            "diffusion-corrected cathodic branch."),
        MIXED: ("🔀 Mixed: Passive + Diffusion-Limited",
                "Both features detected. Full global model with all terms active."),
    }


# ═══════════════════════════════════════════════════════════════
# REGION DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_regions(E, i):
    """Detect Ecorr, passive region, limiting current, breakdown potential."""
    reg = {}
    n = len(E)
    abs_i = np.abs(i)

    # ── Ecorr ───────────────────────────────────────────
    sc = np.where(np.diff(np.sign(i)))[0]
    if len(sc) > 0:
        k = sc[0]; d = i[k+1] - i[k]
        reg["Ecorr"] = float(E[k] - i[k]*(E[k+1]-E[k])/d) if abs(d) > 0 else float(E[k])
        reg["ecorr_idx"] = k
    else:
        k = int(np.argmin(np.abs(i)))
        reg["Ecorr"] = float(E[k]); reg["ecorr_idx"] = k

    Ec = reg["Ecorr"]

    # ── Cathodic limiting current ───────────────────────
    ci = np.where(E < Ec)[0]
    if len(ci) >= 8:
        log_c = smooth(safe_log10(abs_i[ci]), min(11, (len(ci)//2)*2-1 or 5))
        dlog = np.abs(np.gradient(log_c, E[ci]))
        threshold = np.percentile(dlog, 20)
        flat = dlog < max(threshold, 0.5)
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            best_run = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best_run[1]]
            e_range = abs(E[ci[idxs[-1]]] - E[ci[idxs[0]]])
            far = abs(E[ci[idxs[0]]] - Ec) > 0.08
            if len(idxs) >= 4 and e_range > 0.03 and far:
                reg.update(
                    limit_idx=ci[idxs],
                    iL=float(np.median(abs_i[ci[idxs]])),
                    E_lim_start=float(E[ci[idxs[0]]]),
                    E_lim_end=float(E[ci[idxs[-1]]]),
                )

    # ── Anodic: passive region + breakdown ──────────────
    ai = np.where(E > Ec)[0]
    if len(ai) >= 8:
        log_a = smooth(safe_log10(abs_i[ai]), min(11, (len(ai)//2)*2-1 or 5))
        dlog = np.abs(np.gradient(log_a, E[ai]))
        threshold = np.percentile(dlog, 30)
        flat = dlog < max(threshold, 1.0)
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            best_run = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best_run[1]]
            e_range = abs(E[ai[idxs[-1]]] - E[ai[idxs[0]]])
            if len(idxs) >= 4 and e_range > 0.04:
                ps, pe = ai[idxs[0]], ai[idxs[-1]]
                i_in_region = np.median(abs_i[ps:pe+1])
                pre_passive = np.where((E > Ec) & (E < E[ps]))[0]
                is_passive = True
                if len(pre_passive) > 2:
                    i_peak = np.max(abs_i[pre_passive])
                    is_passive = i_in_region < i_peak * 0.8
                else:
                    is_passive = e_range > 0.10

                if is_passive:
                    reg.update(
                        passive_s=int(ps), passive_e=int(pe),
                        E_ps=float(E[ps]), E_pe=float(E[pe]),
                        ipass=float(np.median(abs_i[ps:pe+1])),
                        Epp=float(E[ps]),
                    )
                    if pe + 3 < n:
                        d_after = np.gradient(safe_log10(abs_i[pe:]), E[pe:])
                        thr = np.mean(dlog) + 2.0 * np.std(dlog)
                        jump = np.where(np.abs(d_after) > max(thr, 3.0))[0]
                        if len(jump):
                            eb_idx = pe + jump[0]
                            reg["Eb_idx"] = int(eb_idx)
                            reg["Eb"] = float(E[eb_idx])
                        reg["tp_idx"] = reg.get("Eb_idx", pe)

    # ── Classify curve type ─────────────────────────────
    has_passive = reg.get("passive_s") is not None
    has_limiting = reg.get("iL") is not None
    if has_passive and has_limiting:
        reg["curve_type"] = CurveType.MIXED
    elif has_passive:
        reg["curve_type"] = CurveType.PASSIVE
    elif has_limiting:
        reg["curve_type"] = CurveType.DIFFUSION_LIMITED
    else:
        reg["curve_type"] = CurveType.ACTIVE_ONLY

    return reg


# ═══════════════════════════════════════════════════════════════
# GLOBAL FITTING ENGINE
# ═══════════════════════════════════════════════════════════════

class GlobalFitter:
    """
    Two-stage fitting:
      1. Local analysis → extract initial parameter guesses
      2. Global model fit across ENTIRE curve with multi-strategy optimization
    """

    def __init__(self, E, i, reg):
        self.E = E
        self.i = i
        self.abs_i = np.abs(i)
        self.log_abs_i = safe_log10(i)
        self.reg = reg
        self.Ec = reg["Ecorr"]
        self.ec_idx = reg["ecorr_idx"]
        self.ic0 = max(float(np.abs(i[self.ec_idx])), 1e-12)
        self.curve_type = reg["curve_type"]
        self.R = {}
        self.log = []

    # ── Stage 1: Local Tafel analysis for initial guesses ─

    def _find_tafel_window(self, side="anodic"):
        E, log_i, Ec = self.E, self.log_abs_i, self.Ec
        if side == "anodic":
            E_limit = self.reg.get("E_ps", Ec + 0.50)
            E_limit = min(E_limit - 0.005, Ec + 0.30)
            best = None
            for lo in np.arange(0.005, 0.06, 0.005):
                for hi in np.arange(lo + 0.015, min(lo + 0.20, E_limit - Ec), 0.005):
                    m = (E > Ec + lo) & (E < Ec + hi)
                    if m.sum() < 4: continue
                    s, b, r, *_ = linregress(E[m], log_i[m])
                    if s <= 0: continue
                    ba_mV = (1/s) * 1000
                    if 20 < ba_mV < 500 and r**2 > 0.90:
                        if best is None or r**2 > best["r2"]:
                            best = dict(slope=s, intercept=b, r2=r**2, ba=1/s,
                                        mask=m, lo=lo, hi=hi, n=int(m.sum()))
            return best
        else:
            E_lim_end = self.reg.get("E_lim_end")
            best = None
            for lo in np.arange(0.005, 0.10, 0.005):
                for hi in np.arange(lo + 0.015, lo + 0.30, 0.005):
                    m = (E < Ec - lo) & (E > Ec - hi)
                    if E_lim_end is not None:
                        m = m & (E > E_lim_end + 0.005)
                    if m.sum() < 4: continue
                    s, b, r, *_ = linregress(E[m], log_i[m])
                    if s >= 0: continue
                    bc_mV = (-1/s) * 1000
                    if 20 < bc_mV < 500 and r**2 > 0.90:
                        if best is None or r**2 > best["r2"]:
                            best = dict(slope=s, intercept=b, r2=r**2, bc=-1/s,
                                        mask=m, lo=lo, hi=hi, n=int(m.sum()))
            return best

    def _local_analysis(self):
        """Extract initial guesses from Tafel slopes and region detection."""
        an = self._find_tafel_window("anodic")
        ca = self._find_tafel_window("cathodic")

        if an:
            self.R["an"] = an
            self.log.append(
                f"✅ Anodic Tafel: βa = {an['ba']*1000:.1f} mV/dec, "
                f"R² = {an['r2']:.4f}")
        else:
            self.log.append("ℹ️ No clean anodic Tafel region — using default βa guess.")

        if ca:
            self.R["ca"] = ca
            self.log.append(
                f"✅ Cathodic Tafel: βc = {ca['bc']*1000:.1f} mV/dec, "
                f"R² = {ca['r2']:.4f}")
        else:
            self.log.append("ℹ️ No clean cathodic Tafel region — using default βc guess.")

        # Intersection icorr
        if an and ca:
            ds = an["slope"] - ca["slope"]
            if abs(ds) > 1e-10:
                E_i = (ca["intercept"] - an["intercept"]) / ds
                logi = an["slope"] * E_i + an["intercept"]
                self.R["Ecorr_tafel"] = float(E_i)
                self.R["icorr_tafel"] = float(10**logi)
                self.log.append(
                    f"✅ Tafel intersection: Ecorr = {E_i:.4f} V, "
                    f"icorr = {10**logi:.3e} A/cm²")

        # Local BV fit (narrow window) for refined Ecorr/icorr
        ba0 = an["ba"] if an else 0.06
        bc0 = ca["bc"] if ca else 0.12
        Eps = self.reg.get("E_ps")
        hw_an = min(0.15, (Eps - self.Ec) * 0.7) if Eps else 0.15
        hw_cat = 0.15
        m = (self.E > self.Ec - hw_cat) & (self.E < self.Ec + hw_an)
        if m.sum() >= 6:
            try:
                p0 = [self.Ec, self.ic0, ba0, bc0]
                bnd = ([self.E.min(), 1e-14, 0.005, 0.005],
                       [self.E.max(), 1e-1, 0.8, 0.8])
                popt, _ = curve_fit(butler_volmer, self.E[m], self.i[m], p0=p0,
                                    bounds=bnd, maxfev=50000, method="trf",
                                    ftol=1e-14, xtol=1e-14)
                self.R["local_bv"] = dict(Ecorr=float(popt[0]), icorr=float(popt[1]),
                                          ba=float(popt[2]), bc=float(popt[3]))
                self.log.append(
                    f"✅ Local BV: Ecorr = {popt[0]:.4f}, icorr = {popt[1]:.3e}, "
                    f"βa = {popt[2]*1000:.0f}, βc = {popt[3]*1000:.0f} mV/dec")
            except:
                pass

    def _build_initial_guess(self):
        """Build p0 and bounds for the global model from local analysis."""
        reg = self.reg
        ct = self.curve_type

        # Best estimates from local fits
        lbv = self.R.get("local_bv", {})
        an = self.R.get("an", {})
        ca = self.R.get("ca", {})

        Ecorr0 = lbv.get("Ecorr") or self.R.get("Ecorr_tafel") or self.Ec
        icorr0 = lbv.get("icorr") or self.R.get("icorr_tafel") or self.ic0
        ba0 = lbv.get("ba") or an.get("ba") or 0.060
        bc0 = lbv.get("bc") or ca.get("bc") or 0.120

        # Passive current density
        ipass0 = reg.get("ipass", 1e0)  # very large = no passivation
        if ct == CurveType.ACTIVE_ONLY:
            ipass0 = 1e2  # effectively infinite

        # Limiting current
        iL0 = reg.get("iL", 1e2)  # very large = no diffusion limitation
        if ct == CurveType.ACTIVE_ONLY:
            iL0 = 1e2

        # Breakdown / transpassive
        Eb0 = reg.get("Eb") or reg.get("E_pe") or self.E[-1]
        # Transpassive parameters: estimate from slope after Eb
        a_tp0 = ipass0 if reg.get("Eb") else 1e-10
        b_tp0 = 5.0  # moderate exponential rise

        # If we have data past Eb, estimate transpassive parameters
        if reg.get("Eb"):
            tp_mask = self.E > reg["Eb"]
            if tp_mask.sum() > 3:
                log_tp = safe_log10(self.abs_i[tp_mask])
                E_tp = self.E[tp_mask]
                try:
                    s, intercept, *_ = linregress(E_tp, log_tp)
                    b_tp0 = max(s * 2.303, 1.0)  # convert to natural
                    a_tp0 = 10**intercept / np.exp(b_tp0 * (E_tp[0] - reg["Eb"]))
                    a_tp0 = max(a_tp0, 1e-10)
                except:
                    pass

        self._p0 = {
            "Ecorr": Ecorr0, "icorr": icorr0, "ba": ba0, "bc": bc0,
            "ipass": ipass0, "iL": iL0, "Eb": Eb0, "a_tp": a_tp0, "b_tp": b_tp0,
        }

        self.log.append(
            f"   Initial guess: Ecorr={Ecorr0:.4f}, icorr={icorr0:.2e}, "
            f"βa={ba0*1000:.0f}, βc={bc0*1000:.0f} mV/dec")
        if ct in (CurveType.PASSIVE, CurveType.MIXED):
            self.log.append(f"   ipass={ipass0:.2e}, Eb={Eb0:.3f}")
        if ct in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
            self.log.append(f"   iL={iL0:.2e}")

    # ── Stage 2: Global model fitting ─────────────────────

    def _global_fit_de(self):
        """Differential Evolution global optimization of the full model."""
        E, i = self.E, self.i
        p = self._p0
        ct = self.curve_type
        log_abs_data = safe_log10(i)

        # Build DE bounds based on curve type
        Ec = p["Ecorr"]
        ic = p["icorr"]

        bounds = [
            (Ec - 0.10, Ec + 0.10),                              # Ecorr
            (np.log10(max(ic*1e-3, 1e-14)), np.log10(ic*1e3)),   # log10(icorr)
            (0.010, 0.500),                                       # ba
            (0.010, 0.500),                                       # bc
        ]

        # ipass bounds
        if ct in (CurveType.PASSIVE, CurveType.MIXED):
            ip = p["ipass"]
            bounds.append((np.log10(max(ip*0.01, 1e-10)), np.log10(ip*100)))
        else:
            bounds.append((0.0, 5.0))  # effectively infinite

        # iL bounds
        if ct in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
            il = p["iL"]
            bounds.append((np.log10(max(il*0.01, 1e-8)), np.log10(il*100)))
        else:
            bounds.append((0.0, 5.0))  # effectively infinite

        # Eb, a_tp, b_tp bounds
        if ct in (CurveType.PASSIVE, CurveType.MIXED):
            Eb = p["Eb"]
            bounds.append((Eb - 0.15, min(Eb + 0.15, E[-1])))        # Eb
            bounds.append((np.log10(max(p["a_tp"]*0.001, 1e-12)),
                           np.log10(max(p["a_tp"]*1000, 1e-2))))      # log10(a_tp)
            bounds.append((0.5, 30.0))                                 # b_tp
        else:
            bounds.append((E[-1] - 0.1, E[-1] + 0.5))   # Eb far away
            bounds.append((-12.0, -8.0))                  # tiny a_tp
            bounds.append((0.5, 5.0))                     # b_tp irrelevant

        def objective(x):
            try:
                pred = global_model(E, x[0], 10**x[1], x[2], x[3],
                                    10**x[4], 10**x[5], x[6], 10**x[7], x[8])
                log_pred = safe_log10(pred)
                # Primary: log-space residual (weighs all decades equally)
                res_log = np.sum((log_abs_data - log_pred)**2)
                # Secondary: relative residual in linear space (helps near Ecorr)
                res_rel = np.sum(((i - pred) / (np.abs(i) + 1e-12))**2) * 0.05
                return res_log + res_rel
            except:
                return 1e30

        self.log.append("🔧 Running Differential Evolution global optimization…")

        try:
            result = differential_evolution(
                objective, bounds, seed=42,
                maxiter=4000, tol=1e-13,
                popsize=35, workers=1,
                mutation=(0.5, 1.5), recombination=0.9,
                polish=False  # we'll do our own polishing
            )
            x = result.x
            params = dict(
                Ecorr=float(x[0]), icorr=float(10**x[1]),
                ba=float(x[2]), bc=float(x[3]),
                ipass=float(10**x[4]), iL=float(10**x[5]),
                Eb=float(x[6]), a_tp=float(10**x[7]), b_tp=float(x[8]),
            )
            pred = global_model(E, params["Ecorr"], params["icorr"],
                                params["ba"], params["bc"],
                                params["ipass"], params["iL"],
                                params["Eb"], params["a_tp"], params["b_tp"])
            r2_log = _r2(log_abs_data, safe_log10(pred))
            r2_lin = _r2(i, pred)
            params.update(r2_log=r2_log, r2_lin=r2_lin,
                          method="Global DE", success=True)

            self.log.append(
                f"✅ DE result: R²(log) = {r2_log:.4f}, R²(lin) = {r2_lin:.4f}")

            self.R["global_de"] = params
            self._de_x = x  # save for polishing
            return params
        except Exception as ex:
            self.log.append(f"⚠️ DE failed: {ex}")
            return None

    def _polish_nelder_mead(self, x0_dict):
        """Nelder-Mead polish of the DE result."""
        E, i = self.E, self.i
        log_abs_data = safe_log10(i)

        x0 = [
            x0_dict["Ecorr"], np.log10(max(x0_dict["icorr"], 1e-14)),
            x0_dict["ba"], x0_dict["bc"],
            np.log10(max(x0_dict["ipass"], 1e-14)),
            np.log10(max(x0_dict["iL"], 1e-14)),
            x0_dict["Eb"],
            np.log10(max(x0_dict["a_tp"], 1e-14)),
            x0_dict["b_tp"],
        ]

        def objective(x):
            try:
                pred = global_model(E, x[0], 10**x[1], x[2], x[3],
                                    10**x[4], 10**x[5], x[6], 10**x[7], x[8])
                log_pred = safe_log10(pred)
                return float(np.sum((log_abs_data - log_pred)**2))
            except:
                return 1e30

        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 50000, "xatol": 1e-13, "fatol": 1e-15,
                                   "adaptive": True})
            x = res.x
            params = dict(
                Ecorr=float(x[0]), icorr=float(10**x[1]),
                ba=float(x[2]), bc=float(x[3]),
                ipass=float(10**x[4]), iL=float(10**x[5]),
                Eb=float(x[6]), a_tp=float(10**x[7]), b_tp=float(x[8]),
            )
            pred = global_model(E, *[params[k] for k in
                                     ["Ecorr","icorr","ba","bc","ipass","iL","Eb","a_tp","b_tp"]])
            r2_log = _r2(log_abs_data, safe_log10(pred))
            r2_lin = _r2(i, pred)
            params.update(r2_log=r2_log, r2_lin=r2_lin,
                          method="Global DE + NM polished", success=True)
            self.log.append(f"✅ NM polish: R²(log) = {r2_log:.4f}")
            return params
        except Exception as ex:
            self.log.append(f"⚠️ NM polish failed: {ex}")
            return x0_dict

    def _try_curve_fit_refinement(self, x0_dict):
        """Try scipy curve_fit for final refinement (gradient-based)."""
        E, i = self.E, self.i
        log_abs_data = safe_log10(i)

        p0 = [x0_dict["Ecorr"], x0_dict["icorr"], x0_dict["ba"], x0_dict["bc"],
              x0_dict["ipass"], x0_dict["iL"], x0_dict["Eb"],
              x0_dict["a_tp"], x0_dict["b_tp"]]

        # Wide but sensible bounds
        bnd_lo = [E.min(), 1e-14, 0.005, 0.005, 1e-10, 1e-8,
                  E.min(), 1e-14, 0.1]
        bnd_hi = [E.max(), 1e0, 0.8, 0.8, 1e2, 1e2,
                  E.max() + 0.5, 1e2, 50.0]

        try:
            popt, _ = curve_fit(global_model, E, i, p0=p0,
                                bounds=(bnd_lo, bnd_hi),
                                maxfev=100000, method="trf",
                                ftol=1e-15, xtol=1e-15)
            params = dict(zip(
                ["Ecorr","icorr","ba","bc","ipass","iL","Eb","a_tp","b_tp"],
                [float(v) for v in popt]))
            pred = global_model(E, *popt)
            r2_log = _r2(log_abs_data, safe_log10(pred))
            r2_lin = _r2(i, pred)
            params.update(r2_log=r2_log, r2_lin=r2_lin,
                          method="Global TRF refined", success=True)
            self.log.append(f"✅ TRF refinement: R²(log) = {r2_log:.4f}")
            return params
        except Exception as ex:
            self.log.append(f"ℹ️ TRF refinement did not improve: {ex}")
            return None

    # ── Polarization resistance ───────────────────────────

    def _fit_rp(self):
        E, i, Ec = self.E, self.i, self.Ec
        for dE in [0.010, 0.015, 0.025, 0.05]:
            m = np.abs(E - Ec) < dE
            if m.sum() >= 4:
                s, _, r, *_ = linregress(E[m], i[m])
                if abs(s) > 1e-20:
                    self.R["rp"] = dict(Rp=float(1/s), r2=float(r**2), dE=dE)
                    self.log.append(f"✅ Rp = {1/s:.3e} Ω·cm² (±{dE*1000:.0f} mV)")
                    return

    # ── Master run ────────────────────────────────────────

    def run(self):
        ct = self.curve_type
        self.log.append(f"🔍 Curve type: **{CurveType.DESCRIPTIONS[ct][0]}**")
        self.log.append(f"   {CurveType.DESCRIPTIONS[ct][1]}")
        self.log.append("─" * 50)

        # Stage 1: Local analysis for initial guesses
        self.log.append("**Stage 1: Local Tafel analysis (initial guesses)**")
        self._local_analysis()
        self._build_initial_guess()

        self.log.append("─" * 50)
        self.log.append("**Stage 2: Global model fitting (full curve)**")

        # Stage 2: Global DE
        de_result = self._global_fit_de()

        # Stage 3: Polish
        best = de_result
        if de_result:
            polished = self._polish_nelder_mead(de_result)
            if polished.get("r2_log", 0) >= de_result.get("r2_log", 0):
                best = polished

            # Try TRF refinement
            trf = self._try_curve_fit_refinement(best)
            if trf and trf.get("r2_log", 0) >= best.get("r2_log", 0):
                best = trf

        if best:
            # Compute Stern-Geary
            ba, bc = best["ba"], best["bc"]
            if ba > 0 and bc > 0:
                best["B"] = (ba * bc) / (2.303 * (ba + bc))
            self.R["best"] = best
        else:
            # Fallback to Tafel extrapolation
            an = self.R.get("an", {})
            ca = self.R.get("ca", {})
            self.R["best"] = dict(
                Ecorr=self.R.get("Ecorr_tafel", self.Ec),
                icorr=self.R.get("icorr_tafel", self.ic0),
                ba=an.get("ba"), bc=ca.get("bc"),
                method="Tafel Extrapolation only", success=False)

        # Rp
        self._fit_rp()

        # Passive parameters from region detection (for display)
        reg = self.reg
        if reg.get("passive_s") is not None:
            ps, pe = reg["passive_s"], reg["passive_e"]
            self.R["passive"] = dict(
                ipass=float(np.median(self.abs_i[ps:pe+1])),
                E_start=float(self.E[ps]), E_end=float(self.E[pe]),
                range_V=float(self.E[pe] - self.E[ps]))

        # Quality summary
        self.log.append("─" * 50)
        r2l = best.get("r2_log") if best else None
        if r2l is not None:
            if r2l >= 0.990:
                self.log.append(f"🎯 **Excellent global fit** — R²(log) = {r2l:.4f}")
            elif r2l >= 0.950:
                self.log.append(f"✅ **Good global fit** — R²(log) = {r2l:.4f}")
            elif r2l >= 0.85:
                self.log.append(f"⚠️ **Acceptable fit** — R²(log) = {r2l:.4f}")
            else:
                self.log.append(f"❌ **Poor fit** — R²(log) = {r2l:.4f}")

        return self.R


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def _eval_global_model(E_range, best):
    """Evaluate global model across a potential range."""
    keys = ["Ecorr","icorr","ba","bc","ipass","iL","Eb","a_tp","b_tp"]
    if all(k in best for k in keys):
        return global_model(E_range, *[best[k] for k in keys])
    return None


def plot_polarization(E, i, R, reg, curve_type):
    an = R.get("an", {})
    ca = R.get("ca", {})
    best = R.get("best", {})
    pas = R.get("passive", {})
    log_i = safe_log10(i)

    fig = go.Figure()

    # ── Region fills ────────────────────────────────────
    if pas.get("E_start") is not None:
        fig.add_vrect(x0=pas["E_start"], x1=pas["E_end"],
                      fillcolor=C["passive"], layer="below", line_width=0,
                      annotation=dict(text="Passive",
                                      font=dict(color="#a6e3a1", size=11), yanchor="top"))
    if reg.get("E_lim_start") is not None:
        fig.add_vrect(x0=reg["E_lim_start"], x1=reg["E_lim_end"],
                      fillcolor=C["limiting"], layer="below", line_width=0,
                      annotation=dict(text="Limiting",
                                      font=dict(color="#89dceb", size=11), yanchor="top"))
    if reg.get("tp_idx") is not None:
        tp_start = E[reg["tp_idx"]] if reg["tp_idx"] < len(E) else E[-1]
        fig.add_vrect(x0=tp_start, x1=float(E[-1]),
                      fillcolor=C["transpassive"], layer="below", line_width=0,
                      annotation=dict(text="Transpassive",
                                      font=dict(color="#fab387", size=11), yanchor="top"))

    # ── Key vertical lines ──────────────────────────────
    Ec = reg["Ecorr"]
    fig.add_vline(x=Ec, line=dict(color="#f38ba8", width=1.5, dash="dot"),
                  annotation=dict(text="Ecorr", font=dict(color="#f38ba8", size=10),
                                  yanchor="bottom"))
    if reg.get("Eb"):
        fig.add_vline(x=reg["Eb"], line=dict(color="#f38ba8", width=1.5, dash="dash"),
                      annotation=dict(text="Eb", font=dict(color="#f38ba8", size=10),
                                      yanchor="top"))
    if reg.get("Epp"):
        fig.add_vline(x=reg["Epp"], line=dict(color="#a6e3a1", width=1, dash="dot"),
                      annotation=dict(text="Epp", font=dict(color="#a6e3a1", size=10),
                                      yanchor="bottom"))

    # ── Measured data ───────────────────────────────────
    fig.add_trace(go.Scatter(x=E, y=log_i, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5),
                             hovertemplate="E=%{x:.4f} V<br>log|i|=%{y:.3f}<extra></extra>"))

    # ── Tafel lines (in their valid window only) ────────
    if an.get("slope"):
        an_lo = Ec + an.get("lo", 0) - 0.02
        an_hi = Ec + an.get("hi", 0.15) + 0.03
        E_an = np.linspace(an_lo, min(an_hi, reg.get("E_ps", an_hi)), 80)
        fig.add_trace(go.Scatter(
            x=E_an, y=an["slope"]*E_an + an["intercept"], mode="lines",
            name=f"Anodic Tafel βa={an['ba']*1000:.0f} mV/dec R²={an['r2']:.3f}",
            line=dict(color=C["anodic"], width=1.5, dash="dash")))

    if ca.get("slope"):
        ca_hi = Ec - ca.get("lo", 0) + 0.02
        ca_lo = Ec - ca.get("hi", 0.15) - 0.03
        E_ca = np.linspace(max(ca_lo, reg.get("E_lim_end", ca_lo)), ca_hi, 80)
        fig.add_trace(go.Scatter(
            x=E_ca, y=ca["slope"]*E_ca + ca["intercept"], mode="lines",
            name=f"Cathodic Tafel βc={ca['bc']*1000:.0f} mV/dec R²={ca['r2']:.3f}",
            line=dict(color=C["cathodic"], width=1.5, dash="dash")))

    # ── GLOBAL MODEL FIT (overlays entire curve) ────────
    E_model = np.linspace(E.min(), E.max(), 600)
    i_model = _eval_global_model(E_model, best)
    if i_model is not None:
        r2l = best.get("r2_log")
        r2_str = f"R²(log)={r2l:.3f}" if r2l is not None else ""
        method = best.get("method", "Global Fit")
        fig.add_trace(go.Scatter(
            x=E_model, y=safe_log10(i_model), mode="lines",
            name=f"Global Fit  {r2_str}",
            line=dict(color=C["global_fit"], width=3)))

    # ── icorr marker ────────────────────────────────────
    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic and ec:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(max(ic, 1e-20))], mode="markers",
            name=f"icorr = {ic:.3e} A/cm²",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))))

    # ── iL line ─────────────────────────────────────────
    if reg.get("iL") and curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
        fig.add_hline(y=np.log10(reg["iL"]),
                      line=dict(color="#89dceb", width=1, dash="dot"),
                      annotation=dict(text=f"iL={reg['iL']:.2e}",
                                      font=dict(color="#89dceb", size=10)))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Potentiodynamic Polarization Curve — Global Fit",
                   font=dict(size=17, color=C["text"]), x=0.0),
        xaxis=dict(title="Potential (V vs Ref)", gridcolor=C["grid"],
                   color=C["text"], zeroline=False, showline=True, linecolor=C["grid"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)", gridcolor=C["grid"],
                   color=C["text"], zeroline=False, showline=True, linecolor=C["grid"]),
        legend=dict(bgcolor="rgba(24,24,37,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11), x=0.01, y=0.01),
        height=520, margin=dict(l=70, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def plot_active_zoom(E, i, R, reg, curve_type):
    an = R.get("an", {})
    ca = R.get("ca", {})
    best = R.get("best", {})
    Ec = reg["Ecorr"]
    log_i = safe_log10(i)

    hw = 0.20 if curve_type == CurveType.PASSIVE else 0.30
    E_ps = reg.get("E_ps")
    zm_hi = min(Ec + hw, E_ps + 0.03) if E_ps else Ec + hw
    zm = (E >= Ec - hw - 0.05) & (E <= zm_hi)
    E_z, y_z = E[zm], log_i[zm]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E_z, y=y_z, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5)))

    E_fit = np.linspace(E_z.min(), E_z.max(), 200)
    if an.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=an["slope"]*E_fit+an["intercept"],
                                 mode="lines",
                                 name=f"Anodic βa={an['ba']*1000:.0f}",
                                 line=dict(color=C["anodic"], width=2, dash="dash")))
    if ca.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=ca["slope"]*E_fit+ca["intercept"],
                                 mode="lines",
                                 name=f"Cathodic βc={ca['bc']*1000:.0f}",
                                 line=dict(color=C["cathodic"], width=2, dash="dash")))

    # Global model in zoom
    i_model = _eval_global_model(E_fit, best)
    if i_model is not None:
        fig.add_trace(go.Scatter(x=E_fit, y=safe_log10(i_model), mode="lines",
                                 name="Global Fit",
                                 line=dict(color=C["global_fit"], width=3)))

    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(max(ic, 1e-20))], mode="markers",
            name=f"icorr={ic:.3e}",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))))
    fig.add_vline(x=Ec, line=dict(color="#f38ba8", width=1, dash="dot"))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Active / Tafel Region (zoom)", font=dict(size=15, color=C["text"])),
        xaxis=dict(title="Potential (V)", gridcolor=C["grid"], color=C["text"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)", gridcolor=C["grid"], color=C["text"]),
        legend=dict(bgcolor="rgba(24,24,37,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11)),
        height=420, margin=dict(l=70, r=20, t=50, b=60),
    )
    return fig


def plot_residuals(E, i, R, reg, curve_type):
    best = R.get("best", {})
    log_abs_data = safe_log10(i)

    i_model = _eval_global_model(E, best)
    if i_model is None:
        return None

    log_model = safe_log10(i_model)
    residuals = log_abs_data - log_model

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.06,
                        subplot_titles=("Global Fit Overlay",
                                        "Residuals in log-space"))

    fig.add_trace(go.Scatter(x=E, y=log_abs_data, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2)), row=1, col=1)

    E_dense = np.linspace(E.min(), E.max(), 600)
    i_dense = _eval_global_model(E_dense, best)
    if i_dense is not None:
        fig.add_trace(go.Scatter(x=E_dense, y=safe_log10(i_dense), mode="lines",
                                 name="Global Fit",
                                 line=dict(color=C["global_fit"], width=2.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=E, y=residuals, mode="lines",
                             name="Residual",
                             line=dict(color="#fab387", width=1.2),
                             fill="tozeroy",
                             fillcolor="rgba(250,179,135,0.08)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#585b70", width=1, dash="dot"), row=2)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_res = float(np.max(np.abs(residuals)))
    fig.add_annotation(text=f"RMSE(log) = {rmse:.4f}  |  max = {max_res:.3f} dec",
                       xref="paper", yref="paper", x=0.98, y=0.38,
                       showarrow=False, font=dict(color="#a6adc8", size=11),
                       bgcolor="rgba(24,24,37,0.8)", bordercolor=C["grid"])

    # Mark regions in residual plot
    if reg.get("E_lim_start"):
        for row in (1, 2):
            fig.add_vrect(x0=reg["E_lim_start"], x1=reg["E_lim_end"],
                          fillcolor="rgba(137,220,235,0.06)", layer="below",
                          line_width=0, row=row)
    pas = R.get("passive", {})
    if pas.get("E_start"):
        for row in (1, 2):
            fig.add_vrect(x0=pas["E_start"], x1=pas["E_end"],
                          fillcolor="rgba(166,227,161,0.06)", layer="below",
                          line_width=0, row=row)

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        height=440, margin=dict(l=70, r=20, t=40, b=60), showlegend=True,
        legend=dict(bgcolor="rgba(24,24,37,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11)),
    )
    fig.update_yaxes(gridcolor=C["grid"], color=C["text"])
    fig.update_xaxes(gridcolor=C["grid"], color=C["text"], title_text="Potential (V)", row=2)
    return fig


def plot_component_breakdown(E, R, reg, curve_type):
    """Show individual current components of the global model."""
    best = R.get("best", {})
    keys = ["Ecorr","icorr","ba","bc","ipass","iL","Eb","a_tp","b_tp"]
    if not all(k in best for k in keys):
        return None

    E_m = np.linspace(E.min(), E.max(), 600)
    eta = E_m - best["Ecorr"]

    # Anodic (activation → passive limited)
    i_a_kin = best["icorr"] * np.exp(2.303 * eta / best["ba"])
    i_anodic = i_a_kin * best["ipass"] / (i_a_kin + best["ipass"])

    # Cathodic (activation → diffusion limited)
    i_c_kin = best["icorr"] * np.exp(-2.303 * eta / best["bc"])
    i_cathodic = i_c_kin / (1.0 + i_c_kin / best["iL"])

    # Transpassive
    sigmoid_tp = 1.0 / (1.0 + np.exp(-40.0 * (E_m - best["Eb"])))
    i_tp = best["a_tp"] * np.exp(np.clip(best["b_tp"] * (E_m - best["Eb"]), -50, 50)) * sigmoid_tp

    # Total
    i_total = i_anodic - i_cathodic + i_tp

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=E_m, y=safe_log10(i_anodic), mode="lines",
                             name="Anodic (active→passive)",
                             line=dict(color="#f9e2af", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=E_m, y=safe_log10(i_cathodic), mode="lines",
                             name="Cathodic (kinetic→diffusion)",
                             line=dict(color="#cba6f7", width=1.5, dash="dot")))
    if curve_type in (CurveType.PASSIVE, CurveType.MIXED):
        valid_tp = i_tp > 1e-15
        if valid_tp.any():
            fig.add_trace(go.Scatter(x=E_m[valid_tp], y=safe_log10(i_tp[valid_tp]),
                                     mode="lines", name="Transpassive",
                                     line=dict(color="#fab387", width=1.5, dash="dot")))

    fig.add_trace(go.Scatter(x=E_m, y=safe_log10(i_total), mode="lines",
                             name="Total (global model)",
                             line=dict(color=C["global_fit"], width=3)))

    # ipass horizontal line
    if curve_type in (CurveType.PASSIVE, CurveType.MIXED):
        fig.add_hline(y=np.log10(best["ipass"]),
                      line=dict(color="#a6e3a1", width=1, dash="dot"),
                      annotation=dict(text=f"ipass={best['ipass']:.2e}",
                                      font=dict(color="#a6e3a1", size=10)))
    if curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
        fig.add_hline(y=np.log10(best["iL"]),
                      line=dict(color="#89dceb", width=1, dash="dot"),
                      annotation=dict(text=f"iL={best['iL']:.2e}",
                                      font=dict(color="#89dceb", size=10)))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Model Component Breakdown",
                   font=dict(size=15, color=C["text"])),
        xaxis=dict(title="Potential (V)", gridcolor=C["grid"], color=C["text"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)", gridcolor=C["grid"], color=C["text"]),
        legend=dict(bgcolor="rgba(24,24,37,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11)),
        height=420, margin=dict(l=70, r=20, t=50, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# PARAMETER CARDS
# ═══════════════════════════════════════════════════════════════

def pcard(label, val, unit="", color="#cdd6f4"):
    if val is None:
        disp = "—"
    elif isinstance(val, str):
        disp = val
    elif abs(val) < 0.001 or abs(val) > 9999:
        disp = f"{val:.4e}"
    else:
        disp = f"{val:.4f}"
    st.markdown(f"""
    <div class="pcard">
        <div class="plabel">{label}</div>
        <div class="pval" style="color:{color}">{disp}</div>
        <div class="punit">{unit}</div>
    </div>""", unsafe_allow_html=True)


def show_curve_type_banner(curve_type):
    title, desc = CurveType.DESCRIPTIONS[curve_type]
    clr = {"active_only": "#f9e2af", "passive": "#a6e3a1",
           "diffusion_limited": "#89dceb", "mixed": "#cba6f7"}.get(curve_type, "#cdd6f4")
    st.markdown(f"""
    <div class="type-box">
        <div class="type-title" style="color:{clr}">{title}</div>
        <div class="type-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def show_parameters(R, reg, ew, rho, curve_type):
    best = R.get("best", {})
    an = R.get("an", {})
    ca = R.get("ca", {})
    pas = R.get("passive", {})
    rp_r = R.get("rp", {})

    Ec = best.get("Ecorr", reg["Ecorr"])
    ic = best.get("icorr")
    ba = best.get("ba")
    bc = best.get("bc")
    B = best.get("B")
    Rp = rp_r.get("Rp")
    ic_sg = (B / Rp) if (B and Rp and Rp > 0) else None
    CR = ic * 3.27 * ew / rho if ic else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    # Badges
    badges = []
    if reg.get("E_lim_start") is not None:
        badges.append('<span class="badge bb">Limiting current</span>')
    if pas.get("E_start") is not None:
        badges.append('<span class="badge bg">Passive region</span>')
    if reg.get("Eb") is not None:
        badges.append('<span class="badge br">Breakdown</span>')
    if reg.get("tp_idx") is not None:
        badges.append('<span class="badge by">Transpassive</span>')
    if reg.get("Epp") is not None:
        badges.append('<span class="badge bg">Epp / Flade</span>')
    if badges:
        st.markdown("**Detected regions:** " + "".join(badges), unsafe_allow_html=True)
    show_curve_type_banner(curve_type)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="sechead">⚡ Corrosion</div>', unsafe_allow_html=True)
        pcard("Ecorr", Ec, "V vs Ref", "#f38ba8")
        pcard("icorr (Global fit)", ic, "A cm⁻²", "#fab387")
        if ic_sg: pcard("icorr (Stern-Geary)", ic_sg, "A cm⁻²", "#f9e2af")
        if CR:    pcard("Corrosion rate", CR, "mm year⁻¹", "#eba0ac")
        if CR_sg: pcard("CR (Stern-Geary)", CR_sg, "mm year⁻¹", "#f5c2e7")

    with col2:
        st.markdown('<div class="sechead">📐 Kinetics</div>', unsafe_allow_html=True)
        pcard("βa  anodic Tafel", ba*1000 if ba else None, "mV dec⁻¹", "#a6e3a1")
        pcard("βc  cathodic Tafel", bc*1000 if bc else None, "mV dec⁻¹", "#94e2d5")
        if B:  pcard("B  Stern-Geary", B*1000, "mV", "#89dceb")
        if Rp: pcard("Rp  polarization resist.", Rp, "Ω cm²", "#89b4fa")

        m = best.get("method", "—")
        r2l = best.get("r2_log")
        r2 = best.get("r2_lin")
        parts = []
        if r2l is not None: parts.append(f"R²(log) = {r2l:.4f}")
        if r2 is not None:  parts.append(f"R²(lin) = {r2:.4f}")
        cls = "ok-box" if (r2l and r2l >= 0.95) else "warn-box"
        pfx = "✅" if (r2l and r2l >= 0.95) else "⚠️"
        st.markdown(f'<div class="{cls}">{pfx} {m}<br>{", ".join(parts)}</div>',
                    unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sechead">🛡️ Passivity</div>', unsafe_allow_html=True)
        if curve_type in (CurveType.PASSIVE, CurveType.MIXED):
            pcard("ipass (fitted)", best.get("ipass"), "A cm⁻²", "#a6e3a1")
            if pas:
                pcard("ipass (measured)", pas.get("ipass"), "A cm⁻²", "#94e2d5")
                pcard("Passive start", pas.get("E_start"), "V", "#94e2d5")
                pcard("Passive end", pas.get("E_end"), "V", "#94e2d5")
                pcard("Passive range", pas.get("range_V", 0)*1000, "mV", "#a6adc8")
            if reg.get("Epp") is not None:
                pcard("Epp / Flade", reg["Epp"], "V", "#a6e3a1")
        else:
            st.info("No passive region detected.")

    with col4:
        st.markdown('<div class="sechead">💥 Breakdown & Diffusion</div>', unsafe_allow_html=True)
        if reg.get("Eb"):
            pcard("Eb  breakdown (detected)", reg["Eb"], "V", "#f38ba8")
        if best.get("Eb") and curve_type in (CurveType.PASSIVE, CurveType.MIXED):
            pcard("Eb  (fitted)", best["Eb"], "V", "#f38ba8")
            if Ec: pcard("Pitting index Eb−Ecorr",
                         (best["Eb"]-Ec)*1000, "mV", "#f38ba8")
        if curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
            if reg.get("iL"):
                pcard("iL (detected)", reg["iL"], "A cm⁻²", "#89dceb")
            if best.get("iL"):
                pcard("iL (fitted)", best["iL"], "A cm⁻²", "#89dceb")
        if best.get("a_tp") and curve_type in (CurveType.PASSIVE, CurveType.MIXED):
            pcard("Transpassive a_tp", best["a_tp"], "A cm⁻²", "#fab387")
            pcard("Transpassive b_tp", best["b_tp"], "V⁻¹", "#fab387")


def build_summary_df(R, reg, ew, rho, curve_type):
    best = R.get("best", {})
    an, ca, pas, rp_r = R.get("an",{}), R.get("ca",{}), R.get("passive",{}), R.get("rp",{})
    Ec = best.get("Ecorr", reg["Ecorr"])
    ic = best.get("icorr")
    ba, bc = best.get("ba"), best.get("bc")
    B = best.get("B")
    Rp = rp_r.get("Rp")
    CR = ic * 3.27 * ew / rho if ic else None
    ic_sg = B/Rp if (B and Rp) else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    rows = [
        ("Curve Type", CurveType.DESCRIPTIONS[curve_type][0]),
        ("Ecorr (V)", Ec), ("icorr (A/cm²)", ic),
        ("icorr_SG (A/cm²)", ic_sg),
        ("βa (mV/dec)", ba*1000 if ba else None),
        ("βc (mV/dec)", bc*1000 if bc else None),
        ("B (mV)", B*1000 if B else None),
        ("Rp (Ω·cm²)", Rp),
        ("CR (mm/yr)", CR), ("CR_SG (mm/yr)", CR_sg),
        ("ipass fitted (A/cm²)", best.get("ipass")),
        ("ipass measured (A/cm²)", pas.get("ipass")),
        ("iL fitted (A/cm²)", best.get("iL")),
        ("iL detected (A/cm²)", reg.get("iL")),
        ("Eb fitted (V)", best.get("Eb")),
        ("Eb detected (V)", reg.get("Eb")),
        ("Epp (V)", reg.get("Epp")),
        ("a_tp (A/cm²)", best.get("a_tp")),
        ("b_tp (V⁻¹)", best.get("b_tp")),
        ("Method", best.get("method")),
        ("R²(log)", best.get("r2_log")),
        ("R²(lin)", best.get("r2_lin")),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ═══════════════════════════════════════════════════════════════
# MATERIALS
# ═══════════════════════════════════════════════════════════════
MATERIALS = {
    "Carbon Steel / Iron": (27.92, 7.87),
    "304 Stainless Steel": (25.10, 7.90),
    "316 Stainless Steel": (25.56, 8.00),
    "Copper": (31.77, 8.96),
    "Aluminum": (8.99, 2.70),
    "Nickel": (29.36, 8.91),
    "Titanium": (11.99, 4.51),
    "Zinc": (32.69, 7.14),
    "Custom": (27.92, 7.87),
}


# ═══════════════════════════════════════════════════════════════
# DEMO DATA
# ═══════════════════════════════════════════════════════════════

def run_demo(choice):
    np.random.seed(42)

    if "full" in choice.lower() or "breakdown" in choice.lower():
        E = np.linspace(-0.65, 0.85, 500)
        Ecorr, icorr, ba, bc = -0.38, 2e-6, 0.065, 0.110
        ipass, iL = 5e-6, 5e-4
        Eb, a_tp, b_tp = 0.55, 5e-6, 12.0
        i = global_model(E, Ecorr, icorr, ba, bc, ipass, iL, Eb, a_tp, b_tp)

    elif "active" in choice.lower():
        E = np.linspace(-0.65, 0.45, 400)
        i = 5e-6*(np.exp(2.303*(E+0.45)/0.06) - np.exp(-2.303*(E+0.45)/0.12))

    elif "diffusion" in choice.lower():
        E = np.linspace(-0.80, 0.30, 400)
        Ecorr, icorr, ba, bc = -0.40, 3e-6, 0.070, 0.120
        iL = 2e-4
        i = global_model(E, Ecorr, icorr, ba, bc, 1e3, iL, 0.8, 1e-15, 1.0)

    else:  # passive only
        E = np.linspace(-0.55, 0.70, 400)
        Ecorr, icorr, ba, bc = -0.40, 2e-6, 0.065, 0.110
        ipass = 4e-6
        i = global_model(E, Ecorr, icorr, ba, bc, ipass, 1e3, 0.60, 3e-6, 8.0)

    noise = np.random.normal(0, np.abs(i)*0.03 + 3e-9, size=len(i))
    return E, i + noise


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_and_display(E_raw, i_dens, area, ew, rho):
    with st.spinner("Detecting regions & classifying curve…"):
        reg = detect_regions(E_raw, i_dens)
    curve_type = reg["curve_type"]

    with st.spinner(f"Fitting global model ({CurveType.DESCRIPTIONS[curve_type][0]})…"):
        fitter = GlobalFitter(E_raw, i_dens, reg)
        R = fitter.run()

    # ── Plots ───────────────────────────────────────────
    st.markdown("---")

    fig_full = plot_polarization(E_raw, i_dens, R, reg, curve_type)
    st.plotly_chart(fig_full, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_active_zoom(E_raw, i_dens, R, reg, curve_type),
                        use_container_width=True)
    with c2:
        fig_res = plot_residuals(E_raw, i_dens, R, reg, curve_type)
        if fig_res:
            st.plotly_chart(fig_res, use_container_width=True)

    # Component breakdown
    if curve_type != CurveType.ACTIVE_ONLY:
        with st.expander("🔬 Model component breakdown (anodic, cathodic, transpassive)"):
            fig_comp = plot_component_breakdown(E_raw, R, reg, curve_type)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)

    # ── Parameters ──────────────────────────────────────
    st.markdown("---")
    show_parameters(R, reg, ew, rho, curve_type)

    # ── Log & download ──────────────────────────────────
    st.markdown("---")
    c_log, c_dl = st.columns([3, 1])
    with c_log:
        with st.expander("🪵 Fitting log (detailed)"):
            for msg in fitter.log:
                st.markdown(f"- {msg}")
    with c_dl:
        df_sum = build_summary_df(R, reg, ew, rho, curve_type)
        st.download_button("⬇️ Results (CSV)", df_sum.to_csv(index=False).encode(),
                           "tafel_results.csv", "text/csv", use_container_width=True)
        df_proc = pd.DataFrame({"E_V": E_raw, "i_Acm2": i_dens,
                                 "log_abs_i": safe_log10(i_dens)})
        st.download_button("⬇️ Processed data (CSV)", df_proc.to_csv(index=False).encode(),
                           "tafel_data.csv", "text/csv", use_container_width=True)

    # ── Model equation display ──────────────────────────
    best = R.get("best", {})
    if best.get("success"):
        with st.expander("📐 Global model equation"):
            st.markdown(f"""
**Global Polarization Model:**

```
i_total = i_anodic − i_cathodic + i_transpassive
```

**Anodic** (activation → passive-limited):
```
i_a_kinetic = icorr · exp(2.303·η / βa)
i_anodic = i_a_kinetic · ipass / (i_a_kinetic + ipass)
```
→ When i_a << ipass: pure Tafel kinetics
→ When i_a >> ipass: saturates at ipass = **{best.get('ipass', 0):.3e}** A/cm²

**Cathodic** (activation → diffusion-limited):
```
i_c_kinetic = icorr · exp(−2.303·η / βc)
i_cathodic = i_c_kinetic / (1 + i_c_kinetic / iL)
```
→ When i_c << iL: pure Tafel kinetics
→ When i_c >> iL: saturates at iL = **{best.get('iL', 0):.3e}** A/cm²

**Transpassive** (above Eb):
```
i_tp = a_tp · exp(b_tp · (E − Eb)) · σ(E − Eb)
```
→ σ = sigmoid turn-on at Eb = **{best.get('Eb', 0):.3f}** V

**Fitted parameters:**
- Ecorr = {best.get('Ecorr', 0):.4f} V, icorr = {best.get('icorr', 0):.3e} A/cm²
- βa = {best.get('ba', 0)*1000:.1f} mV/dec, βc = {best.get('bc', 0)*1000:.1f} mV/dec
""")


def main():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                border:1px solid #313244; border-radius:12px;
                padding:20px 28px; margin-bottom:20px;">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting Tool v3</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">
        Global multi-region model · Fits entire curve (active + passive + transpassive + diffusion)
      </p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        area = st.number_input("Electrode area (cm²)", 0.001, 100.0, 1.0, 0.01)
        mat = st.selectbox("Material", list(MATERIALS.keys()))
        ew0, rho0 = MATERIALS[mat]
        if mat == "Custom":
            ew = st.number_input("EW (g/eq)", 1.0, 300.0, ew0)
            rho = st.number_input("ρ (g/cm³)", 0.5, 25.0, rho0)
        else:
            ew, rho = ew0, rho0

        st.divider()
        st.markdown("""
        <div style="font-size:11px; color:#a6adc8; line-height:2.0;">
        <b style="color:#89b4fa;">Global model equation:</b><br>
        i = i_anodic − i_cathodic + i_transpassive<br><br>
        🟡 <b>Anodic</b>: BV × passive saturation<br>
        🟣 <b>Cathodic</b>: BV × diffusion saturation<br>
        🟠 <b>Transpassive</b>: exponential above Eb<br><br>
        One model · one fit · entire curve
        </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your polarization data", type=["csv","txt","xlsx","xls"])

    if uploaded is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            demo_choice = st.selectbox("Or try a demo", [
                "Full curve (limiting + passive + breakdown)",
                "Active/Tafel only",
                "With passive region only",
                "Diffusion-limited cathodic"])
            if st.button("▶  Run demo", use_container_width=True, type="primary"):
                st.session_state["demo"] = demo_choice
            st.markdown("""<br>
            <div style="background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:16px 20px;">
            <b style="color:#89b4fa;">What's new in v3 — Global Fit</b><br><br>
            <span style="color:#a6adc8;font-size:13px;">
            ✓ <b>Single model fits the ENTIRE curve</b> — no separate local fits<br>
            ✓ Physics-based: BV + passive saturation + diffusion limit + transpassive<br>
            ✓ Fit line overlays all regions (active, passive, transpassive)<br>
            ✓ Component breakdown plot shows each current contribution<br>
            ✓ 3-stage optimization: DE global → Nelder-Mead → TRF gradient<br>
            ✓ Model equation display with all fitted parameters<br>
            ✓ Residual plot across full potential range
            </span></div>""", unsafe_allow_html=True)
        return

    # Process file
    with st.spinner("Reading…"):
        try: df = load_any_file(uploaded)
        except Exception as ex: st.error(f"Read error: {ex}"); return
    with st.spinner("Detecting columns…"):
        try: e_col, i_col, i_factor = auto_detect_columns(df)
        except Exception as ex: st.error(f"Column error: {ex}"); return

    with st.expander(f"📋 Auto-detected: **{e_col}** · **{i_col}**", expanded=False):
        st.dataframe(df[[e_col, i_col]].head(10), use_container_width=True)

    E_raw = df[e_col].values.astype(float)
    i_raw = df[i_col].values.astype(float) * i_factor
    ok = np.isfinite(E_raw) & np.isfinite(i_raw)
    E_raw, i_raw = E_raw[ok], i_raw[ok]
    idx = np.argsort(E_raw)
    E_raw, i_raw = E_raw[idx], i_raw[idx]
    i_dens = i_raw / area

    process_and_display(E_raw, i_dens, area, ew, rho)


if __name__ == "__main__":
    if "demo" in st.session_state:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                    border:1px solid #313244; border-radius:12px;
                    padding:20px 28px; margin-bottom:20px;">
          <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting v3</h1>
          <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">Demo</p>
        </div>""", unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            area = st.number_input("Electrode area (cm²)", 0.001, 100.0, 1.0)
            mat = st.selectbox("Material", list(MATERIALS.keys()))
            ew, rho = MATERIALS[mat]
            if mat == "Custom":
                ew = st.number_input("EW", 1.0, 300.0, ew)
                rho = st.number_input("ρ", 0.5, 25.0, rho)
            if st.button("← Back"):
                del st.session_state["demo"]; st.rerun()

        E_d, i_d = run_demo(st.session_state["demo"])
        st.info(f"🧪 Demo: **{st.session_state['demo']}**")
        process_and_display(E_d, i_d / area, area, ew, rho)
    else:
        main()
