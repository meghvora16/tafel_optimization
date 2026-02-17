"""
Tafel Fitting Tool — v2: Smart Curve Classification + Adaptive Fitting
=======================================================================
Auto-detects curve type (active, passive, diffusion-limited, mixed),
applies the correct electrochemical model, and iteratively optimizes
until the fit meets quality thresholds.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from itertools import groupby
import warnings, io, re

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Tafel Fitting v2", page_icon="⚡", layout="wide")

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

# ═══════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════
C = dict(
    data="#89b4fa", anodic="#f9e2af", cathodic="#cba6f7", bv="#a6e3a1",
    passive="rgba(166,227,161,0.12)", limiting="rgba(137,220,235,0.12)",
    transpassive="rgba(243,188,168,0.10)", ecorr="#f38ba8",
    grid="#313244", bg="#1e1e2e", paper="#181825", text="#cdd6f4",
    fit_band="rgba(166,227,161,0.15)",
)

# ═══════════════════════════════════════════════════════════════
# COLUMN AUTO-DETECTION
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
    r"\(a\)|_a$|/a$":    1.0,
    r"\(ma\)|_ma$|/ma$": 1e-3,
    r"\(µa\)|_ua$|/ua$": 1e-6,
    r"a/cm":             1.0,
    r"ma/cm":            1e-3,
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
        numeric_count = sum(1 for p in parts if re.match(r"^-?[\d.eE+]+$", p))
        if numeric_count >= 2:
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
                df = df.dropna(axis=1, how="all")
                return df
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

def _r2_log(y_true, y_pred):
    """R² computed in log-space for better assessment of fit quality across decades."""
    log_t = safe_log10(y_true)
    log_p = safe_log10(y_pred)
    return _r2(log_t, log_p)


# ═══════════════════════════════════════════════════════════════
# ELECTROCHEMICAL MODELS
# ═══════════════════════════════════════════════════════════════

def butler_volmer(E, Ecorr, icorr, ba, bc):
    """Standard Butler-Volmer equation."""
    eta = E - Ecorr
    return icorr * (np.exp(2.303 * eta / ba) - np.exp(-2.303 * eta / bc))

def bv_log_abs(E, Ecorr, icorr, ba, bc):
    """log10|i| from Butler-Volmer — for fitting in log-space."""
    i_bv = butler_volmer(E, Ecorr, icorr, ba, bc)
    return safe_log10(i_bv)

def bv_diffusion_limited(E, Ecorr, icorr, ba, bc, iL):
    """Butler-Volmer with cathodic diffusion-limited current.
    i = icorr * [exp(η/ba) - exp(-η/bc) / (1 + |icorr/iL| * exp(-2.303*η/bc))]
    This correctly asymptotes to iL for large cathodic overpotentials.
    """
    eta = E - Ecorr
    i_a = icorr * np.exp(2.303 * eta / ba)
    i_c_kinetic = icorr * np.exp(-2.303 * eta / bc)
    # Mixed kinetic-diffusion cathodic branch
    i_c = i_c_kinetic / (1.0 + i_c_kinetic / np.abs(iL))
    return i_a - i_c

def bv_passive(E, Ecorr, icorr, ba, bc, ipass, Epp):
    """Butler-Volmer that transitions to passive current above Epp.
    Uses a smooth sigmoid transition to avoid discontinuities.
    """
    eta = E - Ecorr
    # Active BV current
    i_active = icorr * (np.exp(2.303 * eta / ba) - np.exp(-2.303 * eta / bc))
    # Smooth transition to passive current
    # sigmoid centered at Epp, width ~30 mV
    transition = 1.0 / (1.0 + np.exp(-80.0 * (E - Epp)))
    # In passive region, current is limited to ipass
    i_passive = np.sign(i_active) * ipass
    # For E > Epp, blend from active to passive
    i_result = i_active * (1.0 - transition) + i_passive * transition
    return i_result

def bv_full_mixed(E, Ecorr, icorr, ba, bc, iL, ipass, Epp):
    """Full model: BV + diffusion-limited cathodic + passive anodic."""
    eta = E - Ecorr
    i_a_kinetic = icorr * np.exp(2.303 * eta / ba)
    i_c_kinetic = icorr * np.exp(-2.303 * eta / bc)
    i_c = i_c_kinetic / (1.0 + i_c_kinetic / np.abs(iL))
    i_active = i_a_kinetic - i_c
    transition = 1.0 / (1.0 + np.exp(-80.0 * (E - Epp)))
    i_passive = ipass
    i_result = i_active * (1.0 - transition) + i_passive * transition
    return i_result


# ═══════════════════════════════════════════════════════════════
# CURVE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

class CurveType:
    ACTIVE_ONLY = "active_only"
    PASSIVE = "passive"
    DIFFUSION_LIMITED = "diffusion_limited"
    MIXED = "mixed"  # both passive + diffusion

    DESCRIPTIONS = {
        ACTIVE_ONLY: ("⚡ Active / Standard Tafel",
                      "Pure activation-controlled system. Standard Butler-Volmer applies "
                      "on both branches. Tafel slopes directly measurable."),
        PASSIVE: ("🛡️ Passive System",
                  "Material forms a protective passive film. Anodic branch shows "
                  "activation → passivation transition. Fitting restricted to "
                  "activation-controlled region only. Uses modified BV+passive model."),
        DIFFUSION_LIMITED: ("🌊 Diffusion-Limited Cathodic",
                            "Cathodic reaction is mass-transport limited (O₂ reduction). "
                            "Uses BV equation with limiting current correction. "
                            "Cathodic Tafel extracted from mixed kinetic-diffusion region."),
        MIXED: ("🔀 Mixed: Passive + Diffusion-Limited",
                "Both cathodic diffusion limitation and anodic passivation detected. "
                "Full mixed model applied. Parameters extracted from narrow "
                "activation window near Ecorr."),
    }


def classify_curve(E, i, reg):
    """Classify the polarization curve type based on detected features."""
    has_passive = reg.get("passive_s") is not None
    has_limiting = reg.get("iL") is not None

    if has_passive and has_limiting:
        return CurveType.MIXED
    elif has_passive:
        return CurveType.PASSIVE
    elif has_limiting:
        return CurveType.DIFFUSION_LIMITED
    else:
        return CurveType.ACTIVE_ONLY


# ═══════════════════════════════════════════════════════════════
# REGION DETECTION (improved)
# ═══════════════════════════════════════════════════════════════

def detect_regions(E, i):
    """Robust region detection using derivative analysis."""
    reg = {}
    n = len(E)
    abs_i = np.abs(i)
    log_abs = safe_log10(i)

    # ── Ecorr: zero-crossing interpolation ──────────────
    sc = np.where(np.diff(np.sign(i)))[0]
    if len(sc) > 0:
        k = sc[0]; d = i[k+1] - i[k]
        reg["Ecorr"] = float(E[k] - i[k]*(E[k+1]-E[k])/d) if abs(d) > 0 else float(E[k])
        reg["ecorr_idx"] = k
    else:
        k = int(np.argmin(np.abs(i)))
        reg["Ecorr"] = float(E[k]); reg["ecorr_idx"] = k

    Ec = reg["Ecorr"]
    ec_idx = reg["ecorr_idx"]

    # ── Cathodic diffusion-limited region detection ─────
    ci = np.where(E < Ec)[0]
    if len(ci) >= 8:
        log_c = smooth(safe_log10(abs_i[ci]), min(11, (len(ci)//2)*2-1 or 5))
        dlog = np.gradient(log_c, E[ci])
        abs_dlog = np.abs(dlog)

        # Diffusion limiting = region where |d(log|i|)/dE| is very small
        # AND current is relatively high (plateau at high cathodic overpotential)
        # Use adaptive threshold: < 20th percentile of derivative magnitude
        threshold = np.percentile(abs_dlog, 20)
        flat = abs_dlog < max(threshold, 0.5)  # at least 0.5 dec/V

        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            best_run = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best_run[1]]
            # Only count as limiting if it's far enough from Ecorr (>100mV)
            # and the plateau spans enough points
            e_range = abs(E[ci[idxs[-1]]] - E[ci[idxs[0]]])
            far_from_ecorr = abs(E[ci[idxs[0]]] - Ec) > 0.08
            if len(idxs) >= 4 and e_range > 0.03 and far_from_ecorr:
                reg.update(
                    limit_idx=ci[idxs],
                    iL=float(np.median(abs_i[ci[idxs]])),
                    E_lim_start=float(E[ci[idxs[0]]]),
                    E_lim_end=float(E[ci[idxs[-1]]]),
                )

    # ── Anodic passive region detection ─────────────────
    ai = np.where(E > Ec)[0]
    if len(ai) >= 8:
        log_a = smooth(safe_log10(abs_i[ai]), min(11, (len(ai)//2)*2-1 or 5))
        dlog = np.gradient(log_a, E[ai])
        abs_dlog = np.abs(dlog)

        # Passive = region of very low |d(log|i|)/dE| on anodic side
        # Typically the current density stays nearly constant in this region
        threshold = np.percentile(abs_dlog, 30)
        flat = abs_dlog < max(threshold, 1.0)

        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            best_run = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best_run[1]]
            e_range = abs(E[ai[idxs[-1]]] - E[ai[idxs[0]]])

            # Passive region must:
            # 1. Span a reasonable potential range (>50 mV)
            # 2. Have current density that is significantly lower than the active peak
            # 3. Be at least slightly above Ecorr
            if len(idxs) >= 4 and e_range > 0.04:
                ps, pe = ai[idxs[0]], ai[idxs[-1]]

                # Check if current in this region is lower than the peak anodic current
                # (a true passive region has lower current than the active dissolution peak)
                i_in_region = np.median(abs_i[ps:pe+1])
                # Find peak anodic current between Ecorr and passive start
                pre_passive = np.where((E > Ec) & (E < E[ps]))[0]
                if len(pre_passive) > 2:
                    i_peak = np.max(abs_i[pre_passive])
                    is_passive = i_in_region < i_peak * 0.8
                else:
                    # No clear active peak — check if derivative is genuinely flat
                    is_passive = e_range > 0.10

                if is_passive:
                    reg.update(
                        passive_s=int(ps), passive_e=int(pe),
                        E_ps=float(E[ps]), E_pe=float(E[pe]),
                        ipass=float(np.median(abs_i[ps:pe+1])),
                        Epp=float(E[ps]),
                    )

                    # Breakdown detection
                    if pe + 3 < n:
                        d_after = np.gradient(safe_log10(abs_i[pe:]), E[pe:])
                        thr = np.mean(abs_dlog) + 2.0 * np.std(abs_dlog)
                        jump = np.where(np.abs(d_after) > max(thr, 3.0))[0]
                        if len(jump):
                            eb_idx = pe + jump[0]
                            reg["Eb_idx"] = int(eb_idx)
                            reg["Eb"] = float(E[eb_idx])
                        reg["tp_idx"] = reg.get("Eb_idx", pe)

    return reg


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE FITTING ENGINE
# ═══════════════════════════════════════════════════════════════

class AdaptiveFitter:
    """
    Curve-type-aware fitter that:
    1. Classifies the curve
    2. Selects appropriate model & fitting window
    3. Fits with multi-strategy optimization
    4. Checks fit quality — if bad, widens search or changes model
    """

    R2_GOOD = 0.990
    R2_ACCEPTABLE = 0.950
    R2_MINIMUM = 0.85

    def __init__(self, E, i, reg):
        self.E = E
        self.i = i
        self.abs_i = np.abs(i)
        self.log_abs_i = safe_log10(i)
        self.reg = reg
        self.Ec = reg["Ecorr"]
        self.ec_idx = reg["ecorr_idx"]
        self.ic0 = max(float(np.abs(i[self.ec_idx])), 1e-12)
        self.R = {}
        self.log = []
        self.curve_type = classify_curve(E, i, reg)

    # ── Tafel window search (improved) ────────────────────

    def _find_best_tafel_window(self, side="anodic"):
        """Grid search for the best linear Tafel region.
        Returns dict with slope, intercept, r2, ba/bc, mask, etc. or None.
        """
        E, log_i, Ec = self.E, self.log_abs_i, self.Ec

        if side == "anodic":
            # Exclude passive region
            E_limit = self.reg.get("E_ps", Ec + 0.50)
            # Also limit to reasonable distance from Ecorr
            E_limit = min(E_limit - 0.005, Ec + 0.30)

            best = None
            for lo in np.arange(0.005, 0.06, 0.005):
                for hi in np.arange(lo + 0.015, min(lo + 0.20, E_limit - Ec), 0.005):
                    m = (E > Ec + lo) & (E < Ec + hi)
                    if m.sum() < 4:
                        continue
                    s, b, r, *_ = linregress(E[m], log_i[m])
                    if s <= 0:
                        continue
                    ba_mV = (1/s) * 1000
                    if 20 < ba_mV < 500 and r**2 > 0.90:
                        if best is None or r**2 > best["r2"]:
                            best = dict(slope=s, intercept=b, r2=r**2,
                                        ba=1/s, mask=m, lo=lo, hi=hi, n=int(m.sum()))
            return best

        else:  # cathodic
            E_lim_end = self.reg.get("E_lim_end")
            E_lower = E_lim_end + 0.005 if E_lim_end is not None else Ec - 0.50

            best = None
            for lo in np.arange(0.005, 0.10, 0.005):
                for hi in np.arange(lo + 0.015, min(lo + 0.30, Ec - E_lower), 0.005):
                    m = (E < Ec - lo) & (E > Ec - hi)
                    if E_lim_end is not None:
                        m = m & (E > E_lim_end + 0.005)
                    if m.sum() < 4:
                        continue
                    s, b, r, *_ = linregress(E[m], log_i[m])
                    if s >= 0:
                        continue
                    bc_mV = (-1/s) * 1000
                    if 20 < bc_mV < 500 and r**2 > 0.90:
                        if best is None or r**2 > best["r2"]:
                            best = dict(slope=s, intercept=b, r2=r**2,
                                        bc=-1/s, mask=m, lo=lo, hi=hi, n=int(m.sum()))
            return best

    # ── Tafel extrapolation ───────────────────────────────

    def fit_tafel_lines(self):
        an = self._find_best_tafel_window("anodic")
        ca = self._find_best_tafel_window("cathodic")

        if an:
            self.R["an"] = an
            self.log.append(
                f"✅ Anodic Tafel: βa = {an['ba']*1000:.1f} mV/dec, "
                f"R² = {an['r2']:.4f}, window [{self.Ec+an['lo']:.3f}–"
                f"{self.Ec+an['hi']:.3f} V] ({an['n']} pts)")
        else:
            if self.curve_type == CurveType.PASSIVE:
                self.log.append(
                    "ℹ️ No anodic Tafel region — expected for passive system. "
                    "βa will come from model fit near Ecorr.")
            else:
                self.log.append("⚠️ No clean anodic Tafel region found.")

        if ca:
            self.R["ca"] = ca
            self.log.append(
                f"✅ Cathodic Tafel: βc = {ca['bc']*1000:.1f} mV/dec, "
                f"R² = {ca['r2']:.4f}, window [{self.Ec-ca['hi']:.3f}–"
                f"{self.Ec-ca['lo']:.3f} V] ({ca['n']} pts)")
        else:
            if self.curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
                self.log.append(
                    "ℹ️ No clean cathodic Tafel region — expected for "
                    "diffusion-limited system. βc from mixed-kinetic model.")
            else:
                self.log.append("⚠️ No clean cathodic Tafel region found.")

        # Intersection
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

    # ── Fitting mask based on curve type ──────────────────

    def _get_fitting_mask(self, half_width=None):
        """Get the data mask for model fitting, appropriate for curve type."""
        E, Ec = self.E, self.Ec
        reg = self.reg

        if self.curve_type == CurveType.ACTIVE_ONLY:
            hw = half_width or 0.25
            return (E > Ec - hw) & (E < Ec + hw)

        elif self.curve_type == CurveType.PASSIVE:
            # Fit ONLY the activation region: Ecorr ± window, stopping before passivation
            E_ps = reg.get("E_ps", Ec + 0.30)
            hw_cathodic = half_width or 0.20
            hw_anodic = min(0.15, (E_ps - Ec) * 0.85)
            return (E > Ec - hw_cathodic) & (E < Ec + hw_anodic)

        elif self.curve_type == CurveType.DIFFUSION_LIMITED:
            # Include wider cathodic range to capture diffusion limitation
            E_lim = reg.get("E_lim_end", Ec - 0.40)
            hw_anodic = half_width or 0.20
            # Include data up to the limiting region
            return (E > E_lim - 0.02) & (E < Ec + hw_anodic)

        else:  # MIXED
            E_ps = reg.get("E_ps", Ec + 0.30)
            E_lim = reg.get("E_lim_end", Ec - 0.40)
            hw_anodic = min(0.15, (E_ps - Ec) * 0.85)
            return (E > E_lim - 0.02) & (E < Ec + hw_anodic)

    # ── Model fitting based on curve type ─────────────────

    def _fit_active_model(self):
        """Standard Butler-Volmer for active-only curves."""
        E, i = self.E, self.i
        ba0 = self.R.get("an", {}).get("ba", 0.06)
        bc0 = self.R.get("ca", {}).get("bc", 0.12)

        results = []
        # Try multiple fitting window sizes
        for hw in [0.15, 0.20, 0.25, 0.30, 0.40]:
            m = self._get_fitting_mask(hw)
            if m.sum() < 6:
                continue
            E_f, i_f = E[m], i[m]

            # Strategy 1: Direct BV fit in current space
            p0 = [self.Ec, self.ic0, ba0, bc0]
            bnd = ([E.min(), 1e-14, 0.005, 0.005],
                   [E.max(), 1e-1, 0.8, 0.8])
            try:
                popt, _ = curve_fit(butler_volmer, E_f, i_f, p0=p0, bounds=bnd,
                                    maxfev=50000, method="trf", ftol=1e-14, xtol=1e-14)
                pred = butler_volmer(E_f, *popt)
                r2 = _r2(i_f, pred)
                r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))
                results.append(dict(
                    Ecorr=float(popt[0]), icorr=float(popt[1]),
                    ba=float(popt[2]), bc=float(popt[3]),
                    r2=r2, r2_log=r2_log, hw=hw,
                    method="Butler-Volmer (TRF)", success=True))
            except Exception:
                pass

            # Strategy 2: Log-space fitting (better for wide dynamic range)
            try:
                log_abs_f = safe_log10(i_f)
                p0_log = [self.Ec, self.ic0, ba0, bc0]
                popt2, _ = curve_fit(bv_log_abs, E_f, log_abs_f, p0=p0_log, bounds=bnd,
                                     maxfev=50000, method="trf", ftol=1e-14, xtol=1e-14)
                pred2 = butler_volmer(E_f, *popt2)
                r2 = _r2(i_f, pred2)
                r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred2)))
                results.append(dict(
                    Ecorr=float(popt2[0]), icorr=float(popt2[1]),
                    ba=float(popt2[2]), bc=float(popt2[3]),
                    r2=r2, r2_log=r2_log, hw=hw,
                    method="Butler-Volmer log-space (TRF)", success=True))
            except Exception:
                pass

        return results

    def _fit_passive_model(self):
        """Modified BV for passive systems — fit only the active region near Ecorr."""
        E, i = self.E, self.i
        ba0 = self.R.get("an", {}).get("ba", 0.06)
        bc0 = self.R.get("ca", {}).get("bc", 0.12)

        results = []
        # For passive systems: narrow fitting window, stay in activation region
        E_ps = self.reg.get("E_ps", self.Ec + 0.25)

        for hw_cat in [0.10, 0.15, 0.20, 0.25]:
            for hw_an_frac in [0.6, 0.7, 0.8, 0.9]:
                hw_an = min(hw_cat, (E_ps - self.Ec) * hw_an_frac)
                if hw_an < 0.01:
                    continue
                m = (E > self.Ec - hw_cat) & (E < self.Ec + hw_an)
                if m.sum() < 6:
                    continue
                E_f, i_f = E[m], i[m]

                p0 = [self.Ec, self.ic0, ba0, bc0]
                bnd = ([E.min(), 1e-14, 0.005, 0.005],
                       [E.max(), 1e-1, 0.8, 0.8])
                try:
                    popt, _ = curve_fit(butler_volmer, E_f, i_f, p0=p0, bounds=bnd,
                                        maxfev=50000, method="trf", ftol=1e-14, xtol=1e-14)
                    pred = butler_volmer(E_f, *popt)
                    r2 = _r2(i_f, pred)
                    r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))
                    results.append(dict(
                        Ecorr=float(popt[0]), icorr=float(popt[1]),
                        ba=float(popt[2]), bc=float(popt[3]),
                        r2=r2, r2_log=r2_log,
                        hw_cat=hw_cat, hw_an=hw_an,
                        method="BV active-region only (TRF)", success=True))
                except Exception:
                    pass

                # Also try log-space
                try:
                    log_abs_f = safe_log10(i_f)
                    popt2, _ = curve_fit(bv_log_abs, E_f, log_abs_f, p0=p0, bounds=bnd,
                                         maxfev=50000, method="trf", ftol=1e-14, xtol=1e-14)
                    pred2 = butler_volmer(E_f, *popt2)
                    r2 = _r2(i_f, pred2)
                    r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred2)))
                    results.append(dict(
                        Ecorr=float(popt2[0]), icorr=float(popt2[1]),
                        ba=float(popt2[2]), bc=float(popt2[3]),
                        r2=r2, r2_log=r2_log,
                        hw_cat=hw_cat, hw_an=hw_an,
                        method="BV active-region log-space (TRF)", success=True))
                except Exception:
                    pass

        return results

    def _fit_diffusion_model(self):
        """BV with cathodic diffusion-limited current correction."""
        E, i = self.E, self.i
        ba0 = self.R.get("an", {}).get("ba", 0.06)
        bc0 = self.R.get("ca", {}).get("bc", 0.12)
        iL0 = self.reg.get("iL", 1e-3)

        results = []
        for hw in [0.20, 0.30, 0.40, 0.50]:
            m = self._get_fitting_mask(hw)
            if m.sum() < 8:
                continue
            E_f, i_f = E[m], i[m]

            p0 = [self.Ec, self.ic0, ba0, bc0, iL0]
            bnd = ([E.min(), 1e-14, 0.005, 0.005, iL0*0.1],
                   [E.max(), 1e-1, 0.8, 0.8, iL0*10])
            try:
                popt, _ = curve_fit(bv_diffusion_limited, E_f, i_f, p0=p0, bounds=bnd,
                                    maxfev=80000, method="trf", ftol=1e-14, xtol=1e-14)
                pred = bv_diffusion_limited(E_f, *popt)
                r2 = _r2(i_f, pred)
                r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))
                results.append(dict(
                    Ecorr=float(popt[0]), icorr=float(popt[1]),
                    ba=float(popt[2]), bc=float(popt[3]),
                    iL=float(popt[4]),
                    r2=r2, r2_log=r2_log, hw=hw,
                    method="BV+Diffusion (TRF)", success=True))
            except Exception:
                pass

        # Also try: fit active region only with standard BV (cathodic-emphasized)
        for hw_an in [0.10, 0.15, 0.20]:
            for hw_cat in [0.05, 0.08, 0.12]:
                E_lim = self.reg.get("E_lim_end", self.Ec - 0.40)
                m = (E > max(E_lim + 0.01, self.Ec - hw_cat * 3)) & (E < self.Ec + hw_an)
                m = m & (E > self.Ec - hw_cat)  # narrow cathodic window
                if m.sum() < 5:
                    continue
                E_f, i_f = E[m], i[m]
                p0 = [self.Ec, self.ic0, ba0, bc0]
                bnd = ([E.min(), 1e-14, 0.005, 0.005],
                       [E.max(), 1e-1, 0.8, 0.8])
                try:
                    popt, _ = curve_fit(butler_volmer, E_f, i_f, p0=p0, bounds=bnd,
                                        maxfev=50000, method="trf", ftol=1e-14, xtol=1e-14)
                    pred = butler_volmer(E_f, *popt)
                    r2 = _r2(i_f, pred)
                    r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))
                    results.append(dict(
                        Ecorr=float(popt[0]), icorr=float(popt[1]),
                        ba=float(popt[2]), bc=float(popt[3]),
                        r2=r2, r2_log=r2_log,
                        hw_an=hw_an, hw_cat=hw_cat,
                        method="BV narrow-window (TRF)", success=True))
                except Exception:
                    pass

        return results

    def _fit_mixed_model(self):
        """Full mixed model with both diffusion and passivation."""
        results = []
        # Get results from both sub-models
        results.extend(self._fit_passive_model())
        results.extend(self._fit_diffusion_model())
        return results

    # ── Differential Evolution global optimizer ───────────

    def _fit_de_global(self, best_so_far=None):
        """Global optimization with DE as fallback/refinement."""
        E, i = self.E, self.i
        ba0 = best_so_far.get("ba", 0.06) if best_so_far else 0.06
        bc0 = best_so_far.get("bc", 0.12) if best_so_far else 0.12
        ic0 = best_so_far.get("icorr", self.ic0) if best_so_far else self.ic0

        results = []
        model_func = butler_volmer
        n_params = 4

        if self.curve_type == CurveType.DIFFUSION_LIMITED:
            iL0 = self.reg.get("iL", 1e-3)
            model_func = bv_diffusion_limited
            n_params = 5

        for hw in [0.15, 0.25, 0.35]:
            m = self._get_fitting_mask(hw)
            if m.sum() < 6:
                continue
            E_f, i_f = E[m], i[m]
            log_abs_f = safe_log10(np.abs(i_f))

            # Objective: minimize weighted residual in both linear and log space
            def objective(p):
                try:
                    if n_params == 5:
                        pred = model_func(E_f, p[0], 10**p[1], p[2], p[3], 10**p[4])
                    else:
                        pred = model_func(E_f, p[0], 10**p[1], p[2], p[3])

                    # Combined objective: log-space residual (main) + linear residual
                    log_pred = safe_log10(pred)
                    res_log = np.sum((log_abs_f - log_pred)**2)
                    res_lin = np.sum(((i_f - pred) / (np.abs(i_f) + 1e-12))**2)
                    return res_log + 0.1 * res_lin
                except:
                    return 1e30

            if n_params == 5:
                iL0 = self.reg.get("iL", 1e-3)
                bounds = [
                    (self.Ec - 0.15, self.Ec + 0.15),
                    (np.log10(max(ic0 * 1e-4, 1e-14)), np.log10(ic0 * 1e4)),
                    (0.01, 0.5), (0.01, 0.5),
                    (np.log10(max(iL0 * 0.01, 1e-10)), np.log10(iL0 * 100)),
                ]
            else:
                bounds = [
                    (self.Ec - 0.15, self.Ec + 0.15),
                    (np.log10(max(ic0 * 1e-4, 1e-14)), np.log10(ic0 * 1e4)),
                    (0.01, 0.5), (0.01, 0.5),
                ]

            try:
                res = differential_evolution(objective, bounds, seed=42,
                                             maxiter=3000, tol=1e-12,
                                             popsize=30, workers=1,
                                             mutation=(0.5, 1.5), recombination=0.9)
                p = res.x
                if n_params == 5:
                    pred = model_func(E_f, p[0], 10**p[1], p[2], p[3], 10**p[4])
                    result = dict(
                        Ecorr=float(p[0]), icorr=float(10**p[1]),
                        ba=float(p[2]), bc=float(p[3]), iL=float(10**p[4]),
                        method="BV+Diffusion (DE global)")
                else:
                    pred = model_func(E_f, p[0], 10**p[1], p[2], p[3])
                    result = dict(
                        Ecorr=float(p[0]), icorr=float(10**p[1]),
                        ba=float(p[2]), bc=float(p[3]),
                        method="Butler-Volmer (DE global)")

                r2 = _r2(i_f, pred)
                r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))
                result.update(r2=r2, r2_log=r2_log, hw=hw, success=True, fallback=True)
                results.append(result)
            except Exception as ex:
                self.log.append(f"⚠️ DE failed for hw={hw}: {ex}")

        return results

    # ── Nelder-Mead refinement ────────────────────────────

    def _refine_nelder_mead(self, best):
        """Polish the best solution with Nelder-Mead simplex."""
        E, i = self.E, self.i
        hw = best.get("hw", best.get("hw_cat", 0.20))
        m = self._get_fitting_mask(hw)
        if m.sum() < 6:
            return best
        E_f, i_f = E[m], i[m]
        log_abs_f = safe_log10(np.abs(i_f))

        has_iL = "iL" in best
        model_func = bv_diffusion_limited if has_iL else butler_volmer

        def objective(p):
            try:
                if has_iL:
                    pred = model_func(E_f, p[0], 10**p[1], p[2], p[3], 10**p[4])
                else:
                    pred = model_func(E_f, p[0], 10**p[1], p[2], p[3])
                log_pred = safe_log10(pred)
                return float(np.sum((log_abs_f - log_pred)**2))
            except:
                return 1e30

        x0 = [best["Ecorr"], np.log10(max(best["icorr"], 1e-14)),
              best["ba"], best["bc"]]
        if has_iL:
            x0.append(np.log10(max(best["iL"], 1e-14)))

        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-14})
            p = res.x
            if has_iL:
                pred = model_func(E_f, p[0], 10**p[1], p[2], p[3], 10**p[4])
            else:
                pred = model_func(E_f, p[0], 10**p[1], p[2], p[3])

            r2 = _r2(i_f, pred)
            r2_log = _r2(safe_log10(np.abs(i_f)), safe_log10(np.abs(pred)))

            refined = dict(best)
            refined.update(
                Ecorr=float(p[0]), icorr=float(10**p[1]),
                ba=float(p[2]), bc=float(p[3]),
                r2=r2, r2_log=r2_log,
                method=best["method"] + " + NM refined")
            if has_iL:
                refined["iL"] = float(10**p[4])

            if r2_log >= best.get("r2_log", 0):
                return refined
        except:
            pass
        return best

    # ── Polarization resistance ───────────────────────────

    def fit_rp(self):
        E, i, Ec = self.E, self.i, self.Ec
        for dE in [0.010, 0.015, 0.025, 0.05]:
            m = np.abs(E - Ec) < dE
            if m.sum() >= 4:
                s, _, r, *_ = linregress(E[m], i[m])
                if abs(s) > 1e-20:
                    self.R["rp"] = dict(Rp=float(1/s), r2=float(r**2), dE=dE)
                    self.log.append(
                        f"✅ Rp = {1/s:.3e} Ω·cm² "
                        f"(±{dE*1000:.0f} mV window, R² = {r**2:.3f})")
                    return

    # ── Passive/transpassive parameters ───────────────────

    def fit_passive_params(self):
        E, i, reg = self.E, self.i, self.reg
        ps, pe = reg.get("passive_s"), reg.get("passive_e")
        if ps is not None and pe is not None:
            iabs = np.abs(i[ps:pe+1])
            self.R["passive"] = dict(
                ipass=float(np.median(iabs)),
                E_start=float(E[ps]), E_end=float(E[pe]),
                range_V=float(E[pe] - E[ps]))
        tp = reg.get("tp_idx")
        if tp is not None and tp + 4 < len(E):
            s, _, r, *_ = linregress(E[tp:], safe_log10(np.abs(i[tp:])))
            self.R["tp"] = dict(slope=float(s), r2=float(r**2), E_start=float(E[tp]))

    # ── Master run ────────────────────────────────────────

    def run(self):
        self.log.append(f"🔍 Curve classified as: **{CurveType.DESCRIPTIONS[self.curve_type][0]}**")
        self.log.append(f"   {CurveType.DESCRIPTIONS[self.curve_type][1]}")

        # Step 1: Tafel slopes
        self.fit_tafel_lines()

        # Step 2: Type-specific model fitting
        self.log.append("─" * 40)
        self.log.append("🔧 Running type-specific model fitting...")

        if self.curve_type == CurveType.ACTIVE_ONLY:
            candidates = self._fit_active_model()
        elif self.curve_type == CurveType.PASSIVE:
            candidates = self._fit_passive_model()
        elif self.curve_type == CurveType.DIFFUSION_LIMITED:
            candidates = self._fit_diffusion_model()
        else:
            candidates = self._fit_mixed_model()

        # Step 3: Check quality and run DE if needed
        best_candidate = None
        if candidates:
            # Pick best by r2_log (better metric for wide dynamic range)
            candidates.sort(key=lambda c: c.get("r2_log", 0), reverse=True)
            best_candidate = candidates[0]
            self.log.append(
                f"✅ Best initial fit: {best_candidate['method']}, "
                f"R²(log) = {best_candidate.get('r2_log', 0):.4f}, "
                f"R²(lin) = {best_candidate.get('r2', 0):.4f}")

        if best_candidate is None or best_candidate.get("r2_log", 0) < self.R2_ACCEPTABLE:
            self.log.append("⚠️ Fit quality below threshold — running global DE optimization...")
            de_candidates = self._fit_de_global(best_candidate)
            if de_candidates:
                de_candidates.sort(key=lambda c: c.get("r2_log", 0), reverse=True)
                if best_candidate is None or de_candidates[0].get("r2_log", 0) > best_candidate.get("r2_log", 0):
                    best_candidate = de_candidates[0]
                    self.log.append(
                        f"✅ DE improved fit: R²(log) = {best_candidate.get('r2_log', 0):.4f}")

        # Step 4: Nelder-Mead refinement
        if best_candidate:
            refined = self._refine_nelder_mead(best_candidate)
            if refined.get("r2_log", 0) > best_candidate.get("r2_log", 0):
                best_candidate = refined
                self.log.append(
                    f"✅ NM refinement: R²(log) = {best_candidate.get('r2_log', 0):.4f}")

        # Step 5: If still not great, try last-resort cathodic-only extrapolation
        if best_candidate is None or best_candidate.get("r2_log", 0) < self.R2_MINIMUM:
            self.log.append("⚠️ Model fit poor — using Tafel extrapolation as fallback.")
            ca = self.R.get("ca")
            an = self.R.get("an")
            if ca or an:
                ec = self.R.get("Ecorr_tafel", self.Ec)
                ic = self.R.get("icorr_tafel", self.ic0)
                best_candidate = dict(
                    Ecorr=ec, icorr=ic,
                    ba=an["ba"] if an else None,
                    bc=ca["bc"] if ca else None,
                    r2=None, r2_log=None,
                    method="Tafel Extrapolation only", success=False)

        # Step 6: Finalize
        if best_candidate:
            ba, bc = best_candidate.get("ba"), best_candidate.get("bc")
            if ba and bc and ba > 0 and bc > 0:
                best_candidate["B"] = (ba * bc) / (2.303 * (ba + bc))
            self.R["best"] = best_candidate

            # Store the fitting mask for visualization
            hw = best_candidate.get("hw", best_candidate.get("hw_cat", 0.20))
            self.R["fit_mask"] = self._get_fitting_mask(hw)

        # Additional parameters
        self.fit_rp()
        self.fit_passive_params()

        # Quality summary
        self.log.append("─" * 40)
        r2l = best_candidate.get("r2_log") if best_candidate else None
        if r2l is not None:
            if r2l >= self.R2_GOOD:
                self.log.append(f"🎯 **Excellent fit** — R²(log) = {r2l:.4f}")
            elif r2l >= self.R2_ACCEPTABLE:
                self.log.append(f"✅ **Good fit** — R²(log) = {r2l:.4f}")
            elif r2l >= self.R2_MINIMUM:
                self.log.append(f"⚠️ **Acceptable fit** — R²(log) = {r2l:.4f}")
            else:
                self.log.append(f"❌ **Poor fit** — R²(log) = {r2l:.4f}. "
                                "Data may not follow standard electrochemical models.")

        return self.R


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def _generate_model_curve(E_range, best, curve_type, reg):
    """Generate the model prediction curve based on curve type and parameters."""
    Ecorr = best["Ecorr"]
    icorr = best["icorr"]
    ba = best.get("ba")
    bc = best.get("bc")

    if not (ba and bc and icorr):
        return None, None

    if "iL" in best and curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
        i_model = bv_diffusion_limited(E_range, Ecorr, icorr, ba, bc, best["iL"])
    else:
        i_model = butler_volmer(E_range, Ecorr, icorr, ba, bc)

    return E_range, i_model


def plot_polarization(E, i, R, reg, curve_type):
    an = R.get("an", {})
    ca = R.get("ca", {})
    best = R.get("best", {})
    pas = R.get("passive", {})
    tp = R.get("tp", {})
    log_i = safe_log10(i)
    fit_mask = R.get("fit_mask")

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
    if tp.get("E_start") is not None:
        fig.add_vrect(x0=tp["E_start"], x1=float(E[-1]),
                      fillcolor=C["transpassive"], layer="below", line_width=0,
                      annotation=dict(text="Transpassive",
                                      font=dict(color="#fab387", size=11), yanchor="top"))

    # ── Fitting region highlight ────────────────────────
    if fit_mask is not None and fit_mask.any():
        E_fit_min = E[fit_mask].min()
        E_fit_max = E[fit_mask].max()
        fig.add_vrect(x0=E_fit_min, x1=E_fit_max,
                      fillcolor=C["fit_band"], layer="below", line_width=0,
                      annotation=dict(text="Fit region",
                                      font=dict(color="#a6e3a1", size=9),
                                      yanchor="bottom", y=0))

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
                             hovertemplate="E = %{x:.4f} V<br>log|i| = %{y:.3f}<extra></extra>"))

    # ── Tafel lines (only in their valid range) ─────────
    E_full = np.linspace(E.min(), E.max(), 400)
    if an.get("slope"):
        # Only draw anodic Tafel within the actual Tafel region (+ small extension)
        an_lo = Ec + an.get("lo", 0) - 0.03
        an_hi = Ec + an.get("hi", 0.15) + 0.05
        if curve_type == CurveType.PASSIVE:
            an_hi = min(an_hi, reg.get("E_ps", an_hi) + 0.02)
        E_an = np.linspace(an_lo, an_hi, 100)
        y_an = an["slope"] * E_an + an["intercept"]
        fig.add_trace(go.Scatter(x=E_an, y=y_an, mode="lines",
                                 name=f"Anodic Tafel  βa={an['ba']*1000:.0f} mV/dec  R²={an['r2']:.3f}",
                                 line=dict(color=C["anodic"], width=1.8, dash="dash")))

    if ca.get("slope"):
        ca_hi = Ec - ca.get("lo", 0) + 0.03
        ca_lo = Ec - ca.get("hi", 0.15) - 0.05
        if curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
            ca_lo = max(ca_lo, reg.get("E_lim_end", ca_lo) - 0.02)
        E_ca = np.linspace(ca_lo, ca_hi, 100)
        y_ca = ca["slope"] * E_ca + ca["intercept"]
        fig.add_trace(go.Scatter(x=E_ca, y=y_ca, mode="lines",
                                 name=f"Cathodic Tafel  βc={ca['bc']*1000:.0f} mV/dec  R²={ca['r2']:.3f}",
                                 line=dict(color=C["cathodic"], width=1.8, dash="dash")))

    # ── Model curve (type-appropriate) ──────────────────
    if best.get("ba") and best.get("bc") and best.get("icorr"):
        try:
            E_model, i_model = _generate_model_curve(E_full, best, curve_type, reg)
            if i_model is not None:
                r2_lbl = ""
                if best.get("r2_log") is not None:
                    r2_lbl = f", R²(log)={best['r2_log']:.3f}"
                if best.get("r2") is not None:
                    r2_lbl += f", R²={best['r2']:.3f}"
                model_name = best.get("method", "Model").split("(")[0].strip()
                fig.add_trace(go.Scatter(
                    x=E_model, y=safe_log10(i_model), mode="lines",
                    name=f"{model_name}{r2_lbl}",
                    line=dict(color=C["bv"], width=2.5)))
        except:
            pass

    # ── icorr marker ────────────────────────────────────
    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic and ec:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(max(ic, 1e-20))], mode="markers",
            name=f"icorr = {ic:.3e} A/cm²",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))))

    # ── Limiting current line ───────────────────────────
    if reg.get("iL") and curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
        iL = reg["iL"]
        fig.add_hline(y=np.log10(iL), line=dict(color="#89dceb", width=1, dash="dot"),
                      annotation=dict(text=f"iL = {iL:.2e}", font=dict(color="#89dceb", size=10)))

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        title=dict(text="Potentiodynamic Polarization Curve",
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

    # Zoom window — tighter for passive systems
    if curve_type == CurveType.PASSIVE:
        zm_lo = Ec - 0.25
        zm_hi = min(Ec + 0.25, reg.get("E_ps", Ec + 0.25) + 0.05)
    else:
        zm_lo = Ec - 0.35
        zm_hi = Ec + 0.35

    zm = (E >= zm_lo) & (E <= zm_hi)
    E_z = E[zm]; y_z = log_i[zm]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E_z, y=y_z, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5)))

    # Fitting region highlight
    fit_mask = R.get("fit_mask")
    if fit_mask is not None:
        fm_zoom = fit_mask[zm] if len(fit_mask) == len(E) else None
        if fm_zoom is not None and fm_zoom.any():
            E_fm = E_z[fm_zoom]
            fig.add_vrect(x0=E_fm.min(), x1=E_fm.max(),
                          fillcolor=C["fit_band"], layer="below", line_width=0)

    E_fit = np.linspace(zm_lo, zm_hi, 300)
    if an.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=an["slope"]*E_fit+an["intercept"],
                                 mode="lines",
                                 name=f"Anodic βa={an['ba']*1000:.0f} mV/dec",
                                 line=dict(color=C["anodic"], width=2, dash="dash")))
    if ca.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=ca["slope"]*E_fit+ca["intercept"],
                                 mode="lines",
                                 name=f"Cathodic βc={ca['bc']*1000:.0f} mV/dec",
                                 line=dict(color=C["cathodic"], width=2, dash="dash")))

    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(max(ic, 1e-20))], mode="markers",
            name=f"Ecorr={ec:.4f}V, icorr={ic:.3e}",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))))

    if best.get("ba") and best.get("bc") and best.get("icorr"):
        try:
            _, i_model = _generate_model_curve(E_fit, best, curve_type, reg)
            if i_model is not None:
                fig.add_trace(go.Scatter(x=E_fit, y=safe_log10(i_model), mode="lines",
                                         name="Model Fit",
                                         line=dict(color=C["bv"], width=2.5)))
        except:
            pass

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


def plot_diagnostic(E, i, R, reg):
    log_i = safe_log10(i)
    log_sm = smooth(log_i, 13)
    d_log = np.gradient(log_sm, E)
    pas = R.get("passive", {})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04,
                        subplot_titles=("log₁₀|i|", "d(log₁₀|i|)/dE — region boundaries"))

    fig.add_trace(go.Scatter(x=E, y=log_i, mode="lines", name="log|i|",
                             line=dict(color=C["data"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=E, y=d_log, mode="lines", name="derivative",
                             line=dict(color="#fab387", width=1.5),
                             fill="tozeroy", fillcolor="rgba(250,179,135,0.08)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#585b70", width=1, dash="dot"), row=2)

    for row in (1, 2):
        fig.add_vline(x=reg["Ecorr"], line=dict(color="#f38ba8", width=1, dash="dot"), row=row)
        if pas.get("E_start"):
            fig.add_vrect(x0=pas["E_start"], x1=pas["E_end"],
                          fillcolor="rgba(166,227,161,0.09)",
                          layer="below", line_width=0, row=row)
        if reg.get("Eb"):
            fig.add_vline(x=reg["Eb"], line=dict(color="#f38ba8", width=1, dash="dash"), row=row)
        if reg.get("E_lim_start"):
            fig.add_vrect(x0=reg["E_lim_start"], x1=reg["E_lim_end"],
                          fillcolor="rgba(137,220,235,0.09)",
                          layer="below", line_width=0, row=row)

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        height=420, margin=dict(l=70, r=20, t=40, b=60), showlegend=False,
    )
    fig.update_yaxes(gridcolor=C["grid"], color=C["text"])
    fig.update_xaxes(gridcolor=C["grid"], color=C["text"], title_text="Potential (V)", row=2)
    return fig


def plot_residuals(E, i, R, reg, curve_type):
    """Residual plot to show fit quality."""
    best = R.get("best", {})
    fit_mask = R.get("fit_mask")

    if not (best.get("ba") and best.get("bc") and best.get("icorr")):
        return None
    if fit_mask is None or not fit_mask.any():
        return None

    E_f = E[fit_mask]
    i_f = i[fit_mask]

    E_dense = np.linspace(E_f.min(), E_f.max(), 300)
    _, i_model_dense = _generate_model_curve(E_dense, best, curve_type, reg)
    _, i_model_pts = _generate_model_curve(E_f, best, curve_type, reg)

    if i_model_pts is None:
        return None

    # Residuals in log space
    log_data = safe_log10(i_f)
    log_model = safe_log10(i_model_pts)
    residuals = log_data - log_model

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=("Fit overlay (fitting region)",
                                        "Residuals (log-space)"))

    # Overlay
    fig.add_trace(go.Scatter(x=E_f, y=log_data, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=E_dense, y=safe_log10(i_model_dense), mode="lines",
                             name="Model", line=dict(color=C["bv"], width=2.5)), row=1, col=1)

    # Residuals
    fig.add_trace(go.Scatter(x=E_f, y=residuals, mode="markers+lines",
                             name="Residual",
                             marker=dict(size=4, color="#fab387"),
                             line=dict(color="#fab387", width=1)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#585b70", width=1, dash="dot"), row=2)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    fig.add_annotation(text=f"RMSE(log) = {rmse:.4f}",
                       xref="paper", yref="paper", x=0.98, y=0.35,
                       showarrow=False, font=dict(color="#a6adc8", size=11),
                       bgcolor="rgba(24,24,37,0.8)", bordercolor=C["grid"])

    fig.update_layout(
        template="plotly_dark", plot_bgcolor=C["bg"], paper_bgcolor=C["paper"],
        height=420, margin=dict(l=70, r=20, t=40, b=60), showlegend=True,
        legend=dict(bgcolor="rgba(24,24,37,0.9)", bordercolor=C["grid"],
                    font=dict(color=C["text"], size=11)),
    )
    fig.update_yaxes(gridcolor=C["grid"], color=C["text"])
    fig.update_xaxes(gridcolor=C["grid"], color=C["text"], title_text="Potential (V)", row=2)
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
    color_map = {
        CurveType.ACTIVE_ONLY: "#f9e2af",
        CurveType.PASSIVE: "#a6e3a1",
        CurveType.DIFFUSION_LIMITED: "#89dceb",
        CurveType.MIXED: "#cba6f7",
    }
    clr = color_map.get(curve_type, "#cdd6f4")
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
    tp = R.get("tp", {})
    rp_r = R.get("rp", {})

    Ec = best.get("Ecorr") or R.get("Ecorr_tafel") or reg["Ecorr"]
    ic = best.get("icorr") or R.get("icorr_tafel")
    ba = best.get("ba") or an.get("ba")
    bc = best.get("bc") or ca.get("bc")
    B = best.get("B")
    Rp = rp_r.get("Rp")

    ic_sg = (B / Rp) if (B and Rp and Rp > 0) else None
    CR = ic * 3.27 * ew / rho if ic else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    # Region badges
    badges = []
    if reg.get("E_lim_start") is not None:
        badges.append('<span class="badge bb">Limiting current</span>')
    if pas.get("E_start") is not None:
        badges.append('<span class="badge bg">Passive region</span>')
    if reg.get("Eb") is not None:
        badges.append('<span class="badge br">Breakdown</span>')
    if tp:
        badges.append('<span class="badge by">Transpassive</span>')
    if reg.get("Epp") is not None:
        badges.append('<span class="badge bg">Epp / Flade</span>')
    if badges:
        st.markdown("**Detected regions:** " + "".join(badges), unsafe_allow_html=True)
        st.write("")

    # Show curve type banner
    show_curve_type_banner(curve_type)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="sechead">⚡ Corrosion</div>', unsafe_allow_html=True)
        pcard("Ecorr", Ec, "V vs Ref", "#f38ba8")
        pcard("icorr (Model fit)", ic, "A cm⁻²", "#fab387")
        if ic_sg:
            pcard("icorr (Stern-Geary)", ic_sg, "A cm⁻²", "#f9e2af")
        if CR:
            pcard("Corrosion rate", CR, "mm year⁻¹", "#eba0ac")
        if CR_sg:
            pcard("CR (Stern-Geary)", CR_sg, "mm year⁻¹", "#f5c2e7")

    with col2:
        st.markdown('<div class="sechead">📐 Kinetics</div>', unsafe_allow_html=True)
        pcard("βa  anodic Tafel slope", ba*1000 if ba else None, "mV decade⁻¹", "#a6e3a1")
        pcard("βc  cathodic Tafel slope", bc*1000 if bc else None, "mV decade⁻¹", "#94e2d5")
        if B:
            pcard("B  Stern-Geary const.", B*1000, "mV", "#89dceb")
        if Rp:
            pcard("Rp  polarization resist.", Rp, "Ω cm²", "#89b4fa")

        # Fit quality indicator
        m = best.get("method", "—")
        r2 = best.get("r2")
        r2_log = best.get("r2_log")
        info_parts = []
        if r2 is not None:
            info_parts.append(f"R² = {r2:.4f}")
        if r2_log is not None:
            info_parts.append(f"R²(log) = {r2_log:.4f}")
        info_str = ", ".join(info_parts)

        is_good = r2_log is not None and r2_log >= 0.95
        cls = "ok-box" if is_good else "warn-box"
        pfx = "✅" if is_good else "⚠️"
        st.markdown(f'<div class="{cls}">{pfx} {m}<br>{info_str}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sechead">🛡️ Passivity</div>', unsafe_allow_html=True)
        if curve_type in (CurveType.PASSIVE, CurveType.MIXED) and pas:
            pcard("ipass", pas["ipass"], "A cm⁻²", "#a6e3a1")
            pcard("Passive start", pas["E_start"], "V", "#94e2d5")
            pcard("Passive end", pas["E_end"], "V", "#94e2d5")
            pcard("Passive range", pas["range_V"]*1000, "mV", "#a6adc8")
        else:
            st.info("No passive region detected." if curve_type == CurveType.ACTIVE_ONLY
                    else "Passive parameters shown only for passive-type curves.")
        if reg.get("Epp") is not None:
            pcard("Epp / Flade potential", reg["Epp"], "V", "#a6e3a1")

    with col4:
        st.markdown('<div class="sechead">💥 Breakdown & Limit</div>', unsafe_allow_html=True)
        if reg.get("Eb"):
            pcard("Eb  breakdown potential", reg["Eb"], "V", "#f38ba8")
            if Ec:
                pcard("Pitting index Eb−Ecorr", (reg["Eb"]-Ec)*1000, "mV", "#f38ba8")
        if curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED) and reg.get("iL"):
            pcard("iL  limiting current", reg["iL"], "A cm⁻²", "#89dceb")
            e_range = f"{reg['E_lim_start']:.3f} → {reg['E_lim_end']:.3f}"
            pcard("iL  E range", e_range, "V", "#89b4fa")
            if best.get("iL"):
                pcard("iL  (fitted)", best["iL"], "A cm⁻²", "#89dceb")
        elif reg.get("iL"):
            pcard("iL  limiting current", reg["iL"], "A cm⁻²", "#89dceb")
        if tp:
            pcard("Transpassive log-slope", tp["slope"], "dec V⁻¹", "#fab387")


# ═══════════════════════════════════════════════════════════════
# BUILD RESULTS SUMMARY DATAFRAME
# ═══════════════════════════════════════════════════════════════

def build_summary_df(R, reg, ew, rho, curve_type):
    best = R.get("best", {})
    an = R.get("an", {})
    ca = R.get("ca", {})
    pas = R.get("passive", {})
    tp = R.get("tp", {})
    rp_r = R.get("rp", {})

    Ec = best.get("Ecorr") or R.get("Ecorr_tafel") or reg["Ecorr"]
    ic = best.get("icorr") or R.get("icorr_tafel")
    ba = best.get("ba") or an.get("ba")
    bc = best.get("bc") or ca.get("bc")
    B = best.get("B")
    Rp = rp_r.get("Rp")
    CR = ic * 3.27 * ew / rho if ic else None
    ic_sg = B/Rp if (B and Rp) else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    rows = [
        ("Curve Type", CurveType.DESCRIPTIONS[curve_type][0]),
        ("Ecorr (V)", Ec),
        ("icorr (A/cm²)", ic),
        ("icorr_SG (A/cm²)", ic_sg),
        ("βa (mV/dec)", ba*1000 if ba else None),
        ("βc (mV/dec)", bc*1000 if bc else None),
        ("B Stern-Geary (mV)", B*1000 if B else None),
        ("Rp (Ω·cm²)", Rp),
        ("Corrosion Rate (mm/yr)", CR),
        ("CR Stern-Geary (mm/yr)", CR_sg),
    ]

    # Only include type-specific parameters
    if curve_type in (CurveType.PASSIVE, CurveType.MIXED):
        rows.extend([
            ("ipass (A/cm²)", pas.get("ipass")),
            ("E_passive_start (V)", pas.get("E_start")),
            ("E_passive_end (V)", pas.get("E_end")),
            ("Passive range (mV)", pas.get("range_V", 0)*1000 if pas else None),
            ("Eb breakdown (V)", reg.get("Eb")),
            ("Pitting index Eb−Ecorr (mV)",
             (reg["Eb"]-Ec)*1000 if reg.get("Eb") and Ec else None),
            ("Epp Flade (V)", reg.get("Epp")),
        ])

    if curve_type in (CurveType.DIFFUSION_LIMITED, CurveType.MIXED):
        rows.extend([
            ("iL limiting (A/cm²)", reg.get("iL")),
            ("iL fitted (A/cm²)", best.get("iL")),
        ])

    if tp and curve_type in (CurveType.PASSIVE, CurveType.MIXED):
        rows.append(("Transpassive slope (dec/V)", tp.get("slope")))

    rows.extend([
        ("Fitting method", best.get("method")),
        ("R² (linear)", best.get("r2")),
        ("R² (log-space)", best.get("r2_log")),
    ])

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ═══════════════════════════════════════════════════════════════
# MATERIAL DATABASE
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

def run_demo():
    np.random.seed(42)
    choice = st.session_state.get("demo", "")

    if "passive" in choice.lower() and "breakdown" in choice.lower():
        # Full curve: limiting + active + passive + breakdown
        E = np.linspace(-0.65, 0.85, 500)
        i = 2e-6*(np.exp(2.303*(E+0.38)/0.065) - np.exp(-2.303*(E+0.38)/0.110))
        # Add diffusion-limited cathodic
        i_c = -2e-6*np.exp(-2.303*(E+0.38)/0.110)
        iL = 5e-4
        i_c_mixed = i_c / (1 + np.abs(i_c)/iL)
        i = 2e-6*np.exp(2.303*(E+0.38)/0.065) + i_c_mixed
        # Add passive plateau
        i += 3e-6/(1+np.exp(-25*(E+0.25)))
        # Breakdown
        m = E > 0.55
        i[m] += 3e-6*np.exp(12*(E[m]-0.55))
    elif "active" in choice.lower():
        E = np.linspace(-0.65, 0.45, 400)
        i = 5e-6*(np.exp(2.303*(E+0.45)/0.06) - np.exp(-2.303*(E+0.45)/0.12))
    elif "diffusion" in choice.lower():
        E = np.linspace(-0.80, 0.30, 400)
        i_a = 3e-6*np.exp(2.303*(E+0.40)/0.070)
        i_c_kin = 3e-6*np.exp(-2.303*(E+0.40)/0.120)
        iL = 2e-4
        i_c = i_c_kin / (1 + i_c_kin/iL)
        i = i_a - i_c
    else:
        # Passive only
        E = np.linspace(-0.55, 0.65, 400)
        i = 2e-6*(np.exp(2.303*(E+0.40)/0.065) - np.exp(-2.303*(E+0.40)/0.110))
        i += 3e-6/(1+np.exp(-20*(E+0.20)))

    noise = np.random.normal(0, np.abs(i)*0.03 + 3e-9, size=len(i))
    i += noise
    return E, i


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def process_and_display(E_raw, i_dens, area, ew, rho, source_label=""):
    """Core processing pipeline — used by both upload and demo modes."""

    # ── Full auto pipeline ──────────────────────────────
    with st.spinner("Detecting regions…"):
        reg = detect_regions(E_raw, i_dens)

    curve_type = classify_curve(E_raw, i_dens, reg)

    with st.spinner(f"Fitting ({CurveType.DESCRIPTIONS[curve_type][0]})…"):
        fitter = AdaptiveFitter(E_raw, i_dens, reg)
        R = fitter.run()

    # ═══════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════
    st.markdown("---")

    fig_full = plot_polarization(E_raw, i_dens, R, reg, curve_type)
    st.plotly_chart(fig_full, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig_zoom = plot_active_zoom(E_raw, i_dens, R, reg, curve_type)
        st.plotly_chart(fig_zoom, use_container_width=True)
    with col_right:
        fig_resid = plot_residuals(E_raw, i_dens, R, reg, curve_type)
        if fig_resid:
            st.plotly_chart(fig_resid, use_container_width=True)
        else:
            fig_diag = plot_diagnostic(E_raw, i_dens, R, reg)
            st.plotly_chart(fig_diag, use_container_width=True)

    # Diagnostic always available in expander
    with st.expander("📊 Derivative diagnostic plot"):
        fig_diag = plot_diagnostic(E_raw, i_dens, R, reg)
        st.plotly_chart(fig_diag, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # PARAMETERS
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    show_parameters(R, reg, ew, rho, curve_type)

    # ═══════════════════════════════════════════════════
    # FITTING LOG + DOWNLOAD
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    c_log, c_dl = st.columns([3, 1])

    with c_log:
        with st.expander("🪵 Fitting log (detailed)"):
            for msg in fitter.log:
                st.markdown(f"- {msg}")

    with c_dl:
        df_sum = build_summary_df(R, reg, ew, rho, curve_type)
        st.download_button(
            "⬇️ Download results (CSV)",
            df_sum.to_csv(index=False).encode(),
            "tafel_results.csv", "text/csv",
            use_container_width=True)
        df_proc = pd.DataFrame({
            "E_V": E_raw,
            "i_Acm2": i_dens,
            "log_abs_i": safe_log10(i_dens),
        })
        st.download_button(
            "⬇️ Download processed data (CSV)",
            df_proc.to_csv(index=False).encode(),
            "tafel_data.csv", "text/csv",
            use_container_width=True)


def main():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                border:1px solid #313244; border-radius:12px;
                padding:20px 28px; margin-bottom:20px;">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting Tool v2</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">
        Smart curve classification · Type-specific models · Adaptive optimization
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.caption("Only electrode area and material cannot be auto-detected.")

        area = st.number_input(
            "Electrode area (cm²)",
            min_value=0.001, max_value=100.0, value=1.0, step=0.01,
            help="Raw current ÷ area = current density. Default 1 cm².")
        mat = st.selectbox("Material (for corrosion rate)", list(MATERIALS.keys()))
        ew0, rho0 = MATERIALS[mat]
        if mat == "Custom":
            ew = st.number_input("Equivalent weight (g/eq)", 1.0, 300.0, ew0)
            rho = st.number_input("Density (g/cm³)", 0.5, 25.0, rho0)
        else:
            ew, rho = ew0, rho0

        st.divider()
        st.markdown("**Curve types auto-detected:**")
        st.markdown("""
        <div style="font-size:11px; color:#a6adc8; line-height:1.8;">
        🟡 <b>Active/Tafel</b> — Standard BV<br>
        🟢 <b>Passive</b> — BV in active region only<br>
        🔵 <b>Diffusion-limited</b> — BV + iL correction<br>
        🟣 <b>Mixed</b> — Both passive + diffusion
        </div>""", unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your file here — CSV, TXT, XLSX, XLS",
        type=["csv", "txt", "xlsx", "xls"],
        label_visibility="visible")

    if uploaded is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            demo_choice = st.selectbox(
                "Or try a built-in demo",
                ["Full curve (limiting + passive + breakdown)",
                 "Active/Tafel only",
                 "With passive region only",
                 "Diffusion-limited cathodic"])
            if st.button("▶  Run demo", use_container_width=True, type="primary"):
                st.session_state["demo"] = demo_choice
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #313244;border-radius:10px;
                        padding:16px 20px;">
            <b style="color:#89b4fa;">What's new in v2</b><br><br>
            <span style="color:#a6adc8;font-size:13px;">
            ✓ <b>Automatic curve type classification</b><br>
            ✓ Type-specific electrochemical models<br>
            ✓ Smart fitting window (avoids passive & diffusion regions)<br>
            ✓ Multi-stage optimization: TRF → DE → Nelder-Mead<br>
            ✓ Log-space + linear-space dual fitting<br>
            ✓ Residual plots for fit quality assessment<br>
            ✓ Only shows parameters relevant to the detected curve type<br>
            ✓ BV+diffusion model for mass-transport limited cathodic<br>
            ✓ Tafel lines drawn only in valid regions (not extrapolated everywhere)
            </span>
            </div>""", unsafe_allow_html=True)
        return

    # ── Process uploaded file ───────────────────────────
    with st.spinner("Reading file…"):
        try:
            df = load_any_file(uploaded)
        except Exception as ex:
            st.error(f"Could not read file: {ex}")
            return

    with st.spinner("Detecting columns and units…"):
        try:
            e_col, i_col, i_factor = auto_detect_columns(df)
        except Exception as ex:
            st.error(f"Column detection failed: {ex}")
            return

    with st.expander(f"📋 Auto-detected: **{e_col}** (potential) · **{i_col}** (current)", expanded=False):
        st.dataframe(df[[e_col, i_col]].head(10), use_container_width=True)
        st.caption(f"File: {uploaded.name} · {df.shape[0]} rows · "
                   f"Current factor: {i_factor} A per file unit")

    E_raw = df[e_col].values.astype(float)
    i_raw = df[i_col].values.astype(float) * i_factor

    ok = np.isfinite(E_raw) & np.isfinite(i_raw)
    E_raw, i_raw = E_raw[ok], i_raw[ok]
    idx = np.argsort(E_raw)
    E_raw, i_raw = E_raw[idx], i_raw[idx]

    i_dens = i_raw / area

    process_and_display(E_raw, i_dens, area, ew, rho, source_label=uploaded.name)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "demo" in st.session_state:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                    border:1px solid #313244; border-radius:12px;
                    padding:20px 28px; margin-bottom:20px;">
          <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting Tool v2</h1>
          <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">Demo mode</p>
        </div>""", unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            area = st.number_input("Electrode area (cm²)", 0.001, 100.0, 1.0)
            mat = st.selectbox("Material", list(MATERIALS.keys()))
            ew, rho = MATERIALS[mat]
            if mat == "Custom":
                ew = st.number_input("EW (g/eq)", 1.0, 300.0, ew)
                rho = st.number_input("ρ (g/cm³)", 0.5, 25.0, rho)
            if st.button("← Back to upload"):
                del st.session_state["demo"]; st.rerun()

        E_d, i_d = run_demo()
        i_dens = i_d / area

        st.info(f"🧪 Demo: **{st.session_state['demo']}**")
        process_and_display(E_d, i_dens, area, ew, rho,
                            source_label=f"Demo: {st.session_state['demo']}")
    else:
        main()
