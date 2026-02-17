"""
Tafel Fitting Tool — Fully Automatic
Upload → instant region detection → fitting → plots + parameters
No manual intervention required.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter
from scipy.stats import linregress
from itertools import groupby
import warnings, io, re

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(page_title="Tafel Fitting", page_icon="⚡", layout="wide")

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"]  { background:#13131f; color:#cdd6f4; }
[data-testid="stSidebar"]                 { background:#1a1a2e; }
section[data-testid="stFileUploadDropzone"] { background:#1e1e2e !important; border:2px dashed #45475a !important; border-radius:12px !important; }
.pcard { background:#1e1e2e; border:1px solid #313244; border-radius:10px;
         padding:14px 16px; margin:4px 0; }
.plabel { color:#a6adc8; font-size:10px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; }
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
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════
C = dict(
    data="#89b4fa", anodic="#f9e2af", cathodic="#cba6f7", bv="#a6e3a1",
    passive="rgba(166,227,161,0.12)", limiting="rgba(137,220,235,0.12)",
    transpassive="rgba(243,188,168,0.10)", ecorr="#f38ba8",
    grid="#313244", bg="#1e1e2e", paper="#181825", text="#cdd6f4"
)

# ═══════════════════════════════════════════════
# COLUMN AUTO-DETECTION
# ═══════════════════════════════════════════════
# Patterns: (potential_keywords, current_keywords, current_unit)
COLUMN_SIGNATURES = [
    # Autolab NOVA
    (r"we.*potential", r"we.*current", "A"),
    # BioLogic
    (r"ewe",           r"i/ma",        "mA"),
    (r"ewe",           r"<i>/ma",      "mA"),
    # Gamry
    (r"^vf$",          r"^im$",        "A"),
    # CHI
    (r"potential/v",   r"current/a",   "A"),
    (r"e/v",           r"i/a",         "A"),
    # Generic: any column with "potential" or "volt" and "current" or "amps"
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
    """Return (E_col, i_col, i_factor_to_A) or raise ValueError."""
    cols_lower = {c: c.lower().strip() for c in df.columns}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for E_pat, I_pat, unit in COLUMN_SIGNATURES:
        e_match = [c for c, cl in cols_lower.items()
                   if re.search(E_pat, cl) and c in numeric]
        i_match = [c for c, cl in cols_lower.items()
                   if re.search(I_pat, cl) and c in numeric and c not in e_match]
        if e_match and i_match:
            # pick the best E column (prefer "WE" or "measured" over "applied")
            e_col = sorted(e_match,
                           key=lambda c: 0 if "we" in c.lower() or "meas" in c.lower() else 1)[0]
            i_col = i_match[0]
            # infer unit factor from column name
            factor = 1e-3 if unit == "mA" else 1.0
            for pat, f in UNIT_HINTS.items():
                if re.search(pat, cols_lower[i_col]):
                    factor = f; break
            return e_col, i_col, factor

    # Last resort: take first two numeric columns
    if len(numeric) >= 2:
        return numeric[0], numeric[1], 1.0

    raise ValueError("Could not find potential and current columns automatically.")


def detect_file_skiprows(raw_bytes, ext):
    """Skip comment/header rows (lines starting with # or non-numeric)."""
    if ext in (".xlsx", ".xls"):
        return 0
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        parts = re.split(r"[,;\t ]+", line.strip())
        numeric_count = sum(1 for p in parts if re.match(r"^-?[\d.eE+]+$", p))
        if numeric_count >= 2:
            # check if prev line looked like a header
            return max(0, i - 1) if i > 0 and not any(
                re.match(r"^-?[\d.eE+]+$", p)
                for p in re.split(r"[,;\t ]+", lines[i-1].strip())) else i
    return 0


def load_any_file(f):
    """Load CSV/TXT/XLSX/XLS into DataFrame, auto-skipping headers."""
    name = f.name.lower()
    ext  = next((e for e in (".xlsx", ".xls", ".csv", ".txt") if name.endswith(e)), ".csv")
    raw  = f.read(); f.seek(0)

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(raw))
        return df

    skip = detect_file_skiprows(raw, ext)
    text = raw.decode("utf-8", errors="replace")
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5:
                # drop all-NaN columns
                df = df.dropna(axis=1, how="all")
                return df
        except Exception:
            pass
    raise ValueError(f"Unable to parse {f.name}")


# ═══════════════════════════════════════════════
# MATH HELPERS
# ═══════════════════════════════════════════════

def safe_log10(x):
    return np.log10(np.maximum(np.abs(x), 1e-20))

def smooth(y, window=11, poly=3):
    n = len(y)
    w = min(window, n if n % 2 == 1 else n - 1)
    return savgol_filter(y, w, poly) if w >= 5 else y.copy()

def butler_volmer(E, Ecorr, icorr, ba, bc):
    eta = E - Ecorr
    return icorr * (np.exp(2.303 * eta / ba) - np.exp(-2.303 * eta / bc))


# ═══════════════════════════════════════════════
# REGION DETECTION
# ═══════════════════════════════════════════════

def detect_regions(E, i):
    reg = {}
    n   = len(E)

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

    # ── Cathodic (E < Ecorr) ─────────────────────────────
    ci = np.where(E < Ec)[0]
    if len(ci) >= 6:
        log_c = smooth(safe_log10(i[ci]), min(9, (len(ci)//2)*2-1 or 5))
        dlog  = np.abs(np.gradient(log_c, E[ci]))
        flat  = dlog < np.percentile(dlog, 25)
        runs  = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            # longest flat run = limiting current plateau
            best = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best[1]]
            reg.update(
                limit_idx   = ci[idxs],
                iL          = float(np.median(np.abs(i[ci[idxs]]))),
                E_lim_start = float(E[ci[idxs[0]]]),
                E_lim_end   = float(E[ci[idxs[-1]]]),
            )

    # ── Anodic (E > Ecorr) ───────────────────────────────
    ai = np.where(E > Ec)[0]
    if len(ai) >= 6:
        log_a = smooth(safe_log10(i[ai]), min(11, (len(ai)//2)*2-1 or 5))
        dlog  = np.abs(np.gradient(log_a, E[ai]))

        # Passive region = longest low-derivative segment
        flat  = dlog < np.percentile(dlog, 35)
        runs  = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            best = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in best[1]]
            if len(idxs) >= 4:
                ps, pe = ai[idxs[0]], ai[idxs[-1]]
                reg.update(
                    passive_s = int(ps), passive_e = int(pe),
                    E_ps      = float(E[ps]), E_pe = float(E[pe]),
                    ipass     = float(np.median(np.abs(i[ps:pe+1]))),
                    Epp       = float(E[ps]),
                )

                # Breakdown = sharp log|i| rise after passive end
                if pe + 3 < n:
                    d_after = np.gradient(safe_log10(np.abs(i[pe:])), E[pe:])
                    # threshold: mean + 2σ of derivative across whole anodic
                    thr = np.mean(dlog) + 2.5 * np.std(dlog)
                    jump = np.where(np.abs(d_after) > thr)[0]
                    if len(jump):
                        eb_idx = pe + jump[0]
                        reg["Eb_idx"] = int(eb_idx)
                        reg["Eb"]     = float(E[eb_idx])
                    # Transpassive starts at Eb (or passive end if no Eb)
                    reg["tp_idx"] = reg.get("Eb_idx", pe)

    return reg


# ═══════════════════════════════════════════════
# FITTING ENGINE
# ═══════════════════════════════════════════════

class AutoFitter:
    """Fully automatic multi-strategy Tafel fitter with fallback chain."""

    def __init__(self, E, i, reg):
        self.E   = E
        self.i   = i
        self.reg = reg
        self.Ec  = reg["Ecorr"]
        self.ic0 = max(float(np.abs(i[reg["ecorr_idx"]])), 1e-12)
        self.R   = {}      # results dict
        self.log = []      # fitting messages

    # ── helpers ────────────────────────────────────────────
    def _bv_mask(self, half_width=0.30):
        E, Ec = self.E, self.Ec
        ps = self.reg.get("E_ps")
        m = (E > Ec - half_width) & (E < Ec + half_width)
        if ps is not None:
            m = m & (E < ps - 0.005)
        return m if m.sum() >= 6 else np.ones(len(E), bool) & (
            (E < ps - 0.005) if ps is not None else True)

    # ── 1. Auto anodic Tafel window search ─────────────────
    def fit_anodic_tafel(self):
        E, i, Ec = self.E, self.i, self.Ec
        log_i = safe_log10(i)
        ps    = self.reg.get("E_ps")

        best = None
        # Grid search over (lo, hi) in V above Ecorr
        for lo in np.arange(0.005, 0.05, 0.005):
            for hi in np.arange(lo + 0.02, lo + 0.18, 0.01):
                m = (E > Ec + lo) & (E < Ec + hi)
                if ps is not None:
                    m = m & (E < ps - 0.005)
                if m.sum() < 4:
                    continue
                s, b, r, *_ = linregress(E[m], log_i[m])
                ba_mV = (1/s)*1000 if s > 0 else None
                if s > 0 and 20 < ba_mV < 600 and r**2 > 0.85:
                    if best is None or r**2 > best["r2"]:
                        best = dict(slope=s, intercept=b, r2=r**2, ba=1/s,
                                    mask=m, lo=lo, hi=hi, n=m.sum())

        if best:
            self.R["an"] = best
            self.log.append(
                f"✅ Anodic Tafel: βa = {best['ba']*1000:.1f} mV/dec, "
                f"R² = {best['r2']:.4f}, "
                f"window [{Ec+best['lo']:.3f}–{Ec+best['hi']:.3f} V] ({best['n']} pts)"
            )
        else:
            self.log.append(
                "⚠️ No clean anodic Tafel region — material likely passivates "
                "immediately after Ecorr (stainless steel, Ti, Al). "
                "βa will come from BV/DE fit."
            )

    # ── 2. Auto cathodic Tafel window search ───────────────
    def fit_cathodic_tafel(self):
        E, i, Ec = self.E, self.i, self.Ec
        log_i = safe_log10(i)
        lim_end = self.reg.get("E_lim_end")  # avoid limiting region

        best = None
        for lo in np.arange(0.01, 0.10, 0.005):
            for hi in np.arange(lo + 0.02, lo + 0.25, 0.01):
                m = (E < Ec - lo) & (E > Ec - hi)
                if lim_end is not None:
                    m = m & (E > lim_end + 0.005)
                if m.sum() < 4:
                    continue
                s, b, r, *_ = linregress(E[m], log_i[m])
                bc_mV = (-1/s)*1000 if s < 0 else None
                if s < 0 and 20 < bc_mV < 600 and r**2 > 0.85:
                    if best is None or r**2 > best["r2"]:
                        best = dict(slope=s, intercept=b, r2=r**2, bc=-1/s,
                                    mask=m, lo=lo, hi=hi, n=m.sum())

        if best:
            self.R["ca"] = best
            self.log.append(
                f"✅ Cathodic Tafel: βc = {best['bc']*1000:.1f} mV/dec, "
                f"R² = {best['r2']:.4f}, "
                f"window [{Ec-best['hi']:.3f}–{Ec-best['lo']:.3f} V] ({best['n']} pts)"
            )
        else:
            self.log.append("⚠️ No clean cathodic Tafel region found.")

    # ── 3. Ecorr / icorr from Tafel line intersection ──────
    def fit_intersection(self):
        an, ca = self.R.get("an"), self.R.get("ca")
        if an and ca:
            ds = an["slope"] - ca["slope"]
            if abs(ds) > 1e-10:
                E_i   = (ca["intercept"] - an["intercept"]) / ds
                logi  = an["slope"] * E_i + an["intercept"]
                self.R["Ecorr_tafel"]  = float(E_i)
                self.R["icorr_tafel"]  = float(10**logi)
                self.log.append(
                    f"✅ Tafel intersection: Ecorr = {E_i:.4f} V, "
                    f"icorr = {10**logi:.3e} A/cm²"
                )

    # ── 4. Butler-Volmer curve_fit ──────────────────────────
    def fit_bv(self):
        E, i = self.E, self.i
        m    = self._bv_mask()
        E_f, i_f = E[m], i[m]

        ba0 = self.R.get("an",  {}).get("ba",  0.06)
        bc0 = self.R.get("ca",  {}).get("bc",  0.12)
        p0  = [self.Ec, self.ic0, ba0, bc0]
        bnd = ([E.min(), 1e-14, 0.005, 0.005],
               [E.max(), 1e-2,  0.8,   0.8])
        try:
            popt, pcov = curve_fit(butler_volmer, E_f, i_f, p0=p0, bounds=bnd,
                                   maxfev=30000, method="trf", ftol=1e-12, xtol=1e-12)
            pred = butler_volmer(E_f, *popt)
            r2   = _r2(i_f, pred)
            self.R["bv"] = dict(
                Ecorr=float(popt[0]), icorr=float(popt[1]),
                ba=float(popt[2]), bc=float(popt[3]),
                r2=r2, method="Butler-Volmer (TRF)", success=True
            )
            self.log.append(
                f"✅ BV fit: Ecorr = {popt[0]:.4f} V, icorr = {popt[1]:.3e}, "
                f"R² = {r2:.4f}"
            )
        except Exception as ex:
            self.R["bv"] = {"success": False}
            self.log.append(f"⚠️ BV TRF failed ({ex}) → trying DE")
            self.fit_de()

    # ── 5. Differential Evolution (global fallback) ─────────
    def fit_de(self):
        bv = self.R.get("bv", {})
        if bv.get("success") and bv.get("r2", 0) >= 0.88:
            return  # BV is good enough
        E, i = self.E, self.i
        m    = self._bv_mask(0.40)
        E_f, i_f = E[m], i[m]
        ic0 = max(self.ic0, 1e-13)

        def obj(p):
            try:
                pred = butler_volmer(E_f, p[0], 10**p[1], p[2], p[3])
                return float(np.sum(((i_f - pred) / (np.abs(i_f) + 1e-12))**2))
            except:
                return 1e30

        bnd = [
            (self.Ec - 0.3, self.Ec + 0.3),
            (np.log10(ic0 * 0.0001), np.log10(ic0 * 10000)),
            (0.01, 0.7), (0.01, 0.7),
        ]
        try:
            res = differential_evolution(obj, bnd, seed=42, maxiter=2000,
                                          tol=1e-10, popsize=25, workers=1)
            p    = res.x
            pred = butler_volmer(E_f, p[0], 10**p[1], p[2], p[3])
            r2   = _r2(i_f, pred)
            self.R["de"] = dict(
                Ecorr=float(p[0]), icorr=float(10**p[1]),
                ba=float(p[2]), bc=float(p[3]),
                r2=r2, method="Butler-Volmer (DE global)", success=True, fallback=True
            )
            self.log.append(
                f"✅ DE fallback: Ecorr = {p[0]:.4f} V, icorr = {10**p[1]:.3e}, "
                f"R² = {r2:.4f}"
            )
        except Exception as ex:
            self.log.append(f"⚠️ DE also failed: {ex}")

    # ── 6. Polarization resistance ──────────────────────────
    def fit_rp(self):
        E, i, Ec = self.E, self.i, self.Ec
        for dE in [0.015, 0.025, 0.05, 0.10]:
            m = np.abs(E - Ec) < dE
            if m.sum() >= 4:
                s, _, r, *_ = linregress(E[m], i[m])
                if abs(s) > 1e-20:
                    self.R["rp"] = dict(Rp=float(1/s), r2=float(r**2), dE=dE)
                    self.log.append(
                        f"✅ Rp = {1/s:.3e} Ω·cm² "
                        f"(±{dE*1000:.0f} mV window, R² = {r**2:.3f})"
                    )
                    return

    # ── 7. Passive & transpassive ───────────────────────────
    def fit_passive(self):
        E, i, reg = self.E, self.i, self.reg
        ps, pe = reg.get("passive_s"), reg.get("passive_e")
        if ps is not None and pe is not None:
            iabs = np.abs(i[ps:pe+1])
            self.R["passive"] = dict(
                ipass=float(np.median(iabs)),
                E_start=float(E[ps]), E_end=float(E[pe]),
                range_V=float(E[pe] - E[ps])
            )
        tp = reg.get("tp_idx")
        if tp is not None and tp + 4 < len(E):
            s, _, r, *_ = linregress(E[tp:], safe_log10(np.abs(i[tp:])))
            self.R["tp"] = dict(slope=float(s), r2=float(r**2), E_start=float(E[tp]))

    # ── Master run ──────────────────────────────────────────
    def run(self):
        self.fit_anodic_tafel()
        self.fit_cathodic_tafel()
        self.fit_intersection()
        self.fit_bv()
        self.fit_de()
        self.fit_rp()
        self.fit_passive()
        self._resolve_best()
        return self.R

    def _resolve_best(self):
        bv = self.R.get("bv", {})
        de = self.R.get("de", {})
        if bv.get("success") and bv.get("r2", 0) >= de.get("r2", 0):
            best = bv
        elif de.get("success"):
            best = de
        else:
            # Pure Tafel fallback
            ec = self.R.get("Ecorr_tafel", self.Ec)
            ic = self.R.get("icorr_tafel", self.ic0)
            best = dict(
                Ecorr=ec, icorr=ic,
                ba=self.R.get("an", {}).get("ba"),
                bc=self.R.get("ca", {}).get("bc"),
                r2=None, method="Tafel Extrapolation only", success=False
            )
        # Stern-Geary constant
        ba, bc = best.get("ba"), best.get("bc")
        if ba and bc and ba > 0 and bc > 0:
            best["B"] = (ba * bc) / (2.303 * (ba + bc))
        self.R["best"] = best


def _r2(y_true, y_pred):
    ss_r = np.sum((y_true - y_pred)**2)
    ss_t = np.sum((y_true - y_true.mean())**2)
    return float(max(0, 1 - ss_r / ss_t)) if ss_t > 0 else 0.0


# ═══════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════

def plot_polarization(E, i, R, reg):
    an     = R.get("an", {})
    ca     = R.get("ca", {})
    best   = R.get("best", {})
    pas    = R.get("passive", {})
    tp     = R.get("tp", {})
    log_i  = safe_log10(i)

    fig = go.Figure()

    # ── Region fills ─────────────────────────────────────
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

    # ── Key vertical lines ────────────────────────────────
    Ec = reg["Ecorr"]
    fig.add_vline(x=Ec, line=dict(color="#f38ba8", width=1.5, dash="dot"),
                  annotation=dict(text="Ecorr", font=dict(color="#f38ba8", size=10), yanchor="bottom"))
    if reg.get("Eb"):
        fig.add_vline(x=reg["Eb"], line=dict(color="#f38ba8", width=1.5, dash="dash"),
                      annotation=dict(text="Eb", font=dict(color="#f38ba8", size=10), yanchor="top"))
    if reg.get("Epp"):
        fig.add_vline(x=reg["Epp"], line=dict(color="#a6e3a1", width=1, dash="dot"),
                      annotation=dict(text="Epp", font=dict(color="#a6e3a1", size=10), yanchor="bottom"))

    # ── Measured data ─────────────────────────────────────
    fig.add_trace(go.Scatter(x=E, y=log_i, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5),
                             hovertemplate="E = %{x:.4f} V<br>log|i| = %{y:.3f}<extra></extra>"))

    # ── Tafel lines (extrapolated across full range) ───────
    E_full = np.linspace(E.min(), E.max(), 300)
    if an.get("slope"):
        y_an = an["slope"] * E_full + an["intercept"]
        fig.add_trace(go.Scatter(x=E_full, y=y_an, mode="lines",
                                 name=f"Anodic Tafel  βa={an['ba']*1000:.0f} mV/dec",
                                 line=dict(color=C["anodic"], width=1.8, dash="dash")))
    if ca.get("slope"):
        y_ca = ca["slope"] * E_full + ca["intercept"]
        fig.add_trace(go.Scatter(x=E_full, y=y_ca, mode="lines",
                                 name=f"Cathodic Tafel  βc={ca['bc']*1000:.0f} mV/dec",
                                 line=dict(color=C["cathodic"], width=1.8, dash="dash")))

    # ── BV fitted curve ───────────────────────────────────
    if best.get("ba") and best.get("bc") and best.get("icorr"):
        try:
            i_bv = butler_volmer(E_full, best["Ecorr"], best["icorr"],
                                  best["ba"], best["bc"])
            r2_lbl = f", R²={best['r2']:.3f}" if best.get("r2") else ""
            fig.add_trace(go.Scatter(x=E_full, y=safe_log10(i_bv), mode="lines",
                                     name=f"BV Fit{r2_lbl}",
                                     line=dict(color=C["bv"], width=2.5)))
        except:
            pass

    # ── icorr marker ─────────────────────────────────────
    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic and ec:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(ic)], mode="markers",
            name=f"icorr = {ic:.3e} A/cm²",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))
        ))

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
        height=500, margin=dict(l=70, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def plot_diagnostic(E, i, R, reg):
    """2-panel: log|i| + d(log|i|)/dE to show region boundaries clearly."""
    log_i   = safe_log10(i)
    log_sm  = smooth(log_i, 13)
    d_log   = np.gradient(log_sm, E)
    pas     = R.get("passive", {})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04,
                        subplot_titles=("log₁₀|i|", "d(log₁₀|i|)/dE  — region boundaries"))

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


def plot_active_zoom(E, i, R, reg):
    """Zoom on the active/Tafel region with fitted lines."""
    an   = R.get("an", {})
    ca   = R.get("ca", {})
    best = R.get("best", {})
    Ec   = reg["Ecorr"]
    log_i = safe_log10(i)

    # Zoom window: ±0.4 V around Ecorr
    zm  = (E >= Ec - 0.40) & (E <= Ec + 0.40)
    E_z = E[zm]; y_z = log_i[zm]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E_z, y=y_z, mode="lines", name="Measured",
                             line=dict(color=C["data"], width=2.5)))

    E_fit = np.linspace(Ec - 0.35, Ec + 0.35, 300)
    if an.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=an["slope"]*E_fit+an["intercept"],
                                 mode="lines",
                                 name=f"Anodic Tafel  βa={an['ba']*1000:.0f} mV/dec",
                                 line=dict(color=C["anodic"], width=2, dash="dash")))
    if ca.get("slope"):
        fig.add_trace(go.Scatter(x=E_fit, y=ca["slope"]*E_fit+ca["intercept"],
                                 mode="lines",
                                 name=f"Cathodic Tafel  βc={ca['bc']*1000:.0f} mV/dec",
                                 line=dict(color=C["cathodic"], width=2, dash="dash")))

    ic = best.get("icorr") or R.get("icorr_tafel")
    ec = best.get("Ecorr") or Ec
    if ic:
        fig.add_trace(go.Scatter(
            x=[ec], y=[np.log10(ic)], mode="markers",
            name=f"Ecorr = {ec:.4f} V\nicorr = {ic:.3e} A/cm²",
            marker=dict(symbol="x-thin", size=18, color="#f38ba8",
                        line=dict(width=4, color="#f38ba8"))))

    if best.get("ba") and best.get("bc") and best.get("icorr"):
        try:
            i_bv = butler_volmer(E_fit, best["Ecorr"], best["icorr"],
                                  best["ba"], best["bc"])
            fig.add_trace(go.Scatter(x=E_fit, y=safe_log10(i_bv), mode="lines",
                                     name="BV Fit",
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


# ═══════════════════════════════════════════════
# PARAMETER CARDS
# ═══════════════════════════════════════════════

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


def show_parameters(R, reg, ew, rho):
    best = R.get("best", {})
    an   = R.get("an",   {})
    ca   = R.get("ca",   {})
    pas  = R.get("passive", {})
    tp   = R.get("tp",   {})
    rp_r = R.get("rp",   {})

    Ec   = best.get("Ecorr") or R.get("Ecorr_tafel") or reg["Ecorr"]
    ic   = best.get("icorr") or R.get("icorr_tafel")
    ba   = best.get("ba") or an.get("ba")
    bc   = best.get("bc") or ca.get("bc")
    B    = best.get("B")
    Rp   = rp_r.get("Rp")

    ic_sg = (B / Rp) if (B and Rp and Rp > 0) else None
    CR    = ic  * 3.27 * ew / rho if ic   else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    # Region badges
    badges = []
    if reg.get("E_lim_start") is not None: badges.append('<span class="badge bb">Limiting current</span>')
    if pas.get("E_start")     is not None: badges.append('<span class="badge bg">Passive region</span>')
    if reg.get("Eb")          is not None: badges.append('<span class="badge br">Breakdown</span>')
    if tp:                                  badges.append('<span class="badge by">Transpassive</span>')
    if reg.get("Epp")         is not None: badges.append('<span class="badge bg">Epp / Flade</span>')
    if badges:
        st.markdown("**Detected regions:** " + "".join(badges), unsafe_allow_html=True)
        st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="sechead">⚡ Corrosion</div>', unsafe_allow_html=True)
        pcard("Ecorr", Ec, "V vs Ref", "#f38ba8")
        pcard("icorr (Tafel / BV)", ic, "A cm⁻²", "#fab387")
        if ic_sg: pcard("icorr (Stern-Geary)", ic_sg, "A cm⁻²", "#f9e2af")
        if CR:    pcard("Corrosion rate", CR, "mm year⁻¹", "#eba0ac")
        if CR_sg: pcard("CR (Stern-Geary)", CR_sg, "mm year⁻¹", "#f5c2e7")

    with col2:
        st.markdown('<div class="sechead">📐 Kinetics</div>', unsafe_allow_html=True)
        pcard("βa  anodic Tafel slope", ba*1000 if ba else None, "mV decade⁻¹", "#a6e3a1")
        pcard("βc  cathodic Tafel slope", bc*1000 if bc else None, "mV decade⁻¹", "#94e2d5")
        if B:  pcard("B  Stern-Geary const.", B*1000,  "mV", "#89dceb")
        if Rp: pcard("Rp  polarization resist.", Rp, "Ω cm²", "#89b4fa")
        m = best.get("method", "—"); r2 = best.get("r2")
        r2s = f"R² = {r2:.4f}" if r2 else ""
        cls = "warn-box" if best.get("fallback") else "ok-box"
        pfx = "⚠️" if best.get("fallback") else "✅"
        st.markdown(f'<div class="{cls}">{pfx} {m}<br>{r2s}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sechead">🛡️ Passivity</div>', unsafe_allow_html=True)
        if pas:
            pcard("ipass", pas["ipass"], "A cm⁻²", "#a6e3a1")
            pcard("Passive start", pas["E_start"], "V", "#94e2d5")
            pcard("Passive end", pas["E_end"], "V", "#94e2d5")
            pcard("Passive range", pas["range_V"]*1000, "mV", "#a6adc8")
        else:
            st.info("No passive region detected.")
        if reg.get("Epp") is not None:
            pcard("Epp / Flade potential", reg["Epp"], "V", "#a6e3a1")

    with col4:
        st.markdown('<div class="sechead">💥 Breakdown & Limit</div>', unsafe_allow_html=True)
        if reg.get("Eb"):
            pcard("Eb  breakdown potential", reg["Eb"], "V", "#f38ba8")
            if Ec: pcard("Pitting index Eb−Ecorr", (reg["Eb"]-Ec)*1000, "mV", "#f38ba8")
        if reg.get("iL"):
            pcard("iL  limiting current", reg["iL"], "A cm⁻²", "#89dceb")
            e_range = f"{reg['E_lim_start']:.3f} → {reg['E_lim_end']:.3f}"
            pcard("iL  E range", e_range, "V", "#89b4fa")
        if tp:
            pcard("Transpassive log-slope", tp["slope"], "dec V⁻¹", "#fab387")


# ═══════════════════════════════════════════════
# BUILD RESULTS SUMMARY DATAFRAME
# ═══════════════════════════════════════════════

def build_summary_df(R, reg, ew, rho):
    best = R.get("best", {})
    an   = R.get("an",   {})
    ca   = R.get("ca",   {})
    pas  = R.get("passive", {})
    tp   = R.get("tp",   {})
    rp_r = R.get("rp",   {})

    Ec  = best.get("Ecorr") or R.get("Ecorr_tafel") or reg["Ecorr"]
    ic  = best.get("icorr") or R.get("icorr_tafel")
    ba  = best.get("ba") or an.get("ba")
    bc  = best.get("bc") or ca.get("bc")
    B   = best.get("B")
    Rp  = rp_r.get("Rp")
    CR  = ic * 3.27 * ew / rho if ic else None
    ic_sg = B/Rp if (B and Rp) else None
    CR_sg = ic_sg * 3.27 * ew / rho if ic_sg else None

    rows = [
        ("Ecorr (V)",                  Ec),
        ("icorr (A/cm²)",              ic),
        ("icorr_SG (A/cm²)",           ic_sg),
        ("βa (mV/dec)",                ba*1000 if ba else None),
        ("βc (mV/dec)",                bc*1000 if bc else None),
        ("B Stern-Geary (mV)",         B*1000 if B else None),
        ("Rp (Ω·cm²)",                 Rp),
        ("Corrosion Rate (mm/yr)",      CR),
        ("CR Stern-Geary (mm/yr)",      CR_sg),
        ("ipass (A/cm²)",              pas.get("ipass")),
        ("E_passive_start (V)",        pas.get("E_start")),
        ("E_passive_end (V)",          pas.get("E_end")),
        ("Passive range (mV)",         pas.get("range_V", 0)*1000 if pas else None),
        ("Eb breakdown (V)",           reg.get("Eb")),
        ("Pitting index Eb−Ecorr (mV)",
         (reg["Eb"]-Ec)*1000 if reg.get("Eb") and Ec else None),
        ("Epp Flade (V)",              reg.get("Epp")),
        ("iL limiting (A/cm²)",        reg.get("iL")),
        ("Transpassive slope (dec/V)", tp.get("slope") if tp else None),
        ("Fitting method",             best.get("method")),
        ("BV fit R²",                  best.get("r2")),
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    return df


# ═══════════════════════════════════════════════
# MATERIAL DATABASE
# ═══════════════════════════════════════════════
MATERIALS = {
    "Carbon Steel / Iron":  (27.92, 7.87),
    "304 Stainless Steel":  (25.10, 7.90),
    "316 Stainless Steel":  (25.56, 8.00),
    "Copper":               (31.77, 8.96),
    "Aluminum":             (8.99,  2.70),
    "Nickel":               (29.36, 8.91),
    "Titanium":             (11.99, 4.51),
    "Zinc":                 (32.69, 7.14),
    "Custom":               (27.92, 7.87),
}

# ═══════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════

def main():
    # ── Header ─────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                border:1px solid #313244; border-radius:12px;
                padding:20px 28px; margin-bottom:20px;">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting Tool</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">
        Upload your polarization data → automatic region detection, fitting, and parameter extraction
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Minimal sidebar (just what can't be auto-detected) ──
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.caption("Only the electrode area and material cannot be auto-detected from the data.")

        area = st.number_input(
            "Electrode area (cm²)",
            min_value=0.001, max_value=100.0, value=1.0, step=0.01,
            help="Raw current ÷ area = current density. Default 1 cm²."
        )
        mat = st.selectbox("Material (for corrosion rate)", list(MATERIALS.keys()))
        ew0, rho0 = MATERIALS[mat]
        if mat == "Custom":
            ew  = st.number_input("Equivalent weight (g/eq)", 1.0, 300.0, ew0)
            rho = st.number_input("Density (g/cm³)",          0.5,  25.0, rho0)
        else:
            ew, rho = ew0, rho0

        st.divider()
        st.caption("Everything else — file format, column names, units, "
                   "region boundaries, Tafel windows — is detected automatically.")

    # ── Upload ──────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your file here — CSV, TXT, XLSX, XLS (Autolab, Gamry, BioLogic, or any generic format)",
        type=["csv", "txt", "xlsx", "xls"],
        label_visibility="visible"
    )

    if uploaded is None:
        # Landing state — show demo button
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            demo_choice = st.selectbox(
                "Or try a built-in demo",
                ["Full curve (limiting + passive + breakdown)",
                 "Active/Tafel only",
                 "With passive region only"]
            )
            if st.button("▶  Run demo", use_container_width=True, type="primary"):
                st.session_state["demo"] = demo_choice
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:16px 20px;">
            <b style="color:#89b4fa;">Auto-detected from your file</b><br><br>
            <span style="color:#a6adc8;font-size:13px;">
            ✓ File format (NOVA, Gamry, BioLogic, generic CSV…)<br>
            ✓ Potential and current columns<br>
            ✓ Current unit (A, mA, µA, A/cm²…)<br>
            ✓ Ecorr and zero-crossing<br>
            ✓ Limiting current region<br>
            ✓ Active / Tafel windows (grid search for best R²)<br>
            ✓ Passive region boundaries<br>
            ✓ Breakdown potential Eb<br>
            ✓ Transpassive region<br>
            ✓ Epp / Flade potential<br>
            ✓ Fitting strategy (BV → DE fallback)
            </span>
            </div>""", unsafe_allow_html=True)
        return

    # ── Process file ────────────────────────────────────────
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

    # Show what was auto-detected (small info bar, not a form)
    with st.expander(f"📋 Auto-detected: **{e_col}** (potential)  ·  **{i_col}** (current)  — click to preview data", expanded=False):
        st.dataframe(df[[e_col, i_col]].head(10), use_container_width=True)
        st.caption(f"File: {uploaded.name}  ·  {df.shape[0]} rows  ·  "
                   f"Current conversion factor: {i_factor} A per file unit")

    # ── Prepare arrays ──────────────────────────────────────
    E_raw = df[e_col].values.astype(float)
    i_raw = df[i_col].values.astype(float) * i_factor   # now in A

    ok = np.isfinite(E_raw) & np.isfinite(i_raw)
    E_raw, i_raw = E_raw[ok], i_raw[ok]
    idx = np.argsort(E_raw)
    E_raw, i_raw = E_raw[idx], i_raw[idx]

    i_dens = i_raw / area   # A/cm²

    # ── Full auto pipeline ──────────────────────────────────
    with st.spinner("Detecting regions…"):
        reg = detect_regions(E_raw, i_dens)

    with st.spinner("Fitting Tafel slopes and Butler-Volmer…"):
        fitter  = AutoFitter(E_raw, i_dens, reg)
        R       = fitter.run()

    # ═══════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════
    st.markdown("---")

    # Full polarization curve
    fig_full = plot_polarization(E_raw, i_dens, R, reg)
    st.plotly_chart(fig_full, use_container_width=True)

    # Two-column: zoom on active + diagnostic
    col_left, col_right = st.columns(2)
    with col_left:
        fig_zoom = plot_active_zoom(E_raw, i_dens, R, reg)
        st.plotly_chart(fig_zoom, use_container_width=True)
    with col_right:
        fig_diag = plot_diagnostic(E_raw, i_dens, R, reg)
        st.plotly_chart(fig_diag, use_container_width=True)

    # ═══════════════════════════════════════════════════════
    # PARAMETERS
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    show_parameters(R, reg, ew, rho)

    # ═══════════════════════════════════════════════════════
    # FITTING LOG + DOWNLOAD
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    c_log, c_dl = st.columns([3, 1])

    with c_log:
        with st.expander("🪵 Fitting log"):
            for msg in fitter.log:
                st.markdown(f"- {msg}")

    with c_dl:
        df_sum = build_summary_df(R, reg, ew, rho)
        st.download_button(
            "⬇️ Download results (CSV)",
            df_sum.to_csv(index=False).encode(),
            "tafel_results.csv", "text/csv",
            use_container_width=True
        )
        df_proc = pd.DataFrame({
            "E_V":       E_raw,
            "i_Acm2":    i_dens,
            "log_abs_i": safe_log10(i_dens),
        })
        st.download_button(
            "⬇️ Download processed data (CSV)",
            df_proc.to_csv(index=False).encode(),
            "tafel_data.csv", "text/csv",
            use_container_width=True
        )


# ── Demo data runner ─────────────────────────────────────
def run_demo():
    np.random.seed(42)
    E = np.linspace(-0.65, 0.85, 400)
    choice = st.session_state.get("demo", "")

    if "passive" in choice.lower() and "breakdown" in choice.lower():
        i  = 2e-6*(np.exp(2.303*(E+0.38)/0.065) - np.exp(-2.303*(E+0.38)/0.110))
        i += -5e-5/(1+np.exp(30*(E+0.52)))
        i += 3e-6/(1+np.exp(-25*(E+0.25)))
        m  = E > 0.55
        i[m] += 3e-6*np.exp(12*(E[m]-0.55))
    elif "active" in choice.lower():
        i = 5e-6*(np.exp(2.303*(E+0.45)/0.06) - np.exp(-2.303*(E+0.45)/0.12))
    else:
        i  = 2e-6*(np.exp(2.303*(E+0.40)/0.065) - np.exp(-2.303*(E+0.40)/0.110))
        i += 3e-6/(1+np.exp(-20*(E+0.20)))

    i += np.random.normal(0, np.abs(i)*0.04 + 4e-9)
    return E, i


if __name__ == "__main__":
    # Handle demo mode
    if "demo" in st.session_state:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                    border:1px solid #313244; border-radius:12px;
                    padding:20px 28px; margin-bottom:20px;">
          <h1 style="margin:0;color:#cdd6f4;font-size:26px;">⚡ Tafel Fitting Tool</h1>
          <p style="margin:4px 0 0;color:#6c7086;font-size:13px;">Demo mode</p>
        </div>""", unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            area = st.number_input("Electrode area (cm²)", 0.001, 100.0, 1.0)
            mat  = st.selectbox("Material", list(MATERIALS.keys()))
            ew, rho = MATERIALS[mat]
            if mat == "Custom":
                ew  = st.number_input("EW (g/eq)", 1.0, 300.0, ew)
                rho = st.number_input("ρ (g/cm³)",  0.5,  25.0, rho)
            if st.button("← Back to upload"):
                del st.session_state["demo"]; st.rerun()

        E_d, i_d = run_demo()
        i_dens   = i_d / area

        st.info(f"🧪 Demo: **{st.session_state['demo']}**")
        with st.spinner("Running…"):
            reg    = detect_regions(E_d, i_dens)
            fitter = AutoFitter(E_d, i_dens, reg)
            R      = fitter.run()

        st.markdown("---")
        st.plotly_chart(plot_polarization(E_d, i_dens, R, reg), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_active_zoom(E_d, i_dens, R, reg), use_container_width=True)
        with c2: st.plotly_chart(plot_diagnostic(E_d, i_dens, R, reg), use_container_width=True)
        st.markdown("---")
        show_parameters(R, reg, ew, rho)
        with st.expander("🪵 Fitting log"):
            for msg in fitter.log: st.markdown(f"- {msg}")
    else:
        main()
