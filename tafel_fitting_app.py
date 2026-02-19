"""
Tafel Fitting Tool — v4: Comprehensive Global Electrochemical Model
====================================================================
Fast global fit (<30s) across ALL electrochemically possible curve types.
Uses reduced-parameter optimization: only optimizes parameters relevant
to the detected curve type (4 params for active, up to 14 for full).

Curve types handled:
  Active only (BV) — 4 params | Active + diffusion — 5 params
  Active-passive — 8 params | Passive + transpassive — 11 params
  + secondary passivity — 14 params | Pitting — 11 params

Physics: film-coverage model for passivation (produces active peak/nose).
Optimization: DE (reduced space) -> L-BFGS-B -> Nelder-Mead.
Data quality diagnostics included.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize, curve_fit
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import linregress
from itertools import groupby
import warnings, io, re, time

warnings.filterwarnings("ignore")

# ===============================================================
# PAGE CONFIG
# ===============================================================
st.set_page_config(page_title="Tafel Fitting Tool", page_icon="⚡", layout="wide")

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
</style>""", unsafe_allow_html=True)

C = dict(data="#89b4fa", anodic="#f9e2af", cathodic="#cba6f7", fit="#a6e3a1",
         passive="rgba(166,227,161,0.10)", limiting="rgba(137,220,235,0.10)",
         transpassive="rgba(243,188,168,0.08)", sec_passive="rgba(203,166,247,0.08)",
         ecorr="#f38ba8", grid="#313244", bg="#1e1e2e", paper="#131320", text="#cdd6f4")

# ===============================================================
# FILE I/O
# ===============================================================
COL_SIG = [
    (r"we.*potential", r"we.*current", "A"), (r"ewe", r"i/ma", "mA"),
    (r"ewe", r"<i>/ma", "mA"), (r"^vf$", r"^im$", "A"),
    (r"potential/v", r"current/a", "A"), (r"e/v", r"i/a", "A"),
    (r"potential|volt|^e$|e \(v\)|e_v", r"current|amps|^i$|i \(a\)|i_a", "A"),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]
UHINT = {r"\(a\)|_a$|/a$":1.0,r"\(ma\)|_ma$|/ma$":1e-3,
         r"\(ua\)|_ua$|/ua$":1e-6,r"a/cm":1.0,r"ma/cm":1e-3}

def auto_detect_columns(df):
    cl = {c: c.lower().strip() for c in df.columns}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for Ep, Ip, u in COL_SIG:
        em = [c for c, v in cl.items() if re.search(Ep, v) and c in num]
        im = [c for c, v in cl.items() if re.search(Ip, v) and c in num and c not in em]
        if em and im:
            ec = sorted(em, key=lambda c: 0 if "we" in c.lower() else 1)[0]
            ic = im[0]; f = 1e-3 if u == "mA" else 1.0
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
    text = raw.decode("utf-8", errors="replace"); lines = text.splitlines()
    skip = 0
    for idx, line in enumerate(lines):
        pts = re.split(r"[,;\t ]+", line.strip())
        if sum(1 for p in pts if re.match(r"^-?[\d.eE+]+$", p)) >= 2:
            skip = max(0, idx-1) if idx > 0 and not any(
                re.match(r"^-?[\d.eE+]+$", p)
                for p in re.split(r"[,;\t ]+", lines[idx-1].strip())) else idx
            break
    for sep in ["\t",";",",",r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5: return df.dropna(axis=1, how="all")
        except: pass
    raise ValueError(f"Cannot parse {f.name}")

# ===============================================================
# MATH
# ===============================================================
def safe_log(x): return np.log10(np.maximum(np.abs(x), 1e-20))

def smooth(y, w=11, p=3):
    n = len(y); w = min(w, n if n%2==1 else n-1)
    return savgol_filter(y, w, min(p, w-1)) if w >= 5 else y.copy()

def _r2(yt, yp):
    sr = np.sum((yt-yp)**2); st_ = np.sum((yt-yt.mean())**2)
    return float(max(0, 1-sr/st_)) if st_ > 0 else 0.0

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -50, 50)))

# ===============================================================
# GLOBAL MODEL  (14 params, film-coverage)
# ===============================================================
PNAMES = ["Ecorr","icorr","ba","bc","iL",
          "Epp","k_pass","ipass","Eb","a_tp","b_tp",
          "Esp","k_sp","ipass2"]
LOG_IDX = (1, 4, 7, 9, 13)

def full_model(E, p):
    Ecorr,icorr,ba,bc,iL = p[0],p[1],p[2],p[3],p[4]
    Epp,k_pass,ipass = p[5],p[6],p[7]
    Eb,a_tp,b_tp = p[8],p[9],p[10]
    Esp,k_sp,ipass2 = p[11],p[12],p[13]
    eta = E - Ecorr
    ic_k = icorr * np.exp(-2.303 * eta / bc)
    i_cat = ic_k / (1.0 + ic_k / iL)
    i_act = icorr * np.exp(2.303 * eta / ba)
    t1 = sigmoid(E - Epp, k_pass)
    i_p1 = i_act * (1.0 - t1) + ipass * t1
    i_tp = a_tp * np.exp(np.clip(b_tp * (E - Eb), -50, 50)) * sigmoid(E - Eb, 40.0)
    t2 = sigmoid(E - Esp, k_sp)
    i_an = (i_p1 + i_tp) * (1.0 - t2) + ipass2 * t2
    return i_an - i_cat

def model_components(E, p):
    Ecorr,icorr,ba,bc,iL = p[0:5]
    Epp,k_pass,ipass = p[5:8]
    Eb,a_tp,b_tp = p[8:11]
    Esp,k_sp,ipass2 = p[11:14]
    eta = E - Ecorr
    ic_k = icorr*np.exp(-2.303*eta/bc)
    i_cat = ic_k/(1.0+ic_k/iL)
    i_act = icorr*np.exp(2.303*eta/ba)
    t1 = sigmoid(E-Epp, k_pass)
    i_p1 = i_act*(1-t1)+ipass*t1
    i_tp = a_tp*np.exp(np.clip(b_tp*(E-Eb),-50,50))*sigmoid(E-Eb,40)
    t2 = sigmoid(E-Esp, k_sp)
    i_an = (i_p1+i_tp)*(1-t2)+ipass2*t2
    return dict(i_cat=i_cat,i_act=i_act,i_p1=i_p1,i_tp=i_tp,i_an=i_an,
                t1=t1,t2=t2,i_tot=i_an-i_cat)

# ===============================================================
# CURVE TYPES
# ===============================================================
class CT:
    ACTIVE="active"; ACTIVE_D="active_d"; PASSIVE="passive"
    PASSIVE_D="passive_d"; PASS_TP="pass_tp"; PASS_TP_SP="pass_tp_sp"
    PITTING="pitting"; FULL="full"
    DESC = {
        "active":     ("Active Only", "Pure activation kinetics.", 4),
        "active_d":   ("Active + Diffusion", "Active + cathodic mass-transport limit.", 5),
        "passive":    ("Active-Passive", "Active dissolution peak -> passive plateau.", 8),
        "passive_d":  ("Active-Passive + Diffusion", "Passivating + cathodic diffusion.", 9),
        "pass_tp":    ("Passive-Transpassive", "Passive + transpassive dissolution.", 11),
        "pass_tp_sp": ("Full: + Secondary Passivity", "Active->passive->transpassive->2nd passive.", 14),
        "pitting":    ("Passive + Pitting", "Passive with sharp pitting breakdown.", 11),
        "full":       ("Full Multi-Region", "All regions active.", 14),
    }
    @staticmethod
    def n_free(ct): return CT.DESC.get(ct,("","",14))[2]
    @staticmethod
    def free_mask(ct):
        m = [True]*4 + [False]*10  # Ecorr,icorr,ba,bc always free
        if ct in (CT.ACTIVE_D,CT.PASSIVE_D,CT.FULL,CT.PASS_TP_SP): m[4]=True
        if ct not in (CT.ACTIVE,CT.ACTIVE_D): m[5]=m[6]=m[7]=True
        if ct in (CT.PASS_TP,CT.PASS_TP_SP,CT.PITTING,CT.FULL): m[8]=m[9]=m[10]=True
        if ct in (CT.PASS_TP_SP,CT.FULL): m[11]=m[12]=m[13]=True
        return m

# ===============================================================
# REGION DETECTION
# ===============================================================
def detect_regions(E, i):
    reg = {}; n = len(E); abs_i = np.abs(i)
    sc = np.where(np.diff(np.sign(i)))[0]
    if len(sc) > 0:
        k = sc[0]; d = i[k+1]-i[k]
        reg["Ecorr"] = float(E[k]-i[k]*(E[k+1]-E[k])/d) if abs(d)>0 else float(E[k])
        reg["ec_idx"] = k
    else:
        k = int(np.argmin(abs_i)); reg["Ecorr"]=float(E[k]); reg["ec_idx"]=k
    Ec = reg["Ecorr"]

    ci = np.where(E < Ec)[0]
    if len(ci) >= 8:
        lc = smooth(safe_log(abs_i[ci]), min(11,(len(ci)//2)*2-1 or 5))
        dl = np.abs(np.gradient(lc, E[ci]))
        thr = np.percentile(dl, 20); flat = dl < max(thr, 0.5)
        runs = [(k2,list(g)) for k2,g in groupby(enumerate(flat), key=lambda x:x[1]) if k2]
        if runs:
            br = max(runs, key=lambda x:len(x[1])); idxs=[s[0] for s in br[1]]
            er = abs(E[ci[idxs[-1]]]-E[ci[idxs[0]]])
            if len(idxs)>=4 and er>0.03 and abs(E[ci[idxs[0]]]-Ec)>0.06:
                reg.update(iL=float(np.median(abs_i[ci[idxs]])),
                           E_ls=float(E[ci[idxs[0]]]),E_le=float(E[ci[idxs[-1]]]))

    ai = np.where(E > Ec)[0]
    if len(ai) < 6:
        reg["ct"] = CT.ACTIVE_D if "iL" in reg else CT.ACTIVE; return reg

    log_a = smooth(safe_log(abs_i[ai]), min(15,(len(ai)//2)*2-1 or 5))
    dlog = np.gradient(log_a, E[ai]); abs_dl = np.abs(dlog)
    peaks = []
    if len(ai) > 20:
        i_sm = smooth(abs_i[ai], min(21,(len(ai)//2)*2-1 or 5))
        order = max(3, len(ai)//20)
        pk_idx = argrelextrema(safe_log(i_sm), np.greater, order=order)[0]
        for pk in pk_idx:
            rest = safe_log(i_sm)[pk:]
            if len(rest) > 5:
                prom = safe_log(i_sm)[pk]-np.min(rest[3:])
                if prom > 0.3:
                    peaks.append(dict(idx=ai[pk],E=float(E[ai[pk]]),prom=float(prom)))
    reg["peaks"] = peaks

    thr_p = np.percentile(abs_dl, 25); flat = abs_dl < max(thr_p, 0.8)
    runs = [(k2,list(g)) for k2,g in groupby(enumerate(flat), key=lambda x:x[1]) if k2]
    pass_regs = []
    for _, ri in runs:
        idxs=[s[0] for s in ri]; er=abs(E[ai[idxs[-1]]]-E[ai[idxs[0]]])
        if len(idxs)>=4 and er>0.03:
            ps,pe = ai[idxs[0]],ai[idxs[-1]]
            im = float(np.median(abs_i[ps:pe+1]))
            pre = np.where((E>Ec) & (E<E[ps]))[0]
            valid = (len(pre)>2 and im<np.max(abs_i[pre])*0.5) or er>0.08
            if valid:
                pass_regs.append(dict(ps=ps,pe=pe,Es=float(E[ps]),Ee=float(E[pe]),ipass=im,rng=er))

    if len(pass_regs)>=1:
        reg["p1"]=pass_regs[0]; reg["Epp"]=pass_regs[0]["Es"]; reg["ipass"]=pass_regs[0]["ipass"]
    if len(pass_regs)>=2:
        reg["p2"]=pass_regs[1]; reg["Esp"]=pass_regs[1]["Es"]; reg["ipass2"]=pass_regs[1]["ipass"]

    if "p1" in reg:
        pe = reg["p1"]["pe"]
        if pe+5 < n:
            da = np.gradient(safe_log(abs_i[pe:]), E[pe:])
            thr_b = np.percentile(np.abs(da), 75)
            jump = np.where(np.abs(da)>max(thr_b, 2.0))[0]
            if len(jump):
                reg["Eb"] = float(E[pe+jump[0]])
                reg["is_pit"] = np.mean(np.abs(da[jump[:min(5,len(jump))]])
                    ) > 10 if len(jump)>2 else False

    hd,hp,ht,hsp = "iL" in reg,"p1" in reg,"Eb" in reg,"p2" in reg
    pit = reg.get("is_pit", False)
    if hsp:          ct = CT.PASS_TP_SP if ht else CT.FULL
    elif ht and pit: ct = CT.PITTING
    elif ht:         ct = CT.PASS_TP if not hd else CT.FULL
    elif hp:         ct = CT.PASSIVE_D if hd else CT.PASSIVE
    elif hd:         ct = CT.ACTIVE_D
    else:            ct = CT.ACTIVE
    reg["ct"] = ct
    return reg

# ===============================================================
# INITIAL GUESS
# ===============================================================
def initial_guess(E, i, reg):
    Ec=reg["Ecorr"]; abs_i=np.abs(i); lg=safe_log(i)
    ba0,bc0,ic0 = 0.060, 0.120, max(abs_i[reg["ec_idx"]], 1e-10)

    Epp = reg.get("Epp", Ec+0.50)
    for lo in np.arange(0.005,0.05,0.005):
        for hi in np.arange(lo+0.02,min(lo+0.18,Epp-Ec-0.01),0.005):
            m=(E>Ec+lo)&(E<Ec+hi)
            if m.sum()<4: continue
            s,b,r,*_=linregress(E[m],lg[m])
            if s>0 and 20<1000/s<500 and r**2>0.88: ba0=1/s; break
        else: continue
        break
    E_le = reg.get("E_le")
    for lo in np.arange(0.005,0.08,0.005):
        for hi in np.arange(lo+0.02,lo+0.25,0.005):
            m=(E<Ec-lo)&(E>Ec-hi)
            if E_le is not None: m=m&(E>E_le+0.005)
            if m.sum()<4: continue
            s,b,r,*_=linregress(E[m],lg[m])
            if s<0 and 20<-1000/s<500 and r**2>0.88: bc0=-1/s; break
        else: continue
        break

    return np.array([
        Ec, ic0, ba0, bc0,
        reg.get("iL", 1e3),
        reg.get("Epp", Ec+0.30),
        40.0 if "p1" in reg else 0.01,
        reg.get("ipass", 1e3),
        reg.get("Eb", E[-1]+0.5),
        reg.get("ipass",1e-6) if "Eb" in reg else 1e-15,
        8.0,
        reg.get("Esp", E[-1]+1.0),
        30.0 if "p2" in reg else 0.01,
        reg.get("ipass2", 1e3)])

# ===============================================================
# FAST OPTIMIZER
# ===============================================================
class FastOptimizer:
    def __init__(self, E, i, reg, p0):
        self.E=E; self.i=i; self.ld=safe_log(i); self.reg=reg
        self.ct=reg["ct"]; self.p_full=p0.copy()
        self.mask=CT.free_mask(self.ct); self.n_free=sum(self.mask)
        self.idx_free=[j for j in range(14) if self.mask[j]]
        self.log=[]; self.best_p=p0.copy(); self.best_score=1e30
        self._build_bounds()

    def _build_bounds(self):
        p=self.p_full; reg=self.reg; Ec=p[0]; ic=p[1]
        lo = np.array([Ec-0.15, max(ic*1e-4,1e-14), 0.010, 0.010,
            reg.get("iL",1e-4)*0.01 if "iL" in reg else 1e-1,
            p[5]-0.20, 5.0, max(p[7]*0.01,1e-10),
            p[8]-0.25, max(p[9]*1e-4,1e-15), 0.5,
            p[11]-0.25, 5.0, max(p[13]*0.01,1e-10)])
        hi = np.array([Ec+0.15, min(ic*1e4,1e0), 0.500, 0.500,
            reg.get("iL",1e0)*100 if "iL" in reg else 1e5,
            p[5]+0.20, 200.0, max(p[7]*100,1e-2),
            p[8]+0.25, max(p[9]*1e4,1e-2), 40.0,
            p[11]+0.25, 200.0, max(p[13]*100,1e-2)])
        for j in range(14):
            if lo[j]>=hi[j]: mid=p[j]; lo[j]=mid-abs(mid)*0.5-1e-6; hi[j]=mid+abs(mid)*0.5+1e-6
        self.lo_full=lo; self.hi_full=hi

    def _pack(self, pf):
        x=[]
        for j in self.idx_free:
            v=pf[j]; x.append(np.log10(max(v,1e-15)) if j in LOG_IDX else v)
        return np.array(x)

    def _unpack(self, x):
        p=self.p_full.copy()
        for k,j in enumerate(self.idx_free):
            p[j] = 10**x[k] if j in LOG_IDX else x[k]
        return p

    def _pack_bounds(self):
        b=[]
        for j in self.idx_free:
            lo,hi = self.lo_full[j], self.hi_full[j]
            if j in LOG_IDX:
                b.append((np.log10(max(lo,1e-15)), np.log10(max(hi,1e-14))))
            else: b.append((lo,hi))
        return b

    def _obj(self, x):
        p=self._unpack(x)
        try: return float(np.sum((self.ld-safe_log(full_model(self.E,p)))**2))
        except: return 1e30

    def _update(self, x, tag):
        p=self._unpack(x); s=self._obj(x)
        r2=_r2(self.ld, safe_log(full_model(self.E,p)))
        if s<self.best_score:
            self.best_p=p.copy(); self.best_score=s
            self.log.append(f"  {tag}: R2(log) = {r2:.6f}")
            return True
        self.log.append(f"  {tag}: R2(log) = {r2:.6f} (no improvement)")
        return False

    def run(self):
        t0=time.time()
        nf = self.n_free
        self.log.append(f"Curve type: {CT.DESC[self.ct][0]}  ({nf} free params)")
        x0=self._pack(self.p_full)
        self.best_score=self._obj(x0)
        r2i=_r2(self.ld, safe_log(full_model(self.E,self.p_full)))
        self.log.append(f"  Initial guess: R2={r2i:.4f}")
        bounds=self._pack_bounds()

        # Stage 1: DE
        popsize=max(10, min(15, nf*2))
        maxiter=max(200, min(600, nf*50))
        self.log.append(f"Stage 1: DE  pop={popsize} iter={maxiter}")
        try:
            t1=time.time()
            res=differential_evolution(self._obj, bounds, seed=42,
                maxiter=maxiter, tol=1e-12, popsize=popsize,
                mutation=(0.5,1.5), recombination=0.9, polish=False)
            self._update(res.x, f"DE ({time.time()-t1:.1f}s, {res.nfev} evals)")
        except Exception as ex:
            self.log.append(f"  DE failed: {ex}")

        # Stage 2: L-BFGS-B
        self.log.append("Stage 2: L-BFGS-B")
        try:
            t1=time.time()
            r2=minimize(self._obj, self._pack(self.best_p), method="L-BFGS-B",
                bounds=bounds, options={"maxiter":15000,"ftol":1e-15,"gtol":1e-12})
            self._update(r2.x, f"L-BFGS-B ({time.time()-t1:.1f}s)")
        except Exception as ex:
            self.log.append(f"  L-BFGS-B: {ex}")

        # Stage 3: Nelder-Mead
        self.log.append("Stage 3: Nelder-Mead polish")
        try:
            t1=time.time()
            r3=minimize(self._obj, self._pack(self.best_p), method="Nelder-Mead",
                options={"maxiter":30000,"xatol":1e-14,"fatol":1e-16,"adaptive":True})
            self._update(r3.x, f"NM ({time.time()-t1:.1f}s)")
        except Exception as ex:
            self.log.append(f"  NM: {ex}")

        dt=time.time()-t0
        r2f=_r2(self.ld, safe_log(full_model(self.E,self.best_p)))
        q = ("Excellent" if r2f>=0.995 else "Good" if r2f>=0.97
             else "Acceptable" if r2f>=0.90 else "Poor")
        self.log.append(f"Result: {q} fit  R2(log) = {r2f:.6f}  [{dt:.1f}s]")
        return self.best_p, r2f

# ===============================================================
# DIAGNOSTICS
# ===============================================================
def diagnose(E, i, reg, bp, r2):
    issues=[]; abs_i=np.abs(i); lg=safe_log(i); n=len(E)
    if n > 20:
        lsm=smooth(lg, min(21,(n//4)*2-1 or 5)); noise=np.std(lg-lsm)
        if noise>0.5: issues.append(("High noise", f"sigma = {noise:.2f} decades. Slower scan rate recommended.", "err"))
        elif noise>0.15: issues.append(("Moderate noise", f"sigma = {noise:.2f} decades.", "warn"))
        else: issues.append(("Low noise", f"sigma = {noise:.3f} decades. Excellent.", "ok"))
    if bp is not None:
        if bp[2]*1000>200: issues.append(("Large anodic Tafel slope",
            f"ba={bp[2]*1000:.0f} mV/dec. Possible IR drop, multi-step mechanism, or mixed potential.", "warn"))
        if bp[3]*1000>200: issues.append(("Large cathodic Tafel slope",
            f"bc={bp[3]*1000:.0f} mV/dec. Possible diffusion in Tafel region or IR drop.", "warn"))
    Er=E[-1]-E[0]
    if Er<0.3: issues.append(("Narrow scan", f"{Er*1000:.0f} mV range.", "warn"))
    if n/max(Er,0.01)<50: issues.append(("Low density", f"{n/max(Er,0.01):.0f} pts/V.", "warn"))
    Ec=reg["Ecorr"]
    if np.sum(np.abs(E-Ec)<0.05)<5: issues.append(("Sparse near Ecorr", "<5 pts within 50 mV.", "warn"))
    if bp is not None:
        res=lg-safe_log(full_model(E,bp))
        if n>10:
            nruns=np.sum(np.abs(np.diff(np.sign(res)))>0)+1
            if nruns<n*0.25: issues.append(("Systematic misfit",
                f"{nruns} sign changes (expect ~{n//2}). Model may miss a feature.", "warn"))
        if bp[1]>0.1: issues.append(("Very high icorr", f"{bp[1]:.2e} A/cm2.", "err"))
        if bp[1]<1e-12: issues.append(("Very low icorr", f"{bp[1]:.2e} A/cm2.", "warn"))
    if r2 is not None and r2<0.90: issues.append(("Poor fit",
        f"R2(log)={r2:.4f}. Possible: coupled reactions, adsorption, roughness, artifacts.", "err"))
    return issues

# ===============================================================
# PLOTS
# ===============================================================
def plot_main(E, i, bp, reg, ct):
    lg=safe_log(i); fig=go.Figure()
    if "p1" in reg:
        p=reg["p1"]; fig.add_vrect(x0=p["Es"],x1=p["Ee"],fillcolor=C["passive"],layer="below",line_width=0,
            annotation=dict(text="Passive",font=dict(color="#a6e3a1",size=11)))
    if "p2" in reg:
        p=reg["p2"]; fig.add_vrect(x0=p["Es"],x1=p["Ee"],fillcolor=C["sec_passive"],layer="below",line_width=0,
            annotation=dict(text="2nd Passive",font=dict(color="#cba6f7",size=10)))
    if "E_ls" in reg:
        fig.add_vrect(x0=reg["E_ls"],x1=reg["E_le"],fillcolor=C["limiting"],layer="below",line_width=0,
            annotation=dict(text="Limiting",font=dict(color="#89dceb",size=11)))
    if reg.get("Eb"):
        te=reg.get("Esp",E[-1]) if "p2" in reg else E[-1]
        fig.add_vrect(x0=reg["Eb"],x1=te,fillcolor=C["transpassive"],layer="below",line_width=0,
            annotation=dict(text="Transpassive",font=dict(color="#fab387",size=10)))
    Ec=reg["Ecorr"]
    fig.add_vline(x=Ec,line=dict(color=C["ecorr"],width=1.5,dash="dot"),
        annotation=dict(text="Ecorr",font=dict(color=C["ecorr"],size=10)))
    if reg.get("Epp"):
        fig.add_vline(x=reg["Epp"],line=dict(color="#a6e3a1",width=1,dash="dot"),
            annotation=dict(text="Epp",font=dict(color="#a6e3a1",size=10)))
    if reg.get("Eb"):
        fig.add_vline(x=reg["Eb"],line=dict(color="#f38ba8",width=1,dash="dash"),
            annotation=dict(text="Eb",font=dict(color="#f38ba8",size=10)))
    fig.add_trace(go.Scatter(x=E,y=lg,mode="lines",name="Measured",line=dict(color=C["data"],width=2.5)))
    if bp is not None:
        Em=np.linspace(E.min(),E.max(),800)
        try:
            im=full_model(Em,bp); r2v=_r2(lg,safe_log(full_model(E,bp)))
            fig.add_trace(go.Scatter(x=Em,y=safe_log(im),mode="lines",
                name=f"Global Fit  R2(log)={r2v:.4f}",line=dict(color=C["fit"],width=3)))
        except: pass
        fig.add_trace(go.Scatter(x=[bp[0]],y=[np.log10(max(bp[1],1e-20))],mode="markers",
            name=f"icorr = {bp[1]:.3e} A/cm2",
            marker=dict(symbol="x-thin",size=18,color=C["ecorr"],line=dict(width=4,color=C["ecorr"]))))
        if ct in (CT.ACTIVE_D,CT.PASSIVE_D,CT.FULL,CT.PASS_TP_SP) and bp[4]<1e2:
            fig.add_hline(y=np.log10(bp[4]),line=dict(color="#89dceb",width=1,dash="dot"),
                annotation=dict(text=f"iL={bp[4]:.2e}",font=dict(color="#89dceb",size=10)))
    fig.update_layout(template="plotly_dark",plot_bgcolor=C["bg"],paper_bgcolor=C["paper"],
        title=dict(text="Potentiodynamic Polarization - Global Fit",font=dict(size=17,color=C["text"])),
        xaxis=dict(title="Potential (V vs Ref)",gridcolor=C["grid"],color=C["text"]),
        yaxis=dict(title="log10|i| (A cm-2)",gridcolor=C["grid"],color=C["text"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=C["grid"],font=dict(color=C["text"],size=11),x=0.01,y=0.01),
        height=540,margin=dict(l=70,r=20,t=50,b=60),hovermode="x unified")
    return fig

def plot_components(E, bp, ct):
    if bp is None: return None
    Em=np.linspace(E.min(),E.max(),800); comp=model_components(Em,bp); fig=go.Figure()
    fig.add_trace(go.Scatter(x=Em,y=safe_log(comp["i_cat"]),mode="lines",name="Cathodic",
        line=dict(color=C["cathodic"],width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=Em,y=safe_log(comp["i_act"]),mode="lines",name="Active Tafel",
        line=dict(color=C["anodic"],width=1,dash="dash")))
    if ct not in (CT.ACTIVE,CT.ACTIVE_D):
        fig.add_trace(go.Scatter(x=Em,y=safe_log(comp["i_p1"]),mode="lines",name="After passivation",
            line=dict(color="#a6e3a1",width=1.5,dash="dot")))
        fig.add_trace(go.Scatter(x=Em,y=safe_log(comp["i_tp"]),mode="lines",name="Transpassive",
            line=dict(color="#fab387",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=Em,y=safe_log(comp["i_tot"]),mode="lines",name="Total",
        line=dict(color=C["fit"],width=3)))
    fig.update_layout(template="plotly_dark",plot_bgcolor=C["bg"],paper_bgcolor=C["paper"],
        title=dict(text="Component Breakdown",font=dict(size=15,color=C["text"])),
        xaxis=dict(title="Potential (V)",gridcolor=C["grid"],color=C["text"]),
        yaxis=dict(title="log10|i|",gridcolor=C["grid"],color=C["text"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=C["grid"],font=dict(color=C["text"],size=10)),
        height=420,margin=dict(l=70,r=20,t=50,b=60))
    return fig

def plot_residuals(E, i, bp):
    if bp is None: return None
    ld=safe_log(i); lp=safe_log(full_model(E,bp)); res=ld-lp; rmse=np.sqrt(np.mean(res**2))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.55,0.45],vertical_spacing=0.06,
        subplot_titles=("Overlay","Residuals (log)"))
    fig.add_trace(go.Scatter(x=E,y=ld,mode="lines",name="Measured",line=dict(color=C["data"],width=2)),row=1,col=1)
    Em=np.linspace(E.min(),E.max(),600)
    fig.add_trace(go.Scatter(x=Em,y=safe_log(full_model(Em,bp)),mode="lines",name="Fit",
        line=dict(color=C["fit"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=E,y=res,mode="lines",name="Residual",line=dict(color="#fab387",width=1),
        fill="tozeroy",fillcolor="rgba(250,179,135,0.06)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(color="#585b70",width=1,dash="dot"),row=2)
    fig.add_annotation(text=f"RMSE(log) = {rmse:.4f}",xref="paper",yref="paper",x=0.98,y=0.38,
        showarrow=False,font=dict(color="#a6adc8",size=11),bgcolor="rgba(19,19,32,0.8)",bordercolor=C["grid"])
    fig.update_layout(template="plotly_dark",plot_bgcolor=C["bg"],paper_bgcolor=C["paper"],height=440,
        margin=dict(l=70,r=20,t=40,b=60),showlegend=True,
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=C["grid"],font=dict(color=C["text"],size=11)))
    fig.update_yaxes(gridcolor=C["grid"],color=C["text"])
    fig.update_xaxes(gridcolor=C["grid"],color=C["text"],title_text="Potential (V)",row=2)
    return fig

# ===============================================================
# PARAMETER DISPLAY
# ===============================================================
def pcard(label, val, unit="", color="#cdd6f4"):
    if val is None: disp="--"
    elif isinstance(val,str): disp=val
    elif abs(val)<0.001 or abs(val)>9999: disp=f"{val:.4e}"
    else: disp=f"{val:.4f}"
    st.markdown(f'<div class="pcard"><div class="plabel">{label}</div>'
                f'<div class="pval" style="color:{color}">{disp}</div>'
                f'<div class="punit">{unit}</div></div>',unsafe_allow_html=True)

def show_params(bp, reg, ew, rho, ct, r2):
    if bp is None: return
    p=dict(zip(PNAMES,bp)); ba=p["ba"]; bc=p["bc"]
    B=(ba*bc)/(2.303*(ba+bc)) if ba>0 and bc>0 else None
    ic=p["icorr"]; CR=ic*3.27*ew/rho if ic else None
    badges=[]
    if "iL" in reg: badges.append('<span class="badge bb">Limiting</span>')
    if "p1" in reg: badges.append('<span class="badge bg">Passive</span>')
    if reg.get("Eb"): badges.append('<span class="badge br">Breakdown</span>')
    if "p2" in reg: badges.append('<span class="badge bp">2nd Passive</span>')
    pk=reg.get("peaks",[])
    if pk: badges.append(f'<span class="badge by">{len(pk)} peak(s)</span>')
    if badges: st.markdown("**Detected:** "+"".join(badges),unsafe_allow_html=True)
    title,desc,nf=CT.DESC.get(ct,("?","",14))
    clr={"active":"#f9e2af","active_d":"#89dceb","passive":"#a6e3a1","passive_d":"#94e2d5",
         "pass_tp":"#fab387","pass_tp_sp":"#cba6f7","pitting":"#f38ba8","full":"#f5c2e7"}.get(ct,"#cdd6f4")
    st.markdown(f'<div class="type-box"><div class="type-title" style="color:{clr}">{title}</div>'
                f'<div class="type-desc">{desc} ({nf} free params)</div></div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown('<div class="sechead">Corrosion</div>',unsafe_allow_html=True)
        pcard("Ecorr",p["Ecorr"],"V vs Ref","#f38ba8")
        pcard("icorr",ic,"A cm-2","#fab387")
        if CR: pcard("Corrosion rate",CR,"mm yr-1","#eba0ac")
        if B: pcard("B (Stern-Geary)",B*1000,"mV","#89dceb")
    with c2:
        st.markdown('<div class="sechead">Kinetics</div>',unsafe_allow_html=True)
        pcard("ba anodic",ba*1000,"mV dec-1","#a6e3a1")
        pcard("bc cathodic",bc*1000,"mV dec-1","#94e2d5")
        cls="ok-box" if r2 and r2>=0.97 else "warn-box"
        pfx="OK" if r2 and r2>=0.97 else "!"
        st.markdown(f'<div class="{cls}">{pfx} R2(log) = {r2:.5f}</div>',unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="sechead">Passivity</div>',unsafe_allow_html=True)
        if ct not in (CT.ACTIVE,CT.ACTIVE_D):
            pcard("Epp",p["Epp"],"V","#a6e3a1"); pcard("ipass",p["ipass"],"A cm-2","#94e2d5")
            pcard("k_pass",p["k_pass"],"V-1","#a6adc8")
            if ct in (CT.PASS_TP_SP,CT.FULL):
                pcard("Esp (2nd)",p["Esp"],"V","#cba6f7"); pcard("ipass2",p["ipass2"],"A cm-2","#cba6f7")
        else: st.info("No passivation.")
    with c4:
        st.markdown('<div class="sechead">Breakdown / Diffusion</div>',unsafe_allow_html=True)
        if ct in (CT.PASS_TP,CT.PASS_TP_SP,CT.PITTING,CT.FULL):
            pcard("Eb",p["Eb"],"V","#f38ba8"); pcard("Eb-Ecorr",(p["Eb"]-p["Ecorr"])*1000,"mV","#f38ba8")
            pcard("a_tp",p["a_tp"],"A cm-2","#fab387"); pcard("b_tp",p["b_tp"],"V-1","#fab387")
        if ct in (CT.ACTIVE_D,CT.PASSIVE_D,CT.FULL,CT.PASS_TP_SP) and p["iL"]<1e2:
            pcard("iL",p["iL"],"A cm-2","#89dceb")
            if reg.get("iL"): pcard("iL (detected)",reg["iL"],"A cm-2","#89b4fa")

# ===============================================================
# MATERIALS & DEMOS
# ===============================================================
MATS = {"Carbon Steel / Iron":(27.92,7.87),"304 Stainless Steel":(25.10,7.90),
    "316 Stainless Steel":(25.56,8.00),"Copper":(31.77,8.96),"Aluminum":(8.99,2.70),
    "Nickel":(29.36,8.91),"Titanium":(11.99,4.51),"Zinc":(32.69,7.14),"Custom":(27.92,7.87)}

def make_demo(ch):
    np.random.seed(42)
    if "secondary" in ch.lower() or "full" in ch.lower():
        E=np.linspace(-0.65,1.0,600)
        p=[-0.38,3e-6,0.060,0.110,5e-4,-0.10,50.0,4e-6,0.50,2e-6,10.0,0.80,40.0,3e-6]
    elif "pitting" in ch.lower():
        E=np.linspace(-0.55,0.55,400)
        p=[-0.35,2e-6,0.055,0.120,1e3,-0.05,60.0,3e-6,0.35,1e-4,25.0,1.5,0.01,1e3]
    elif "trans" in ch.lower():
        E=np.linspace(-0.55,0.75,450)
        p=[-0.38,2e-6,0.065,0.110,1e3,-0.10,45.0,5e-6,0.55,3e-6,8.0,1.5,0.01,1e3]
    elif "passive" in ch.lower():
        E=np.linspace(-0.55,0.50,400)
        p=[-0.40,2e-6,0.065,0.110,1e3,-0.15,50.0,4e-6,0.8,1e-15,1.0,1.5,0.01,1e3]
    elif "diffusion" in ch.lower():
        E=np.linspace(-0.80,0.30,400)
        p=[-0.40,3e-6,0.070,0.120,2e-4,0.8,0.01,1e3,1.5,1e-15,1.0,2.0,0.01,1e3]
    else:
        E=np.linspace(-0.65,0.40,350)
        p=[-0.45,5e-6,0.060,0.120,1e3,0.8,0.01,1e3,1.5,1e-15,1.0,2.0,0.01,1e3]
    i=full_model(E,p)
    return E, i+np.random.normal(0,np.abs(i)*0.03+3e-9,len(E))

# ===============================================================
# MAIN PIPELINE
# ===============================================================
def process(E, i_d, area, ew, rho):
    prog=st.progress(0,text="Detecting regions...")
    reg=detect_regions(E,i_d); ct=reg["ct"]
    prog.progress(15,text="Building initial estimates...")
    p0=initial_guess(E,i_d,reg)
    prog.progress(25,text=f"Optimizing ({CT.n_free(ct)} free params)...")
    opt=FastOptimizer(E,i_d,reg,p0)
    bp,r2=opt.run()
    prog.progress(90,text="Diagnostics...")
    diags=diagnose(E,i_d,reg,bp,r2)
    prog.progress(100,text="Done!"); prog.empty()

    st.markdown("---")
    st.plotly_chart(plot_main(E,i_d,bp,reg,ct),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fc=plot_components(E,bp,ct)
        if fc: st.plotly_chart(fc,use_container_width=True)
    with c2:
        fr=plot_residuals(E,i_d,bp)
        if fr: st.plotly_chart(fr,use_container_width=True)

    st.markdown("---")
    show_params(bp,reg,ew,rho,ct,r2)

    st.markdown("---")
    st.markdown("### Data Quality Diagnostics")
    for title_d,msg,sev in diags:
        cls={"err":"err-box","warn":"warn-box","ok":"ok-box"}[sev]
        st.markdown(f'<div class="{cls}"><b>{title_d}</b><br>{msg}</div>',unsafe_allow_html=True)

    if bp is not None:
        with st.expander("Model equation"):
            p=dict(zip(PNAMES,bp))
            st.markdown(f"""
**Film-Coverage Global Model** (i_net = i_anodic - i_cathodic)

**Cathodic:** i_c = icorr * exp(-2.303*eta/bc) / (1 + i_c_kin/iL)  |  iL = {p['iL']:.3e}

**Anodic active:** i_act = icorr * exp(2.303*eta/ba)

**Primary passivation:** theta1 = sigmoid(k1*(E-Epp))
- i_pass = i_act*(1-theta1) + ipass*theta1
- Epp = {p['Epp']:.4f} V, ipass = {p['ipass']:.3e}, k1 = {p['k_pass']:.1f} V-1
- This produces the active dissolution peak at E ~ Epp

**Transpassive:** i_tp = a_tp * exp(b_tp*(E-Eb)) * sigmoid(E-Eb)  |  Eb = {p['Eb']:.4f} V

**Secondary passivation:** theta2 = sigmoid(k2*(E-Esp))
- i_total = (i_pass+i_tp)*(1-theta2) + ipass2*theta2
- Esp = {p['Esp']:.4f} V

**Fitted:** Ecorr={p['Ecorr']:.4f} V, icorr={p['icorr']:.3e}, ba={p['ba']*1000:.1f} mV/dec, bc={p['bc']*1000:.1f} mV/dec
""")

    with st.expander("Optimization log"):
        for msg in opt.log: st.markdown(f"- {msg}")

    cd1,cd2=st.columns(2)
    with cd1:
        rows=[("Curve type",CT.DESC.get(ct,("?","",0))[0]),("Ecorr (V)",bp[0] if bp is not None else None),
            ("icorr (A/cm2)",bp[1] if bp is not None else None),
            ("ba (mV/dec)",bp[2]*1000 if bp is not None else None),
            ("bc (mV/dec)",bp[3]*1000 if bp is not None else None),("R2(log)",r2)]
        for t,m,s in diags: rows.append((t,m))
        df_csv=pd.DataFrame(rows,columns=["Parameter","Value"])
        st.download_button("Results + Diagnostics",df_csv.to_csv(index=False).encode(),
            "tafel_results.csv","text/csv",use_container_width=True)
    with cd2:
        st.download_button("Processed data",pd.DataFrame(
            {"E_V":E,"i_Acm2":i_d,"log_abs_i":safe_log(i_d)}).to_csv(index=False).encode(),
            "tafel_data.csv","text/csv",use_container_width=True)

def main():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1e2e,#131320);
                border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px">Tafel Fitting Tool v4</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px">
        Global fit | Film-coverage model | All curve types | Fast (&lt;30s) | Diagnostics
      </p>
    </div>""",unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Settings")
        area=st.number_input("Electrode area (cm2)",0.001,100.0,1.0,0.01)
        mat=st.selectbox("Material",list(MATS.keys()))
        ew0,rho0=MATS[mat]
        ew,rho=(st.number_input("EW",1.0,300.0,ew0),st.number_input("rho",0.5,25.0,rho0)) if mat=="Custom" else (ew0,rho0)
        st.divider()
        st.markdown("""<div style="font-size:11px;color:#a6adc8;line-height:2.0">
        <b style="color:#89b4fa">Auto-detected types:</b><br>
        Active | +Diffusion | Passive<br>
        +Transpassive | +2nd Passive | Pitting<br><br>
        <b>Only free params optimized</b><br>
        4 params (active) to 14 params (full)
        </div>""",unsafe_allow_html=True)
    up=st.file_uploader("Upload polarization data",type=["csv","txt","xlsx","xls"])
    if up is None:
        c1,c2,c3=st.columns([1,2,1])
        with c2:
            demo=st.selectbox("Or try a demo",["Active only","Diffusion-limited cathodic",
                "Active to Passive","Passive + Transpassive","Passive + Pitting",
                "Full: Passive + Transpassive + Secondary Passivity"])
            if st.button("Run demo",use_container_width=True,type="primary"):
                st.session_state["demo"]=demo
        return
    with st.spinner("Reading..."):
        try: df=load_any_file(up)
        except Exception as ex: st.error(f"Read error: {ex}"); return
    with st.spinner("Detecting columns..."):
        try: ec,ic,ifac=auto_detect_columns(df)
        except Exception as ex: st.error(f"Column error: {ex}"); return
    with st.expander(f"Detected: {ec} and {ic}",expanded=False):
        st.dataframe(df[[ec,ic]].head(10),use_container_width=True)
    E=df[ec].values.astype(float); ir=df[ic].values.astype(float)*ifac
    ok=np.isfinite(E)&np.isfinite(ir); E,ir=E[ok],ir[ok]
    idx=np.argsort(E); E,ir=E[idx],ir[idx]
    process(E,ir/area,area,ew,rho)

if __name__ == "__main__":
    if "demo" in st.session_state:
        st.markdown('<div style="background:linear-gradient(135deg,#1e1e2e,#131320);'
            'border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">'
            '<h1 style="margin:0;color:#cdd6f4;font-size:26px">Tafel v4</h1></div>',unsafe_allow_html=True)
        with st.sidebar:
            area=st.number_input("Area",0.001,100.0,1.0)
            mat=st.selectbox("Material",list(MATS.keys()))
            ew,rho=MATS[mat]
            if st.button("Back"): del st.session_state["demo"]; st.rerun()
        E_d,i_d=make_demo(st.session_state["demo"])
        st.info(f"Demo: {st.session_state['demo']}")
        process(E_d,i_d/1.0,1.0,ew,rho)
    else:
        main()
