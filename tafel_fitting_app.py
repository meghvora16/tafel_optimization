# ⚡ Tafel Fitting Tool — Global Polarization Curve Fitter
# High-quality + reasonable speed:
# - Dual cathodic (O2 diffusion-limited + H2 evolution)
# - Film-coverage passivation, transpassive, secondary passivity
# - Optional uncompensated resistance Rs (ohmic drop)
# - Optional capacitive current i_cap = Cdl * ν (with scan-direction detection)
# - Robust loss options (default: Log L2 for accuracy)
# - Feature-preserving thinning with Ecorr/Epp/Eb/Esp anchors
# - Adaptive optimization: DE + L-BFGS-B (+ optional NM), local restarts, fallback
# - Final full-data refine
# - Demo fitting removed

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import linregress
from itertools import groupby
import warnings, io, re, time

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Tafel Fitting Tool", page_icon="⚡", layout="wide")
st.markdown("""<style>
body,[data-testid="stAppViewContainer"]{background:#0f0f1a;color:#cdd6f4}
[data-testid="stSidebar"]{background:#1a1a2e}
section[data-testid="stFileUploadDropzone"]{background:#1e1e2e!important;
  border:2px dashed #45475a!important;border-radius:12px!important}
.pc{background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:14px 16px;margin:4px 0}
.pl{color:#a6adc8;font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.pv{font-size:21px;font-weight:700;margin:1px 0}
.pu{color:#585b70;font-size:11px}
.sh{color:#89b4fa;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  border-bottom:1px solid #313244;padding-bottom:5px;margin:14px 0 6px}
.bd{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;margin:1px 2px}
.bg{background:#1c3a2f;color:#a6e3a1;border:1px solid #a6e3a1}
.bb{background:#1a2a3f;color:#89b4fa;border:1px solid #89b4fa}
.by{background:#3a3020;color:#f9e2af;border:1px solid #f9e2af}
.br{background:#3a1a20;color:#f38ba8;border:1px solid #f38ba8}
.bp{background:#2a1a3a;color:#cba6f7;border:1px solid #cba6f7}
.bx{background:#1e1e2e;border-radius:0 8px 8px 0;padding:8px 14px;margin:6px 0;font-size:12px}
.bok{border-left:4px solid #a6e3a1;color:#cdd6f4}
.bwn{border-left:4px solid #f9e2af;color:#f9e2af}
.ber{border-left:4px solid #f38ba8;color:#f38ba8}
.tb{background:linear-gradient(135deg,#1e1e2e,#232336);border:1px solid #45475a;
  border-radius:10px;padding:14px 18px;margin:8px 0}
</style>""", unsafe_allow_html=True)

CL = dict(data="#89b4fa",anodic="#f9e2af",cathodic="#cba6f7",cat2="#f5c2e7",
    fit="#a6e3a1",passive="rgba(166,227,161,0.10)",limiting="rgba(137,220,235,0.10)",
    tp="rgba(243,188,168,0.08)",sp="rgba(203,166,247,0.08)",
    ecorr="#f38ba8",grid="#313244",bg="#1e1e2e",paper="#131320",tx="#cdd6f4")

# ================================================================
# FILE I/O
# ================================================================
COL_SIG = [
    (r"we.*potential",r"we.*current","A"),(r"ewe",r"i/ma","mA"),
    (r"ewe",r"<i>/ma","mA"),(r"^vf$",r"^im$","A"),
    (r"potential/v",r"current/a","A"),(r"e/v",r"i/a","A"),
    (r"potential|volt|^e$|e \(v\)|e_v",r"current|amps|^i$|i \(a\)|i_a","A"),
    (r"potential|volt|^e$",r"current.*ma|ima","mA"),
]
UHINT={r"\(a\)|_a$|/a$":1.0,r"\(ma\)|_ma$|/ma$":1e-3,
       r"\(ua\)|_ua$|/ua$":1e-6,r"a/cm":1.0,r"ma/cm":1e-3}

def auto_cols(df):
    cl={c:c.lower().strip() for c in df.columns}
    num=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for Ep,Ip,u in COL_SIG:
        em=[c for c,v in cl.items() if re.search(Ep,v) and c in num]
        im=[c for c,v in cl.items() if re.search(Ip,v) and c in num and c not in em]
        if em and im:
            ec=sorted(em,key=lambda c:0 if "we" in c.lower() else 1)[0]
            ic=im[0]; f=1e-3 if u=="mA" else 1.0
            for p,fv in UHINT.items():
                if re.search(p,cl[ic]): f=fv; break
            return ec,ic,f
    if len(num)>=2: return num[0],num[1],1.0
    raise ValueError("Cannot detect columns.")

def load_file(f):
    nm=f.name.lower()
    raw=f.read(); f.seek(0)
    if nm.endswith((".xlsx",".xls")): return pd.read_excel(io.BytesIO(raw))
    text=raw.decode("utf-8",errors="replace"); lines=text.splitlines(); skip=0
    for idx,line in enumerate(lines):
        pts=re.split(r"[,;\t ]+",line.strip())
        if sum(1 for p in pts if re.match(r"^-?[\d.eE+]+$",p))>=2:
            skip=max(0,idx-1) if idx>0 and not any(
                re.match(r"^-?[\d.eE+]+$",p)
                for p in re.split(r"[,;\t ]+",lines[idx-1].strip())) else idx
            break
    for sep in ["\t",";",",",r"\s+"]:
        try:
            df=pd.read_csv(io.StringIO(text),sep=sep,skiprows=skip,engine="python")
            if df.shape[1]>=2 and df.shape[0]>5: return df.dropna(axis=1,how="all")
        except: pass
    raise ValueError(f"Cannot parse {f.name}")

# ================================================================
# MATH & HELPERS
# ================================================================
def slog(x): return np.log10(np.maximum(np.abs(x),1e-20))
def sm(y,w=11,p=3):
    n=len(y); w=min(w,n if n%2==1 else n-1)
    return savgol_filter(y,w,min(p,w-1)) if w>=5 else y.copy()
def r2(yt,yp):
    sr=np.sum((yt-yp)**2); st_=np.sum((yt-yt.mean())**2)
    return float(max(0,1-sr/st_)) if st_>0 else 0.0
def sig(x,k=1.0): return 1.0/(1.0+np.exp(-np.clip(k*x,-50,50)))

def scan_direction_sign(E):
    """Sign of dE/dt per point."""
    if len(E) < 2:
        return np.ones_like(E)
    dE = np.diff(E)
    s = np.zeros_like(E, dtype=float)
    sgn = np.sign(dE)
    last = 1.0
    for k in range(len(E)-1):
        if sgn[k] == 0:
            s[k] = last
        else:
            last = sgn[k]
            s[k] = last
    s[-1] = last
    s[s==0] = 1.0
    return s

def feature_preserving_thin(E, i, reg, max_points=1200):
    """
    Adaptive thinning with anchors at detected features: Ecorr index, p1/p2 (passive windows),
    Eb, Esp. Keeps endpoints and distributes by curvature in log-space.
    Returns (E_fit, i_fit, idx).
    """
    n=len(E)
    if n <= max_points:
        return E, i, np.arange(n)

    E = np.asarray(E); i = np.asarray(i)
    lg = slog(i)
    g = np.gradient(lg, E)
    c = np.gradient(g, E)

    # base weights: curvature and slope
    w = 1.0 + 4.0*np.abs(g) + 8.0*np.abs(c)

    # anchor centers and half-windows (in volts)
    anchors_idx = set([0, n-1])
    windows = []
    if "eci" in reg:
        k = int(np.clip(reg["eci"],0,n-1)); anchors_idx.add(k); windows.append((E[k], 0.06))
    if "p1" in reg:
        ps, pe = reg["p1"]["ps"], reg["p1"]["pe"]
        anchors_idx.update([int(np.clip(ps,0,n-1)), int(np.clip(pe,0,n-1))])
        windows.append((reg["p1"]["Es"], 0.08))
    if "Eb" in reg:
        k = int(np.argmin(np.abs(E - reg["Eb"]))); anchors_idx.add(k); windows.append((reg["Eb"], 0.06))
    if "p2" in reg:
        ps, pe = reg["p2"]["ps"], reg["p2"]["pe"]
        anchors_idx.update([int(np.clip(ps,0,n-1)), int(np.clip(pe,0,n-1))])
        windows.append((reg["p2"]["Es"], 0.08))

    # boost weights in windows and around anchor indices
    for E0, hw in windows:
        m = (E >= E0 - hw) & (E <= E0 + hw)
        w[m] *= 12.0
    for a in list(anchors_idx):
        lo = max(0, a-8); hi = min(n-1, a+8)
        w[lo:hi+1] *= 6.0

    s = np.cumsum(w)
    if not np.isfinite(s[-1]) or s[-1] <= 0:
        idx = np.unique(np.linspace(0, n-1, max_points).astype(int))
        return E[idx], i[idx], idx
    s = (s - s[0]) / (s[-1] - s[0] + 1e-12)
    xs = np.linspace(0, 1, max_points)
    idx = np.unique(np.searchsorted(s, xs))

    # enforce anchor indices
    idx = np.unique(np.concatenate([idx, np.array(sorted(list(anchors_idx)), dtype=int)]))
    idx[0]=0; idx[-1]=n-1
    return E[idx], i[idx], idx

# ================================================================
# GLOBAL MODEL  (17 parameters: + Rs)
# ================================================================
PN = ["Ecorr","icorr","ba","bc1","iL",
      "i0_c2","bc2",
      "Epp","k_pass","ipass",
      "Eb","a_tp","b_tp",
      "Esp","k_sp","ipass2",
      "Rs"]
NP = 17
LOG_I = {1,4,5,9,11,15}  # currents (icorr, iL, i0_c2, ipass, a_tp, ipass2)

def _prepare_i_cap(E, i_cap):
    E = np.asarray(E, dtype=float)
    if i_cap is None:
        return np.zeros_like(E)
    arr = np.asarray(i_cap, dtype=float)
    if arr.shape != E.shape:
        if arr.size == 1:
            return np.full_like(E, float(arr))
        return np.zeros_like(E)
    return arr

def gmodel(E, p, i_cap=None):
    """Global polarization model with ohmic drop and optional capacitive current.
    Iterations: 1 if Rs≈0, else 6 for accuracy."""
    Ecorr,icorr,ba,bc1,iL = p[0],p[1],p[2],p[3],p[4]
    i0_c2,bc2 = p[5],p[6]
    Epp,k_pass,ipass = p[7],p[8],p[9]
    Eb,a_tp,b_tp = p[10],p[11],p[12]
    Esp,k_sp,ipass2 = p[13],p[14],p[15]
    Rs = max(p[16], 0.0)

    E = np.asarray(E, dtype=float)
    i_cap_arr = _prepare_i_cap(E, i_cap)

    n_it = 1 if Rs <= 1e-4 else 6
    E_eff = E.copy()
    i_net = np.zeros_like(E)

    for _ in range(n_it):
        eta = E_eff - Ecorr

        # Cathodic 1: O2 reduction (diffusion-limited)
        ic1k = icorr * np.exp(np.clip(-2.303*eta/np.maximum(bc1,1e-12), -50, 50))
        i_c1 = ic1k / (1.0 + np.where(iL>0, ic1k/np.maximum(iL,1e-20), 0.0))

        # Cathodic 2: H2 evolution (activation)
        i_c2 = i0_c2 * np.exp(np.clip(-2.303*eta/np.maximum(bc2,1e-12), -50, 50))

        # Anodic: activation
        i_act = icorr * np.exp(np.clip(2.303*eta/np.maximum(ba,1e-12), -50, 50))

        # Primary passivation film θ1
        t1 = sig(E_eff - Epp, k_pass)
        i_p1 = i_act*(1.0 - t1) + ipass*t1

        # Transpassive
        i_tp = a_tp * np.exp(np.clip(b_tp*(E_eff - Eb), -50, 50)) * sig(E_eff - Eb, 40.0)

        # Secondary passivation film θ2
        t2 = sig(E_eff - Esp, k_sp)
        i_an = (i_p1 + i_tp)*(1.0 - t2) + ipass2*t2

        # Net faradaic + capacitive
        i_net = i_an - (i_c1 + i_c2) + i_cap_arr

        # Update effective potential with ohmic drop
        if Rs > 0:
            E_eff = E - i_net * Rs

    return i_net

def gcomp(E, p, i_cap=None):
    """Return components with ohmic drop and optional capacitive current; robust to i_cap shape."""
    Ecorr,icorr,ba,bc1,iL = p[0:5]
    i0_c2,bc2 = p[5:7]
    Epp,k_pass,ipass = p[7:10]
    Eb,a_tp,b_tp = p[10:13]
    Esp,k_sp,ipass2 = p[13:16]
    Rs = max(p[16], 0.0)

    E = np.asarray(E, dtype=float)
    i_cap_arr = _prepare_i_cap(E, i_cap)
    n_it = 1 if Rs <= 1e-4 else 6

    E_eff = E.copy()
    for _ in range(n_it):
        eta = E_eff - Ecorr
        ic1k = icorr*np.exp(np.clip(-2.303*eta/np.maximum(bc1,1e-12), -50, 50))
        i_c1 = ic1k/(1.0 + np.where(iL>0, ic1k/np.maximum(iL,1e-20), 0.0))
        i_c2 = i0_c2*np.exp(np.clip(-2.303*eta/np.maximum(bc2,1e-12), -50, 50))
        i_act = icorr*np.exp(np.clip(2.303*eta/np.maximum(ba,1e-12), -50, 50))
        t1 = sig(E_eff - Epp, k_pass); i_p1 = i_act*(1.0 - t1) + ipass*t1
        i_tp = a_tp*np.exp(np.clip(b_tp*(E_eff - Eb), -50, 50))*sig(E_eff - Eb, 40.0)
        t2 = sig(E_eff - Esp, k_sp); i_an = (i_p1 + i_tp)*(1.0 - t2) + ipass2*t2
        i_net = i_an - (i_c1 + i_c2) + i_cap_arr
        if Rs > 0:
            E_eff = E - i_net * Rs

    return dict(ic1=i_c1, ic2=i_c2, iact=i_act, ip1=i_p1, itp=i_tp,
                ian=i_an, t1=t1, t2=t2, itot=i_net)

# ================================================================
# CURVE TYPE CLASSIFICATION & ADAPTIVE PARAMETER MASKS
# ================================================================
class CT:
    ACTIVE="A"; ACTIVE_D="AD"; ACTIVE_H2="AH"
    PASSIVE="P"; PASSIVE_D="PD"
    PASS_TP="PT"; PASS_TP_SP="PTS"
    PITTING="PP"; FULL="F"

    INFO = {
        "A":  ("Active Only","Pure Butler-Volmer kinetics.", [0,1,2,3]),
        "AD": ("Active + Diffusion-Limited",
               "Active kinetics with O₂ cathodic diffusion limitation.",
               [0,1,2,3,4]),
        "AH": ("Active + Dual Cathodic",
               "Active kinetics with O₂ diffusion + H₂ evolution.",
               [0,1,2,3,4,5,6]),
        "P":  ("Active–Passive",
               "Active dissolution peak (nose) → passive plateau.",
               [0,1,2,3,7,8,9]),
        "PD": ("Active–Passive + Diffusion",
               "Passivating system with cathodic diffusion limit.",
               [0,1,2,3,4,5,6,7,8,9]),
        "PT": ("Passive–Transpassive",
               "Passive plateau + transpassive dissolution above Eb.",
               [0,1,2,3,4,5,6,7,8,9,10,11,12]),
        "PTS":("Full: + Secondary Passivity",
               "Active→passive→transpassive→secondary passivity.",
               list(range(16))),
        "PP": ("Passive + Pitting",
               "Passive with sharp pitting breakdown at Epit.",
               [0,1,2,3,4,5,6,7,8,9,10,11,12]),
        "F":  ("Full Multi-Region","All regions detected.",
               list(range(16))),
    }

    @staticmethod
    def free_idx(ct):
        return CT.INFO.get(ct,("","",list(range(16))))[2]

    @staticmethod
    def nfree(ct):
        return len(CT.free_idx(ct))

# ================================================================
# REGION DETECTION (sorted copy for stable gradients)
# ================================================================
def detect(E, i):
    idx_sort = np.argsort(E)
    Es = E[idx_sort]; is_ = i[idx_sort]
    reg={}; n=len(Es); ai=np.abs(is_)

    # Ecorr
    sc=np.where(np.diff(np.sign(is_)))[0]
    if len(sc)>0:
        k=sc[0]; d=is_[k+1]-is_[k]
        reg["Ecorr"]=float(Es[k]-is_[k]*(Es[k+1]-Es[k])/d) if abs(d)>0 else float(Es[k])
        reg["eci"]=int(idx_sort[k])
    else:
        k=int(np.argmin(ai)); reg["Ecorr"]=float(Es[k]); reg["eci"]=int(idx_sort[k])
    Ec=reg["Ecorr"]

    # Cathodic
    ci=np.where(Es<Ec)[0]
    if len(ci)>=8:
        lc=sm(slog(ai[ci]),min(15,(len(ci)//2)*2-1 or 5))
        dl=np.abs(np.gradient(lc,Es[ci]))
        thr=np.percentile(dl,20); flat=dl<max(thr,0.5)
        runs=[(k2,list(g)) for k2,g in groupby(enumerate(flat),key=lambda x:x[1]) if k2]
        if runs:
            br=max(runs,key=lambda x:len(x[1])); idxs=[s[0] for s in br[1]]
            er=abs(Es[ci[idxs[-1]]]-Es[ci[idxs[0]]])
            if len(idxs)>=4 and er>0.03 and abs(Es[ci[idxs[0]]]-Ec)>0.06:
                reg.update(iL=float(np.median(ai[ci[idxs]])),
                           Els=float(Es[ci[idxs[0]]]),Ele=float(Es[ci[idxs[-1]]]))
        # H2 evolution heuristic
        if "iL" in reg:
            very_neg = ci[Es[ci] < reg["Ele"]-0.02]
            if len(very_neg) >= 5:
                lc_neg = slog(ai[very_neg])
                grad_neg = np.gradient(lc_neg, Es[very_neg])
                if np.mean(grad_neg[:min(10,len(grad_neg))]) < -1.0:
                    reg["has_H2"] = True
                    try:
                        s,b,rv,*_ = linregress(Es[very_neg[:min(15,len(very_neg))]],
                                               lc_neg[:min(15,len(very_neg))])
                        if s < 0 and abs(1/s) > 0.05:
                            reg["bc2_est"] = abs(1/s)
                            reg["i0_c2_est"] = 10**(b + s*Ec)
                    except: pass
        else:
            if len(ci) >= 20:
                half = len(ci)//2
                try:
                    s1,_,r1,*_=linregress(Es[ci[:half]],lc[:half])
                    s2,_,r2v,*_=linregress(Es[ci[half:]],lc[half:])
                    if r1**2>0.85 and r2v**2>0.85 and abs(s1-s2)>2:
                        reg["has_H2"]=True
                except: pass

    # Anodic
    ani=np.where(Es>Ec)[0]
    if len(ani)<6:
        if "iL" in reg:
            ct = CT.ACTIVE_H2 if reg.get("has_H2") else CT.ACTIVE_D
        else:
            ct = CT.ACTIVE
        reg["ct"]=ct; return reg

    log_a=sm(slog(ai[ani]),min(15,(len(ani)//2)*2-1 or 5))
    dlog=np.gradient(log_a,Es[ani]); adl=np.abs(dlog)

    # Peaks
    peaks=[]
    if len(ani)>20:
        ism=sm(ai[ani],min(21,(len(ani)//2)*2-1 or 5))
        order=max(3,len(ani)//20)
        pki=argrelextrema(slog(ism),np.greater,order=order)[0]
        for pk in pki:
            rest=slog(ism)[pk:]
            if len(rest)>5:
                prom=slog(ism)[pk]-np.min(rest[3:])
                if prom>0.3: peaks.append(dict(idx=int(idx_sort[ani[pk]]),
                                                E=float(Es[ani[pk]]),prom=float(prom)))
    reg["peaks"]=peaks

    # Passive
    thr_p=np.percentile(adl,25); flat=adl<max(thr_p,0.8)
    runs=[(k2,list(g)) for k2,g in groupby(enumerate(flat),key=lambda x:x[1]) if k2]
    pr=[]
    for _,ri in runs:
        idxs=[s[0] for s in ri]; er=abs(Es[ani[idxs[-1]]]-Es[ani[idxs[0]]])
        if len(idxs)>=4 and er>0.03:
            ps,pe=ani[idxs[0]],ani[idxs[-1]]
            im=float(np.median(ai[ps:pe+1]))
            pre=np.where((Es>Ec)&(Es<Es[ps]))[0]
            valid=(len(pre)>2 and im<np.max(ai[pre])*0.5) or er>0.08
            if valid: pr.append(dict(ps=int(idx_sort[ps]),pe=int(idx_sort[pe]),
                                     Es=float(Es[ps]),Ee=float(Es[pe]),ip=im,rng=er))
    if pr:
        reg["p1"]=pr[0]; reg["Epp"]=pr[0]["Es"]; reg["ipass"]=pr[0]["ip"]
    if len(pr)>=2:
        reg["p2"]=pr[1]; reg["Esp"]=pr[1]["Es"]; reg["ipass2"]=pr[1]["ip"]

    # Breakdown
    if "p1" in reg:
        pe_sort = np.where(Es == reg["p1"]["Ee"])[0]
        if len(pe_sort)>0 and pe_sort[0]+5<n:
            pe = pe_sort[0]
            da=np.gradient(slog(ai[pe:]),Es[pe:])
            thr_b=np.percentile(np.abs(da),75)
            jump=np.where(np.abs(da)>max(thr_b,2.0))[0]
            if len(jump):
                reg["Eb"]=float(Es[pe+jump[0]])
                reg["is_pit"]=np.mean(np.abs(da[jump[:min(5,len(jump))]])
                    )>10 if len(jump)>2 else False

    # Classify
    hd,hh,hp,ht,hsp="iL" in reg,reg.get("has_H2",False),"p1" in reg,"Eb" in reg,"p2" in reg
    pit=reg.get("is_pit",False)
    if hsp:          ct=CT.PASS_TP_SP if ht else CT.FULL
    elif ht and pit: ct=CT.PITTING
    elif ht:         ct=CT.PASS_TP if not hd else CT.FULL
    elif hp:         ct=CT.PASSIVE_D if hd else CT.PASSIVE
    elif hd and hh:  ct=CT.ACTIVE_H2
    elif hd:         ct=CT.ACTIVE_D
    else:            ct=CT.ACTIVE
    reg["ct"]=ct
    return reg

# ================================================================
# INITIAL GUESS
# ================================================================
def init_guess(E, i, reg):
    Ec=reg["Ecorr"]; ai=np.abs(i); lg=slog(i)
    ba0,bc0,ic0=0.060,0.120,max(ai[reg["eci"]],1e-12)

    Epp=reg.get("Epp",Ec+0.50)
    for lo in np.arange(0.005,0.05,0.005):
        for hi in np.arange(lo+0.02,min(lo+0.18,Epp-Ec-0.005),0.005):
            m=(E>Ec+lo)&(E<Ec+hi)
            if m.sum()<4: continue
            s,b,rv,*_=linregress(E[m],lg[m])
            if s>0 and 20<1000/s<500 and rv**2>0.85: ba0=1/s; break
        else: continue
        break

    Ele=reg.get("Ele")
    for lo in np.arange(0.005,0.08,0.005):
        for hi in np.arange(lo+0.02,lo+0.25,0.005):
            m=(E<Ec-lo)&(E>Ec-hi)
            if Ele is not None: m=m&(E>Ele+0.005)
            if m.sum()<4: continue
            s,b,rv,*_=linregress(E[m],lg[m])
            if s<0 and 20<-1000/s<500 and rv**2>0.85: bc0=-1/s; break
        else: continue
        break

    Emax=float(np.max(E))
    return np.array([
        Ec,                                         # 0  Ecorr
        ic0,                                        # 1  icorr
        ba0,                                        # 2  ba
        bc0,                                        # 3  bc1
        reg.get("iL",1e2),                          # 4  iL
        reg.get("i0_c2_est",ic0*1e-3),              # 5  i0_c2
        reg.get("bc2_est",0.150),                   # 6  bc2
        reg.get("Epp",Emax+10),                     # 7  Epp
        40.0 if "p1" in reg else 50.0,              # 8  k_pass
        reg.get("ipass",ic0),                       # 9  ipass
        reg.get("Eb",Emax+10),                      # 10 Eb
        reg.get("ipass",ic0)*0.01 if "Eb" in reg else 1e-30, # 11 a_tp
        8.0,                                        # 12 b_tp
        reg.get("Esp",Emax+20),                     # 13 Esp
        50.0,                                       # 14 k_sp
        reg.get("ipass2",ic0),                      # 15 ipass2
        0.0                                         # 16 Rs (Ω·cm²)
    ])

# ================================================================
# OPTIMIZER (DE + L-BFGS-B + optional NM) with restarts
# ================================================================
class Optimizer:
    def __init__(self, E, i, reg, p0, i_cap_vec, fit_rs, rs_bounds, loss_cfg, opt_cfg):
        self.E=E
        self.i=i
        self.ld=slog(i)
        self.i_cap_vec = i_cap_vec
        self.reg=reg
        self.ct=reg["ct"]
        self.pfull=p0.copy()

        base_idx = CT.free_idx(self.ct)
        self.fidx = base_idx + ([16] if fit_rs else [])
        self.nf=len(self.fidx)

        self.loss_cfg = loss_cfg
        self.opt_cfg = opt_cfg  # dict: use_de, de_pop, de_iter, lbfgs_maxiter, use_nm, nm_maxiter, restarts, jitter

        self.log=[]; self.best_p=p0.copy(); self.best_s=1e30
        self._bounds(rs_bounds)

    def _bounds(self, rs_bounds):
        p=self.pfull; reg=self.reg; Ec=p[0]; ic=max(p[1],1e-14)
        Emax=float(np.max(self.E)); rs_lo, rs_hi = rs_bounds
        self.lo=np.array([
            Ec-0.20, max(ic*1e-4,1e-15), 0.010, 0.010,
            max(reg.get("iL",1e-6)*0.01,1e-10),
            max(ic*1e-6,1e-16), 0.04,
            p[7]-0.25 if "p1" in reg else Emax+5,
            5.0, max(p[9]*0.01 if "p1" in reg else ic*0.01,1e-14),
            p[10]-0.25 if "Eb" in reg else Emax+5,
            max(p[11]*1e-4 if "Eb" in reg else 1e-30,1e-30), 0.5,
            p[13]-0.25 if "p2" in reg else Emax+10,
            5.0, max(p[15]*0.01 if "p2" in reg else 1e-14,1e-14),
            max(rs_lo, 0.0)  # Rs
        ])
        self.hi=np.array([
            Ec+0.20, min(ic*1e4,1e0), 0.500, 0.500,
            max(reg.get("iL",1e0)*100,1e-2),
            max(ic*10,1e-6), 0.350,
            p[7]+0.25 if "p1" in reg else Emax+15,
            200.0, max(p[9]*100 if "p1" in reg else ic*100,1e-4),
            p[10]+0.25 if "Eb" in reg else Emax+15,
            max(p[11]*1e4 if "Eb" in reg else 1e-20,1e-10), 40.0,
            p[13]+0.25 if "p2" in reg else Emax+25,
            200.0, max(p[15]*100 if "p2" in reg else 1e-4,1e-4),
            max(rs_hi, 0.0)  # Rs
        ])
        for j in range(NP):
            if self.lo[j]>=self.hi[j]:
                m=self.pfull[j]; self.lo[j]=m-abs(m)*0.5-1e-6; self.hi[j]=m+abs(m)*0.5+1e-6

    def _pack(self, pf):
        return np.array([np.log10(max(pf[j],1e-15)) if j in LOG_I else pf[j] for j in self.fidx])
    def _unpack(self, x):
        p=self.pfull.copy()
        for k,j in enumerate(self.fidx):
            p[j]=10**x[k] if j in LOG_I else x[k]
        return p
    def _pbounds(self):
        return [(np.log10(max(self.lo[j],1e-15)),np.log10(max(self.hi[j],1e-14)))
                if j in LOG_I else (self.lo[j],self.hi[j]) for j in self.fidx]

    def _loss(self, p):
        try:
            im = gmodel(self.E, p, i_cap=self.i_cap_vec)
        except Exception:
            return 1e30
        ld_m = slog(im)
        res_log = self.ld - ld_m

        lt = self.loss_cfg.get("type","log_l2")
        if lt == "log_l2":
            return float(np.sum(res_log**2))
        elif lt == "hybrid":
            alpha = float(self.loss_cfg.get("alpha", 0.7))
            scale = float(self.loss_cfg.get("linear_scale", np.median(np.abs(self.i))+1e-12))
            res_lin = (self.i - im) / (scale if scale>0 else 1.0)
            return float(alpha*np.sum(res_log**2) + (1.0-alpha)*np.sum(res_lin**2))
        elif lt == "huber_log":
            delta = float(self.loss_cfg.get("delta", 0.3))
            ares = np.abs(res_log)
            quad = ares <= delta
            loss = np.where(quad, 0.5*res_log**2, delta*(ares - 0.5*delta))
            return float(np.sum(loss))
        else:
            return float(np.sum(res_log**2))

    def _obj(self, x):
        p=self._unpack(x)
        return self._loss(p)

    def _up(self, x, tag):
        s=self._obj(x)
        try:
            rv=r2(self.ld,slog(gmodel(self.E,self._unpack(x), i_cap=self.i_cap_vec)))
        except Exception:
            rv=0.0
        if s<self.best_s:
            self.best_p=self._unpack(x); self.best_s=s; self.log.append(f"  {tag}: R²={rv:.6f}")
            return True
        self.log.append(f"  {tag}: R²={rv:.6f} (no impr.)")
        return False

    def _local_restart(self, base_x, label, maxiter=1500, jitter=0.06):
        bnds = self._pbounds()
        width = np.array([b[1]-b[0] for b in bnds])
        x0 = base_x + np.random.uniform(-jitter, jitter, size=len(base_x)) * width
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
        try:
            t1=time.time()
            r = minimize(self._obj, x0, method="L-BFGS-B", bounds=bnds,
                         options={"maxiter":int(maxiter),"ftol":1e-12,"gtol":1e-10})
            self._up(r.x, f"{label} ({time.time()-t1:.1f}s)")
        except Exception as ex:
            self.log.append(f"  {label} err: {ex}")

    def run(self):
        t0=time.time()
        title,_,_=CT.INFO.get(self.ct,("Unknown","",[]))
        self.log.append(f"Curve: {title}  ({self.nf} free params)")
        x0=self._pack(self.pfull); self.best_s=self._obj(x0)
        try:
            rv0=r2(self.ld,slog(gmodel(self.E,self.pfull, i_cap=self.i_cap_vec)))
        except Exception:
            rv0=0.0
        self.log.append(f"  Init: R²={rv0:.4f}")
        bnds=self._pbounds()

        # Stage 1: DE (optional)
        if self.opt_cfg.get("use_de", True):
            ps=self.opt_cfg.get("de_pop", max(10, min(12, self.nf)))
            mi=self.opt_cfg.get("de_iter", 300)
            self.log.append(f"Stage 1: DE  pop={ps} iter={mi}")
            try:
                t1=time.time()
                res=differential_evolution(
                    self._obj, bnds, seed=42, maxiter=mi, tol=1e-9,
                    popsize=ps, mutation=(0.5,1.0), recombination=0.85, polish=False
                )
                self._up(res.x,f"DE ({time.time()-t1:.1f}s)")
            except Exception as ex:
                self.log.append(f"  DE err: {ex}")
        else:
            self.log.append("Stage 1: DE skipped")

        # Stage 2: L-BFGS-B (local)
        self.log.append("Stage 2: L-BFGS-B")
        try:
            t1=time.time()
            r2v=minimize(
                self._obj,self._pack(self.best_p),method="L-BFGS-B",bounds=bnds,
                options={"maxiter":int(self.opt_cfg.get("lbfgs_maxiter", 2000)),"ftol":1e-12,"gtol":1e-10}
            )
            self._up(r2v.x,f"L-BFGS-B ({time.time()-t1:.1f}s)")
        except Exception as ex:
            self.log.append(f"  L-BFGS-B err: {ex}")

        # Stage 3: Nelder-Mead (optional)
        if self.opt_cfg.get("use_nm", True):
            self.log.append("Stage 3: Nelder-Mead")
            try:
                t1=time.time()
                r3=minimize(self._obj,self._pack(self.best_p),method="Nelder-Mead",
                    options={"maxiter":int(self.opt_cfg.get("nm_maxiter", 7000)),"xatol":1e-12,"fatol":1e-12,"adaptive":True})
                self._up(r3.x,f"NM ({time.time()-t1:.1f}s)")
            except Exception as ex:
                self.log.append(f"  NM err: {ex}")
        else:
            self.log.append("Stage 3: NM skipped")

        # Local restarts (multi-start)
        restarts = int(self.opt_cfg.get("restarts", 3))
        jitter = float(self.opt_cfg.get("jitter", 0.06))
        if restarts > 0:
            bx = self._pack(self.best_p)
            for k in range(restarts):
                self._local_restart(bx, f"Restart {k+1}", maxiter=self.opt_cfg.get("lbfgs_maxiter", 2000), jitter=jitter)

        dt=time.time()-t0
        try:
            rv=r2(self.ld,slog(gmodel(self.E,self.best_p, i_cap=self.i_cap_vec)))
        except Exception:
            rv=0.0
        q="Excellent" if rv>=0.995 else "Good" if rv>=0.97 else "Acceptable" if rv>=0.90 else "Poor"
        self.log.append(f"Result: {q}  R²(log)={rv:.6f}  [{dt:.1f}s]")
        return self.best_p, rv

# ================================================================
# DIAGNOSTICS
# ================================================================
def diagnose(E, i, reg, bp, rv, i_cap_vec):
    issues=[]; ai=np.abs(i); lg=slog(i); n=len(E); Ec=reg["Ecorr"]
    if n>20:
        lsm=sm(lg,min(21,(n//4)*2-1 or 5)); noise=np.std(lg-lsm)
        if noise>0.5: issues.append(("High noise",f"σ={noise:.2f} dec. Use slower scan rate or average scans.","e"))
        elif noise>0.15: issues.append(("Moderate noise",f"σ={noise:.2f} dec.","w"))
        else: issues.append(("Low noise",f"σ={noise:.3f} dec.","o"))
    if bp is not None:
        if bp[2]*1000>200: issues.append(("Large βa",f"{bp[2]*1000:.0f} mV/dec — possible IR drop or multi-step mechanism.","w"))
        if bp[3]*1000>200: issues.append(("Large βc₁",f"{bp[3]*1000:.0f} mV/dec — possible diffusion or IR drop.","w"))
        if bp[6]*1000>250: issues.append(("Large βc₂ (H₂)",f"{bp[6]*1000:.0f} mV/dec — may indicate coupled reactions.","w"))
        if bp[16] > 0.0:
            issues.append(("Uncompensated resistance", f"Rs ≈ {bp[16]:.2f} Ω·cm² — IR drop correction applied.", "o"))
    Er=float(np.max(E)-np.min(E))
    if Er<0.3: issues.append(("Narrow scan",f"{Er*1000:.0f} mV. May miss regions.","w"))
    if n/max(Er,0.01)<50: issues.append(("Low density",f"{n/max(Er,0.01):.0f} pts/V.","w"))
    if np.sum(np.abs(E-Ec)<0.05)<5: issues.append(("Sparse near Ecorr","<5 pts within ±50 mV.","w"))
    if bp is not None:
        res=lg-slog(gmodel(E,bp, i_cap=i_cap_vec))
        if n>10:
            nruns=np.sum(np.abs(np.diff(np.sign(res)))>0)+1
            if nruns<n*0.25: issues.append(("Systematic misfit",
                f"{nruns} sign changes (expect ~{n//2}). Model may miss a feature.","w"))
        masks = [
            ("near Ecorr",np.abs(E-Ec)<0.05),
            ("cathodic",E<Ec-0.1),
            ("passive",(E>reg.get("Epp",99))&(E<reg.get("Eb",E[-1])) if bp is not None else np.zeros_like(E,dtype=bool))
        ]
        for name,mask in masks:
            if mask.sum()>3:
                rmse=np.sqrt(np.mean((res[mask])**2))
                if rmse>0.5: issues.append((f"Poor fit: {name}",f"RMSE={rmse:.3f} dec.","w"))
        if bp[1]>0.1: issues.append(("Very high icorr",f"{bp[1]:.2e} — check area/units.","e"))
        if bp[1]<1e-14: issues.append(("Very low icorr",f"{bp[1]:.2e}.","w"))
    if rv is not None and rv<0.90: issues.append(("Poor fit",
        f"R²(log)={rv:.4f}. Possible: coupled reactions, adsorption, roughness, artifacts.","e"))
    return issues

# ================================================================
# PLOTS
# ================================================================
def plot_main(E, i, bp, reg, ct, i_cap_vec):
    lg=slog(i); fig=go.Figure()
    if "p1" in reg:
        p=reg["p1"]; fig.add_vrect(x0=p["Es"],x1=p["Ee"],fillcolor=CL["passive"],layer="below",line_width=0,
            annotation=dict(text="Passive",font=dict(color="#a6e3a1",size=11)))
    if "p2" in reg:
        p=reg["p2"]; fig.add_vrect(x0=p["Es"],x1=p["Ee"],fillcolor=CL["sp"],layer="below",line_width=0,
            annotation=dict(text="2nd Passive",font=dict(color="#cba6f7",size=10)))
    if "Els" in reg:
        fig.add_vrect(x0=reg["Els"],x1=reg["Ele"],fillcolor=CL["limiting"],layer="below",line_width=0,
            annotation=dict(text="Limiting",font=dict(color="#89dceb",size=11)))
    if reg.get("Eb"):
        te=reg.get("Esp",E[-1]) if "p2" in reg else E[-1]
        fig.add_vrect(x0=reg["Eb"],x1=te,fillcolor=CL["tp"],layer="below",line_width=0,
            annotation=dict(text="Transpassive",font=dict(color="#fab387",size=10)))
    Ec=reg["Ecorr"]
    fig.add_vline(x=Ec,line=dict(color=CL["ecorr"],width=1.5,dash="dot"),
        annotation=dict(text="Ecorr",font=dict(color=CL["ecorr"],size=10)))
    if reg.get("Epp") and "p1" in reg:
        fig.add_vline(x=reg["Epp"],line=dict(color="#a6e3a1",width=1,dash="dot"),
            annotation=dict(text="Epp",font=dict(color="#a6e3a1",size=10)))
    if reg.get("Eb"):
        fig.add_vline(x=reg["Eb"],line=dict(color="#f38ba8",width=1,dash="dash"),
            annotation=dict(text="Eb",font=dict(color="#f38ba8",size=10)))
    fig.add_trace(go.Scatter(x=E,y=lg,mode="lines",name="Measured",line=dict(color=CL["data"],width=2.5)))
    if bp is not None:
        Em=np.linspace(np.min(E),np.max(E),800)
        try:
            im=gmodel(Em,bp)  # faradaic-only baseline
            rv=r2(lg,slog(gmodel(E,bp, i_cap=i_cap_vec)))
            fig.add_trace(go.Scatter(x=Em,y=slog(im),mode="lines",
                name=f"Global Fit (faradaic)  R²(log)={rv:.4f}",line=dict(color=CL["fit"],width=3)))
        except: pass
        fig.add_trace(go.Scatter(x=[bp[0]],y=[np.log10(max(bp[1],1e-20))],mode="markers",
            name=f"icorr = {bp[1]:.3e} A/cm²",
            marker=dict(symbol="x-thin",size=18,color=CL["ecorr"],line=dict(width=4,color=CL["ecorr"]))))
    fig.update_layout(template="plotly_dark",plot_bgcolor=CL["bg"],paper_bgcolor=CL["paper"],
        title=dict(text="Potentiodynamic Polarization — Global Fit",font=dict(size=17,color=CL["tx"])),
        xaxis=dict(title="Potential (V vs Ref)",gridcolor=CL["grid"],color=CL["tx"]),
        yaxis=dict(title="log₁₀|i| (A cm⁻²)",gridcolor=CL["grid"],color=CL["tx"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=CL["grid"],font=dict(color=CL["tx"],size=11),x=0.01,y=0.01),
        height=540,margin=dict(l=70,r=20,t=50,b=60),hovermode="x unified")
    return fig

def plot_comp(E, bp, ct):
    if bp is None: return None
    Em=np.linspace(np.min(E),np.max(E),500)
    c=gcomp(Em,bp)  # no i_cap for clarity
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=Em,y=slog(c["ic1"]),mode="lines",name="O₂ cathodic",
        line=dict(color=CL["cathodic"],width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=Em,y=slog(c["ic2"]),mode="lines",name="H₂ cathodic",
        line=dict(color=CL["cat2"],width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=Em,y=slog(c["iact"]),mode="lines",name="Active Tafel",
        line=dict(color=CL["anodic"],width=1,dash="dash")))
    if ct not in (CT.ACTIVE,CT.ACTIVE_D,CT.ACTIVE_H2):
        fig.add_trace(go.Scatter(x=Em,y=slog(c["ip1"]),mode="lines",name="After passivation",
            line=dict(color="#a6e3a1",width=1.5,dash="dot")))
        fig.add_trace(go.Scatter(x=Em,y=slog(c["itp"]),mode="lines",name="Transpassive",
            line=dict(color="#fab387",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=Em,y=slog(c["itot"]),mode="lines",name="Total (incl. Rs iteration, no Cdl)",
        line=dict(color=CL["fit"],width=3)))
    fig.update_layout(template="plotly_dark",plot_bgcolor=CL["bg"],paper_bgcolor=CL["paper"],
        title=dict(text="Component Breakdown",font=dict(size=15,color=CL["tx"])),
        xaxis=dict(title="Potential (V)",gridcolor=CL["grid"],color=CL["tx"]),
        yaxis=dict(title="log₁₀|i|",gridcolor=CL["grid"],color=CL["tx"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=CL["grid"],font=dict(color=CL["tx"],size=10)),
        height=420,margin=dict(l=70,r=20,t=50,b=60))
    return fig

def plot_res(E, i, bp, i_cap_vec):
    if bp is None: return None
    ld=slog(i); lp=slog(gmodel(E,bp, i_cap=i_cap_vec)); res=ld-lp; rmse=np.sqrt(np.mean(res**2))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.55,0.45],vertical_spacing=0.06,
        subplot_titles=("Overlay","Residuals (log)"))
    fig.add_trace(go.Scatter(x=E,y=ld,mode="lines",name="Measured",line=dict(color=CL["data"],width=2)),row=1,col=1)
    Em=np.linspace(np.min(E),np.max(E),600)
    fig.add_trace(go.Scatter(x=Em,y=slog(gmodel(Em,bp)),mode="lines",name="Fit (faradaic)",
        line=dict(color=CL["fit"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=E,y=res,mode="lines",name="Residual",line=dict(color="#fab387",width=1),
        fill="tozeroy",fillcolor="rgba(250,179,135,0.06)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(color="#585b70",width=1,dash="dot"),row=2)
    fig.add_annotation(text=f"RMSE(log) = {rmse:.4f}",xref="paper",yref="paper",x=0.98,y=0.38,
        showarrow=False,font=dict(color="#a6adc8",size=11),bgcolor="rgba(19,19,32,0.8)",bordercolor=CL["grid"])
    fig.update_layout(template="plotly_dark",plot_bgcolor=CL["bg"],paper_bgcolor=CL["paper"],height=440,
        margin=dict(l=70,r=20,t=40,b=60),showlegend=True,
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=CL["grid"],font=dict(color=CL["tx"],size=11)))
    fig.update_yaxes(gridcolor=CL["grid"],color=CL["tx"])
    fig.update_xaxes(gridcolor=CL["grid"],color=CL["tx"],title_text="Potential (V)",row=2)
    return fig

# ================================================================
# PARAM DISPLAY
# ================================================================
def pc(label,val,unit="",color="#cdd6f4"):
    if val is None: d="—"
    elif isinstance(val,str): d=val
    elif abs(val)<0.001 or abs(val)>9999: d=f"{val:.4e}"
    else: d=f"{val:.4f}"
    st.markdown(f'<div class="pc"><div class="pl">{label}</div>'
                f'<div class="pv" style="color:{color}">{d}</div>'
                f'<div class="pu">{unit}</div></div>',unsafe_allow_html=True)

def show_p(bp, reg, ew, rho, ct, rv, cap_cfg):
    if bp is None: return
    p=dict(zip(PN,bp)); ba=p["ba"]; bc=p["bc1"]
    B=(ba*bc)/(2.303*(ba+bc)) if ba>0 and bc>0 else None
    ic=p["icorr"]; CR=ic*3.27*ew/rho if ic else None
    bd=[]
    if "iL" in reg: bd.append('<span class="bd bb">O₂ Limiting</span>')
    if reg.get("has_H2"): bd.append('<span class="bd by">H₂ Evolution</span>')
    if "p1" in reg: bd.append('<span class="bd bg">Passive</span>')
    if reg.get("Eb"): bd.append('<span class="bd br">Breakdown</span>')
    if "p2" in reg: bd.append('<span class="bd bp">2nd Passive</span>')
    pk=reg.get("peaks",[])
    if pk: bd.append(f'<span class="bd by">{len(pk)} peak(s)</span>')
    if bd: st.markdown("**Detected:** "+"".join(bd),unsafe_allow_html=True)
    title,desc,_=CT.INFO.get(ct,("?","",[])); nf=CT.nfree(ct)
    clr={"A":"#f9e2af","AD":"#89dceb","AH":"#f5c2e7","P":"#a6e3a1","PD":"#94e2d5",
         "PT":"#fab387","PTS":"#cba6f7","PP":"#f38ba8","F":"#f5c2e7"}.get(ct,"#cdd6f4")
    st.markdown(f'<div class="tb"><div style="font-size:16px;font-weight:700;color:{clr}">{title}</div>'
                f'<div style="font-size:12px;color:#a6adc8">{desc} ({nf} core free params; Rs optional)</div></div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown('<div class="sh">Corrosion</div>',unsafe_allow_html=True)
        pc("Ecorr",p["Ecorr"],"V vs Ref","#f38ba8")
        pc("icorr",ic,"A cm⁻²","#fab387")
        if CR: pc("Corrosion rate",CR,"mm yr⁻¹","#eba0ac")
        if B: pc("B (Stern-Geary)",B*1000,"mV","#89dceb")
    with c2:
        st.markdown('<div class="sh">Kinetics</div>',unsafe_allow_html=True)
        pc("βa anodic",ba*1000,"mV dec⁻¹","#a6e3a1")
        pc("βc₁ (O₂)",bc*1000,"mV dec⁻¹","#94e2d5")
        if ct not in (CT.ACTIVE,): pc("βc₂ (H₂)",p["bc2"]*1000,"mV dec⁻¹","#f5c2e7")
        cls="bx bok" if rv and rv>=0.97 else "bx bwn"
        st.markdown(f'<div class="{cls}">R²(log) = {rv:.5f}</div>',unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="sh">Cathodic</div>',unsafe_allow_html=True)
        if ct not in (CT.ACTIVE,):
            pc("iL (O₂ diffusion)",p["iL"],"A cm⁻²","#89dceb")
            if reg.get("iL"): pc("iL (detected)",reg["iL"],"A cm⁻²","#89b4fa")
            pc("i₀,c₂ (H₂)",p["i0_c2"],"A cm⁻²","#f5c2e7")
        pc("Rs (uncompensated)",p["Rs"],"Ω·cm²","#fab387")
        if cap_cfg.get("include", False):
            pc("Cdl",cap_cfg.get("Cdl",0.0),"F cm⁻²","#cba6f7")
            pc("Scan rate ν",cap_cfg.get("nu",0.0),"V s⁻¹","#cba6f7")
    with c4:
        st.markdown('<div class="sh">Passivity & Breakdown</div>',unsafe_allow_html=True)
        if ct not in (CT.ACTIVE,CT.ACTIVE_D,CT.ACTIVE_H2):
            pc("Epp",p["Epp"],"V","#a6e3a1"); pc("ipass",p["ipass"],"A cm⁻²","#94e2d5")
            if ct in (CT.PASS_TP,CT.PITTING,CT.PASS_TP_SP,CT.FULL):
                pc("Eb",p["Eb"],"V","#f38ba8")
                pc("Eb−Ecorr",(p["Eb"]-p["Ecorr"])*1000,"mV","#f38ba8")
            if ct in (CT.PASS_TP_SP,CT.FULL):
                pc("Esp",p["Esp"],"V","#cba6f7"); pc("ipass₂",p["ipass2"],"A cm⁻²","#cba6f7")

# ================================================================
# MATERIALS & MAIN PIPELINE
# ================================================================
MATS={"Carbon Steel / Iron":(27.92,7.87),"304 Stainless Steel":(25.10,7.90),
    "316 Stainless Steel":(25.56,8.00),"Copper":(31.77,8.96),"Aluminum":(8.99,2.70),
    "Nickel":(29.36,8.91),"Titanium":(11.99,4.51),"Zinc":(32.69,7.14),"Custom":(27.92,7.87)}

def process(E_full, i_full, area, ew, rho, cap_cfg_ui, fit_rs_ui, rs_bounds, loss_cfg_ui,
            speed_mode, max_fit_points, show_components, use_full_fit, escalate_r2, restarts, jitter):
    # Book-keeping for logs
    optA = None; optB = None; optC = None; optRef = None; optRef2 = None

    # Capacitive current vector (full) from UI
    if cap_cfg_ui.get("include", False):
        sgn = scan_direction_sign(E_full)
        i_cap_vec_full = cap_cfg_ui["Cdl"] * cap_cfg_ui["nu"] * sgn
    else:
        i_cap_vec_full = np.zeros_like(E_full)

    prog=st.progress(0,text="Detecting regions (full data)...")
    reg=detect(E_full,i_full); ct=reg["ct"]

    # Stage A: Thinning (or full) — First pass without Rs and Cdl for stability
    prog.progress(12,text=f"Selecting data for fitting...")
    if use_full_fit:
        E_fit, i_fit = E_full, i_full
        i_cap_fit = np.zeros_like(E_fit)  # first pass: disable Cdl
    else:
        E_fit, i_fit, idx_fit = feature_preserving_thin(E_full, i_full, reg, max_points=max_fit_points)
        i_cap_fit = np.zeros_like(E_fit)  # first pass: disable Cdl

    # Initial guess from full data for stability
    prog.progress(20,text="Initial estimates...")
    p0=init_guess(E_full,i_full,reg)

    # Optimizer configs from speed mode
    if speed_mode == "Fast":
        opt_cfg_A = {"use_de": False, "lbfgs_maxiter": 1500, "use_nm": False,
                     "restarts": restarts, "jitter": jitter}
    elif speed_mode == "Balanced":
        opt_cfg_A = {"use_de": True, "de_pop": 10, "de_iter": 260,
                     "lbfgs_maxiter": 1800, "use_nm": False,
                     "restarts": restarts, "jitter": jitter}
    else:  # Thorough
        opt_cfg_A = {"use_de": True, "de_pop": 12, "de_iter": 360,
                     "lbfgs_maxiter": 2200, "use_nm": True, "nm_maxiter": 6000,
                     "restarts": restarts, "jitter": jitter}

    # Stage A optimization (thinned/full, Rs OFF, Cdl OFF)
    prog.progress(30,text=f"Stage A: optimizing (no Rs/Cdl)...")
    optA=Optimizer(E_fit,i_fit,reg,p0,i_cap_fit,False,rs_bounds,{"type":"log_l2"},opt_cfg_A)
    bpA,rvA=optA.run()

    # Stage B: Thinned/full refine with UI Rs/Cdl options
    prog.progress(55,text="Stage B: refine with UI Rs/Cdl...")
    fit_rs_B = fit_rs_ui
    if use_full_fit:
        i_cap_fit_B = i_cap_vec_full if cap_cfg_ui.get("include", False) else np.zeros_like(E_fit)
    else:
        i_cap_fit_B = (i_cap_vec_full[idx_fit] if cap_cfg_ui.get("include", False) else np.zeros_like(E_fit))
    opt_cfg_B = opt_cfg_A.copy()
    optB=Optimizer(E_fit,i_fit,reg,bpA,i_cap_fit_B,fit_rs_B,rs_bounds,loss_cfg_ui,opt_cfg_B)
    bpB,rvB=optB.run()

    # Stage C: Fallback escalation if needed
    if rvB < escalate_r2:
        prog.progress(72,text=f"Stage C: fallback (target R² ≥ {escalate_r2:.2f})...")
        # Add more points if thinning was used
        if not use_full_fit:
            E_fit2, i_fit2, idx_fit2 = feature_preserving_thin(E_full, i_full, reg, max_points=min(len(E_full), max_fit_points*2))
            i_cap_fit2 = i_cap_vec_full[idx_fit2] if cap_cfg_ui.get("include", False) else np.zeros_like(E_fit2)
        else:
            E_fit2, i_fit2 = E_fit, i_fit
            i_cap_fit2 = i_cap_fit_B
        # Stronger settings
        opt_cfg_C = {"use_de": True, "de_pop": 12, "de_iter": 420,
                     "lbfgs_maxiter": 2500, "use_nm": True, "nm_maxiter": 8000,
                     "restarts": max(2,restarts), "jitter": max(0.05,jitter)}
        optC=Optimizer(E_fit2,i_fit2,reg,bpB,i_cap_fit2,fit_rs_ui,rs_bounds,loss_cfg_ui,opt_cfg_C)
        bpC,rvC=optC.run()
        if rvC > rvB:
            bpB, rvB = bpC, rvC

    # Stage D: Final full-data refine (pure log L2) for maximum R²
    prog.progress(90,text="Stage D: final full-data refine...")
    # First refine without Rs/Cdl
    opt_ref_cfg = {"use_de": False, "lbfgs_maxiter": 4000, "use_nm": False, "restarts": 0, "jitter": 0.0}
    optRef=Optimizer(E_full,i_full,reg,bpB,np.zeros_like(E_full),False,rs_bounds,{"type":"log_l2"},opt_ref_cfg)
    bp,rv=optRef.run()
    # If UI enables Rs/Cdl, short refine with them enabled
    if fit_rs_ui or cap_cfg_ui.get("include", False):
        i_cap_vec_use = i_cap_vec_full if cap_cfg_ui.get("include", False) else np.zeros_like(E_full)
        optRef2=Optimizer(E_full,i_full,reg,bp,i_cap_vec_use,fit_rs_ui,rs_bounds,{"type":"log_l2"},
                          {"use_de":False,"lbfgs_maxiter":2000,"use_nm":False,"restarts":0,"jitter":0.0})
        bp,rv=optRef2.run()

    prog.progress(100,text="Done!"); prog.empty()

    # Diagnostics on full data (with UI Cdl)
    i_cap_for_diag = i_cap_vec_full if cap_cfg_ui.get("include", False) else np.zeros_like(E_full)
    diags=diagnose(E_full,i_full,reg,bp,rv,i_cap_for_diag)

    st.markdown("---")
    st.plotly_chart(plot_main(E_full,i_full,bp,reg,ct,i_cap_for_diag),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        if show_components:
            fc=plot_comp(E_full,bp,ct)
            if fc: st.plotly_chart(fc,use_container_width=True)
        else:
            st.info("Component plot disabled.")
    with c2:
        i_cap_for_plot = i_cap_vec_full if cap_cfg_ui.get("include", False) else np.zeros_like(E_full)
        fr=plot_res(E_full,i_full,bp,i_cap_for_plot)
        if fr: st.plotly_chart(fr,use_container_width=True)

    st.markdown("---"); show_p(bp,reg,ew,rho,ct,rv,cap_cfg_ui)
    st.markdown("---"); st.markdown("### Data Quality Diagnostics")
    for t,m,s in diags:
        cls={"e":"bx ber","w":"bx bwn","o":"bx bok"}[s]
        st.markdown(f'<div class="{cls}"><b>{t}</b><br>{m}</div>',unsafe_allow_html=True)

    if bp is not None:
        with st.expander("Model Equation"):
            p=dict(zip(PN,bp))
            st.markdown(f"""
**Dual-Cathodic Film-Coverage Model (with optional Rs)**

`i_net = i_anodic_total − (i_O₂ + i_H₂) + i_cap`

- O₂ (diff.-limited): `i_O₂ = i_kin / (1 + i_kin / iL)`, where `i_kin = icorr·exp(−2.303η/βc₁)`
- H₂ (activation): `i_H₂ = i₀,c₂·exp(−2.303η/βc₂)`
- Active anodic: `i_act = icorr·exp( 2.303η/βa )`
- Primary passivation: `θ₁ = σ(k₁·(E−Epp))`, `i = i_act·(1−θ₁) + ipass·θ₁`
- Transpassive: `i_tp = a_tp·exp(b_tp·(E−Eb))·σ(E−Eb)`
- Secondary passivation: `θ₂ = σ(k₂·(E−Esp))`, `i_an = (i_p1+i_tp)·(1−θ₂) + ipass₂·θ₂`

Ohmic drop: `E_eff = E − Rs·i_net` (self-consistent iterations)

Fitted: Ecorr={p['Ecorr']:.4f} V, icorr={p['icorr']:.3e}, βa={p['ba']*1000:.1f}, βc₁={p['bc1']*1000:.1f} mV/dec, Rs={p['Rs']:.3f} Ω·cm²
""")
    with st.expander("Optimization Log"):
        logs = []
        for obj in [optA, optB, optC, optRef, 'optRef2' in locals() and optRef2 or None]:
            if obj is not None:
                logs += obj.log
        for msg in logs:
            st.markdown(f"- {msg}")
    cd1,cd2=st.columns(2)
    with cd1:
        rows=[("Curve type",CT.INFO.get(ct,("?","",0))[0]),
            ("Ecorr (V)",bp[0] if bp is not None else None),
            ("icorr (A/cm2)",bp[1] if bp is not None else None),
            ("ba (mV/dec)",bp[2]*1000 if bp is not None else None),
            ("bc1 (mV/dec)",bp[3]*1000 if bp is not None else None),
            ("iL (A/cm2)",bp[4] if bp is not None else None),
            ("i0_c2 (A/cm2)",bp[5] if bp is not None else None),
            ("bc2 (mV/dec)",bp[6]*1000 if bp is not None else None),
            ("Rs (Ω·cm2)",bp[16] if bp is not None else None),
            ("R2(log)",rv)]
        for t,m,s in diags: rows.append((t,m))
        st.download_button("Results CSV",pd.DataFrame(rows,columns=["Parameter","Value"]).to_csv(index=False).encode(),
            "tafel_results.csv","text/csv",use_container_width=True)
    with cd2:
        st.download_button("Data CSV",pd.DataFrame(
            {"E_V":E_full,"i_Acm2":i_full,"log_abs_i":slog(i_full)}).to_csv(index=False).encode(),
            "tafel_data.csv","text/csv",use_container_width=True)

# ================================================================
# APP
# ================================================================
def main():
    st.markdown("""<div style="background:linear-gradient(135deg,#1e1e2e,#131320);
        border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px">⚡ Tafel Fitting Tool</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px">
        Dual-cathodic global model · Film-coverage physics · Optional Rs & Cdl · Final full-data refine
      </p></div>""",unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Settings")
        area=st.number_input("Electrode area (cm²)",0.001,100.0,1.0,0.01)
        mat=st.selectbox("Material",list(MATS.keys()))
        ew0,rho0=MATS[mat]
        if mat=="Custom":
            ew=st.number_input("Equivalent weight (EW)",1.0,300.0,ew0)
            rho=st.number_input("Density ρ (g cm⁻³)",0.5,25.0,rho0)
        else:
            ew,rho=ew0,rho0

        st.divider()
        st.markdown("#### Speed & Quality")
        speed_mode = st.selectbox("Speed mode", ["Fast","Balanced","Thorough"], index=2)
        max_fit_points = st.slider("Max points for fitting (thinning)", 300, 4000, 1200, 50)
        use_full_fit = st.checkbox("Use full dataset for fitting (skip thinning)", value=False)
        show_components = st.checkbox("Show component plot", value=True)
        escalate_r2 = st.slider("Fallback threshold R² (log)", 0.90, 0.995, 0.98, 0.005)
        restarts = st.slider("Local restarts (multi-start)", 0, 8, 3, 1)
        jitter = st.slider("Restart jitter (packed-space fraction)", 0.02, 0.20, 0.06, 0.01)

        st.divider()
        st.markdown("#### Ohmic Drop (Rs)")
        enable_rs = st.checkbox("Fit Rs (Ω·cm²)", value=False)
        rs_lo = st.number_input("Rs lower bound (Ω·cm²)", 0.0, 1e4, 0.0, 0.1)
        rs_hi = st.number_input("Rs upper bound (Ω·cm²)", 0.0, 1e4, 200.0, 0.1)
        if not enable_rs:
            st.caption("First pass ignores Rs for stability; final refine can re-enable if checked.")

        st.divider()
        st.markdown("#### Capacitive Current (Cdl · ν)")
        enable_cdl = st.checkbox("Include capacitive current", value=False)
        cdl = st.number_input("Cdl (F/cm²)", 0.0, 1e-1, 0.0, format="%.5f")
        nu = st.number_input("Scan rate ν (V/s)", 0.0, 10.0, 0.0, format="%.4f")
        if enable_cdl and nu == 0.0:
            st.warning("Scan rate is 0; capacitive current will be zero.")

        st.divider()
        st.markdown("#### Loss Function (main fit)")
        loss_type = st.selectbox("Type", ["Log L2", "Hybrid (log+linear)", "Huber (log)"], index=0)
        alpha = st.slider("Hybrid weight α (log vs linear)", 0.0, 1.0, 0.7)
        delta = st.slider("Huber δ (decades)", 0.05, 1.0, 0.30)
        linear_scale = st.number_input("Linear residual scale (A/cm²)", 0.0, 1e3, 0.0, format="%.6f",
                                       help="If 0, auto uses median |i|.")

        st.divider()
        st.markdown("""<div style="font-size:11px;color:#a6adc8;line-height:1.6">
        - First pass disables Rs/Cdl to find a stable basin.<br>
        - Then refines with your Rs/Cdl settings and finally on full data for the best R².<br>
        - Use Thorough mode and higher Max points for challenging curves.<br>
        </div>""",unsafe_allow_html=True)

    up=st.file_uploader("Upload polarization data",type=["csv","txt","xlsx","xls"])
    if up is None:
        st.info("Upload a file to start. Two numeric columns required: Potential (V) and Current (A).")
        return

    with st.spinner("Reading..."):
        try: df=load_file(up)
        except Exception as ex: st.error(f"Read error: {ex}"); return
    with st.spinner("Detecting columns..."):
        try: ec,ic,ifac=auto_cols(df)
        except Exception as ex: st.error(f"Column error: {ex}"); return
    with st.expander(f"Detected: {ec} / {ic}",expanded=False):
        st.dataframe(df[[ec,ic]].head(10),use_container_width=True)

    # Preserve original order (scan direction)
    E=df[ec].values.astype(float)
    ir=df[ic].values.astype(float)*ifac
    ok=np.isfinite(E)&np.isfinite(ir); E_full,ir=E[ok],ir[ok]
    i_full = ir/area

    # Loss configuration
    lt = loss_type.lower()
    if lt.startswith("log l2"):
        loss_cfg = {"type": "log_l2"}
    elif lt.startswith("hybrid"):
        loss_cfg = {"type":"hybrid","alpha":alpha,"linear_scale":(linear_scale if linear_scale>0 else np.median(np.abs(i_full))+1e-12)}
    else:
        loss_cfg = {"type":"huber_log","delta":delta}

    cap_cfg = {"include": enable_cdl, "Cdl": cdl, "nu": nu}
    rs_bounds = (rs_lo, rs_hi)

    process(E_full, i_full, area, ew, rho, cap_cfg, enable_rs, rs_bounds, loss_cfg,
            speed_mode, max_fit_points, show_components, use_full_fit, escalate_r2, restarts, jitter)

if __name__=="__main__":
    main()
