"""
Tafel Fitting Tool
==============================
Publication-ready global electrochemical polarization curve fitter.

Model: Film-coverage with dual cathodic reactions (16 parameters total):
  Cathodic:  O₂ reduction (diffusion-limited) + H₂ evolution (activation)
  Anodic:    Tafel → primary passivation → transpassive → secondary passivity

Adaptive: only optimizes parameters relevant to detected curve type (4→16).
Speed: <1s (active only) to ~25s (full multi-region).
Diagnostics: explains discrepancies in the data.
"""

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
# MATH
# ================================================================
def slog(x): return np.log10(np.maximum(np.abs(x),1e-20))
def sm(y,w=11,p=3):
    n=len(y); w=min(w,n if n%2==1 else n-1)
    return savgol_filter(y,w,min(p,w-1)) if w>=5 else y.copy()
def r2(yt,yp):
    sr=np.sum((yt-yp)**2); st_=np.sum((yt-yt.mean())**2)
    return float(max(0,1-sr/st_)) if st_>0 else 0.0
def sig(x,k=1.0): return 1.0/(1.0+np.exp(-np.clip(k*x,-50,50)))

# ================================================================
# GLOBAL MODEL  (16 parameters)
# ================================================================
# p = [Ecorr, icorr, ba, bc1, iL,       — core + O₂ cathodic
#      i0_c2, bc2,                        — H₂ evolution
#      Epp, k_pass, ipass,               — primary passivation
#      Eb, a_tp, b_tp,                   — transpassive
#      Esp, k_sp, ipass2]               — secondary passivity
#
# Cathodic = O₂ reduction (diffusion-limited) + H₂ evolution (activation)
# Anodic   = Tafel → θ₁-passivation → transpassive → θ₂-secondary passive

PN = ["Ecorr","icorr","ba","bc1","iL",
      "i0_c2","bc2",
      "Epp","k_pass","ipass",
      "Eb","a_tp","b_tp",
      "Esp","k_sp","ipass2"]
NP = 16
LOG_I = {1,4,5,9,11,15}  # indices of current-like params (log-transformed)

def gmodel(E, p):
    """Global polarization model — returns net current density."""
    Ecorr,icorr,ba,bc1,iL = p[0],p[1],p[2],p[3],p[4]
    i0_c2,bc2 = p[5],p[6]
    Epp,k_pass,ipass = p[7],p[8],p[9]
    Eb,a_tp,b_tp = p[10],p[11],p[12]
    Esp,k_sp,ipass2 = p[13],p[14],p[15]
    eta = E - Ecorr
    # Cathodic 1: O₂ reduction → diffusion-limited
    ic1k = icorr * np.exp(np.clip(-2.303*eta/bc1,-50,50))
    i_c1 = ic1k / (1.0 + ic1k/iL)
    # Cathodic 2: H₂ evolution (pure activation)
    i_c2 = i0_c2 * np.exp(np.clip(-2.303*eta/bc2,-50,50))
    # Anodic: activation
    i_act = icorr * np.exp(np.clip(2.303*eta/ba,-50,50))
    # Primary passivation film θ₁
    t1 = sig(E-Epp, k_pass)
    i_p1 = i_act*(1-t1) + ipass*t1
    # Transpassive
    i_tp = a_tp * np.exp(np.clip(b_tp*(E-Eb),-50,50)) * sig(E-Eb,40)
    # Secondary passivation film θ₂
    t2 = sig(E-Esp, k_sp)
    i_an = (i_p1+i_tp)*(1-t2) + ipass2*t2
    return i_an - (i_c1 + i_c2)

def gcomp(E, p):
    """Return individual model components for visualization."""
    Ecorr,icorr,ba,bc1,iL = p[0:5]
    i0_c2,bc2 = p[5:7]
    Epp,k_pass,ipass = p[7:10]
    Eb,a_tp,b_tp = p[10:13]
    Esp,k_sp,ipass2 = p[13:16]
    eta = E-Ecorr
    ic1k = icorr*np.exp(np.clip(-2.303*eta/bc1,-50,50))
    i_c1 = ic1k/(1+ic1k/iL)
    i_c2 = i0_c2*np.exp(np.clip(-2.303*eta/bc2,-50,50))
    i_act = icorr*np.exp(np.clip(2.303*eta/ba,-50,50))
    t1 = sig(E-Epp,k_pass); i_p1 = i_act*(1-t1)+ipass*t1
    i_tp = a_tp*np.exp(np.clip(b_tp*(E-Eb),-50,50))*sig(E-Eb,40)
    t2 = sig(E-Esp,k_sp); i_an = (i_p1+i_tp)*(1-t2)+ipass2*t2
    return dict(ic1=i_c1,ic2=i_c2,iact=i_act,ip1=i_p1,itp=i_tp,
                ian=i_an,t1=t1,t2=t2,itot=i_an-(i_c1+i_c2))

# ================================================================
# CURVE TYPE CLASSIFICATION & ADAPTIVE PARAMETER MASKS
# ================================================================
class CT:
    ACTIVE="A"; ACTIVE_D="AD"; ACTIVE_H2="AH"
    PASSIVE="P"; PASSIVE_D="PD"
    PASS_TP="PT"; PASS_TP_SP="PTS"
    PITTING="PP"; FULL="F"

    INFO = {
        "A":  ("Active Only","Pure Butler-Volmer kinetics.",
               [0,1,2,3]),
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
# REGION DETECTION
# ================================================================
def detect(E, i):
    reg={}; n=len(E); ai=np.abs(i)
    # Ecorr
    sc=np.where(np.diff(np.sign(i)))[0]
    if len(sc)>0:
        k=sc[0]; d=i[k+1]-i[k]
        reg["Ecorr"]=float(E[k]-i[k]*(E[k+1]-E[k])/d) if abs(d)>0 else float(E[k])
        reg["eci"]=k
    else:
        k=int(np.argmin(ai)); reg["Ecorr"]=float(E[k]); reg["eci"]=k
    Ec=reg["Ecorr"]

    # ── Cathodic analysis ───────────────────────────────
    ci=np.where(E<Ec)[0]
    if len(ci)>=8:
        lc=sm(slog(ai[ci]),min(11,(len(ci)//2)*2-1 or 5))
        dl=np.abs(np.gradient(lc,E[ci]))
        # Limiting current (flat region)
        thr=np.percentile(dl,20); flat=dl<max(thr,0.5)
        runs=[(k2,list(g)) for k2,g in groupby(enumerate(flat),key=lambda x:x[1]) if k2]
        if runs:
            br=max(runs,key=lambda x:len(x[1])); idxs=[s[0] for s in br[1]]
            er=abs(E[ci[idxs[-1]]]-E[ci[idxs[0]]])
            if len(idxs)>=4 and er>0.03 and abs(E[ci[idxs[0]]]-Ec)>0.06:
                reg.update(iL=float(np.median(ai[ci[idxs]])),
                           Els=float(E[ci[idxs[0]]]),Ele=float(E[ci[idxs[-1]]]))

        # H₂ evolution: current rising again after limiting plateau
        if "iL" in reg:
            # Check if current increases at very negative E (past limiting)
            very_neg = ci[E[ci] < reg["Ele"]-0.02]
            if len(very_neg) >= 5:
                lc_neg = slog(ai[very_neg])
                grad_neg = np.gradient(lc_neg, E[very_neg])
                # Negative gradient = current increasing with decreasing E
                if np.mean(grad_neg[:min(10,len(grad_neg))]) < -1.0:
                    reg["has_H2"] = True
                    # Estimate H2 params
                    try:
                        s,b,rv,*_ = linregress(E[very_neg[:min(15,len(very_neg))]],
                                               lc_neg[:min(15,len(very_neg))])
                        if s < 0 and abs(1/s) > 0.05:
                            reg["bc2_est"] = abs(1/s)
                            reg["i0_c2_est"] = 10**(b + s*Ec)
                    except: pass
        else:
            # No clear limiting plateau — check if there's a second slope
            if len(ci) >= 20:
                half = len(ci)//2
                try:
                    s1,_,r1,*_=linregress(E[ci[:half]],lc[:half])
                    s2,_,r2v,*_=linregress(E[ci[half:]],lc[half:])
                    if r1**2>0.85 and r2v**2>0.85 and abs(s1-s2)>2:
                        reg["has_H2"]=True
                except: pass

    # ── Anodic analysis ─────────────────────────────────
    ani=np.where(E>Ec)[0]
    if len(ani)<6:
        if "iL" in reg:
            ct = CT.ACTIVE_H2 if reg.get("has_H2") else CT.ACTIVE_D
        else:
            ct = CT.ACTIVE
        reg["ct"]=ct; return reg

    log_a=sm(slog(ai[ani]),min(15,(len(ani)//2)*2-1 or 5))
    dlog=np.gradient(log_a,E[ani]); adl=np.abs(dlog)

    # Anodic peaks (active dissolution nose)
    peaks=[]
    if len(ani)>20:
        ism=sm(ai[ani],min(21,(len(ani)//2)*2-1 or 5))
        order=max(3,len(ani)//20)
        pki=argrelextrema(slog(ism),np.greater,order=order)[0]
        for pk in pki:
            rest=slog(ism)[pk:]
            if len(rest)>5:
                prom=slog(ism)[pk]-np.min(rest[3:])
                if prom>0.3: peaks.append(dict(idx=ani[pk],E=float(E[ani[pk]]),prom=float(prom)))
    reg["peaks"]=peaks

    # Passive regions (flat)
    thr_p=np.percentile(adl,25); flat=adl<max(thr_p,0.8)
    runs=[(k2,list(g)) for k2,g in groupby(enumerate(flat),key=lambda x:x[1]) if k2]
    pr=[]
    for _,ri in runs:
        idxs=[s[0] for s in ri]; er=abs(E[ani[idxs[-1]]]-E[ani[idxs[0]]])
        if len(idxs)>=4 and er>0.03:
            ps,pe=ani[idxs[0]],ani[idxs[-1]]
            im=float(np.median(ai[ps:pe+1]))
            pre=np.where((E>Ec)&(E<E[ps]))[0]
            valid=(len(pre)>2 and im<np.max(ai[pre])*0.5) or er>0.08
            if valid: pr.append(dict(ps=ps,pe=pe,Es=float(E[ps]),Ee=float(E[pe]),ip=im,rng=er))
    if pr: reg["p1"]=pr[0]; reg["Epp"]=pr[0]["Es"]; reg["ipass"]=pr[0]["ip"]
    if len(pr)>=2: reg["p2"]=pr[1]; reg["Esp"]=pr[1]["Es"]; reg["ipass2"]=pr[1]["ip"]

    # Breakdown/transpassive
    if "p1" in reg:
        pe=reg["p1"]["pe"]
        if pe+5<n:
            da=np.gradient(slog(ai[pe:]),E[pe:])
            thr_b=np.percentile(np.abs(da),75)
            jump=np.where(np.abs(da)>max(thr_b,2.0))[0]
            if len(jump):
                reg["Eb"]=float(E[pe+jump[0]])
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
    # For passive types, always enable cathodic params if data is wide enough
    if ct in (CT.PASSIVE,CT.PASSIVE_D,CT.PASS_TP,CT.PITTING,CT.PASS_TP_SP,CT.FULL):
        if len(ci)>=10: ct_idx=CT.free_idx(ct)
        # Always enable dual cathodic for passive+ systems (they usually need it)
        if 4 not in CT.free_idx(ct):
            pass  # idx already includes cathodic if passive_d or higher
    reg["ct"]=ct
    return reg

# ================================================================
# INITIAL GUESS
# ================================================================
def init_guess(E, i, reg):
    Ec=reg["Ecorr"]; ai=np.abs(i); lg=slog(i)
    ba0,bc0,ic0=0.060,0.120,max(ai[reg["eci"]],1e-12)

    # Tafel slopes via grid search
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

    # Defaults — inactive components far from data
    Emax=E[-1]; Emin=E[0]
    return np.array([
        Ec,                                         # 0  Ecorr
        ic0,                                        # 1  icorr
        ba0,                                        # 2  ba
        bc0,                                        # 3  bc1
        reg.get("iL",1e2),                          # 4  iL
        reg.get("i0_c2_est",ic0*1e-3),              # 5  i0_c2
        reg.get("bc2_est",0.150),                    # 6  bc2
        reg.get("Epp",Emax+10),                     # 7  Epp
        40.0 if "p1" in reg else 50.0,              # 8  k_pass
        reg.get("ipass",ic0),                        # 9  ipass
        reg.get("Eb",Emax+10),                      # 10 Eb
        reg.get("ipass",ic0)*0.01 if "Eb" in reg else 1e-30, # 11 a_tp
        8.0,                                        # 12 b_tp
        reg.get("Esp",Emax+20),                     # 13 Esp
        50.0,                                       # 14 k_sp
        reg.get("ipass2",ic0),                       # 15 ipass2
    ])

# ================================================================
# OPTIMIZER
# ================================================================
class Optimizer:
    def __init__(self, E, i, reg, p0):
        self.E=E; self.i=i; self.ld=slog(i); self.reg=reg
        self.ct=reg["ct"]; self.pfull=p0.copy()
        self.fidx=CT.free_idx(self.ct)
        self.nf=len(self.fidx)
        self.log=[]; self.best_p=p0.copy(); self.best_s=1e30
        self._bounds()

    def _bounds(self):
        p=self.pfull; reg=self.reg; Ec=p[0]; ic=max(p[1],1e-14)
        Emax=self.E[-1]; Emin=self.E[0]
        # Full 16 bounds
        self.lo=np.array([
            Ec-0.15, max(ic*1e-4,1e-15), 0.010, 0.010,
            max(reg.get("iL",1e-6)*0.01,1e-10),
            max(ic*1e-6,1e-16), 0.04,
            p[7]-0.25 if "p1" in reg else Emax+5,
            5.0, max(p[9]*0.01 if "p1" in reg else ic*0.01,1e-14),
            p[10]-0.25 if "Eb" in reg else Emax+5,
            max(p[11]*1e-4 if "Eb" in reg else 1e-30,1e-30), 0.5,
            p[13]-0.25 if "p2" in reg else Emax+10,
            5.0, max(p[15]*0.01 if "p2" in reg else 1e-14,1e-14)])
        self.hi=np.array([
            Ec+0.15, min(ic*1e4,1e0), 0.500, 0.500,
            max(reg.get("iL",1e0)*100,1e-2),
            max(ic*10,1e-6), 0.350,
            p[7]+0.25 if "p1" in reg else Emax+15,
            200.0, max(p[9]*100 if "p1" in reg else ic*100,1e-4),
            p[10]+0.25 if "Eb" in reg else Emax+15,
            max(p[11]*1e4 if "Eb" in reg else 1e-20,1e-10), 40.0,
            p[13]+0.25 if "p2" in reg else Emax+25,
            200.0, max(p[15]*100 if "p2" in reg else 1e-4,1e-4)])
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
    def _obj(self, x):
        p=self._unpack(x)
        try: return float(np.sum((self.ld-slog(gmodel(self.E,p)))**2))
        except: return 1e30
    def _up(self, x, tag):
        s=self._obj(x); rv=r2(self.ld,slog(gmodel(self.E,self._unpack(x))))
        if s<self.best_s: self.best_p=self._unpack(x); self.best_s=s; self.log.append(f"  {tag}: R²={rv:.6f}"); return True
        self.log.append(f"  {tag}: R²={rv:.6f} (no impr.)"); return False

    def run(self):
        t0=time.time()
        self.log.append(f"Curve: {CT.INFO[self.ct][0]}  ({self.nf} free params)")
        x0=self._pack(self.pfull); self.best_s=self._obj(x0)
        self.log.append(f"  Init: R²={r2(self.ld,slog(gmodel(self.E,self.pfull))):.4f}")
        bnds=self._pbounds()
        # DE — adaptive sizing: fewer params = tighter search
        ps=max(8,min(12,self.nf)); mi=max(150,min(500,self.nf*35))
        self.log.append(f"Stage 1: DE  pop={ps} iter={mi}")
        try:
            t1=time.time()
            res=differential_evolution(self._obj,bnds,seed=42,maxiter=mi,tol=1e-12,
                popsize=ps,mutation=(0.5,1.5),recombination=0.9,polish=False)
            self._up(res.x,f"DE ({time.time()-t1:.1f}s)")
        except Exception as ex: self.log.append(f"  DE err: {ex}")
        # L-BFGS-B
        self.log.append("Stage 2: L-BFGS-B")
        try:
            t1=time.time()
            r2v=minimize(self._obj,self._pack(self.best_p),method="L-BFGS-B",bounds=bnds,
                options={"maxiter":20000,"ftol":1e-15,"gtol":1e-12})
            self._up(r2v.x,f"L-BFGS-B ({time.time()-t1:.1f}s)")
        except: pass
        # NM
        self.log.append("Stage 3: Nelder-Mead")
        try:
            t1=time.time()
            r3=minimize(self._obj,self._pack(self.best_p),method="Nelder-Mead",
                options={"maxiter":15000,"xatol":1e-13,"fatol":1e-15,"adaptive":True})
            self._up(r3.x,f"NM ({time.time()-t1:.1f}s)")
        except: pass
        dt=time.time()-t0
        rv=r2(self.ld,slog(gmodel(self.E,self.best_p)))
        q="Excellent" if rv>=0.995 else "Good" if rv>=0.97 else "Acceptable" if rv>=0.90 else "Poor"
        self.log.append(f"Result: {q}  R²(log)={rv:.6f}  [{dt:.1f}s]")
        return self.best_p, rv

# ================================================================
# DIAGNOSTICS
# ================================================================
def diagnose(E, i, reg, bp, rv):
    issues=[]; ai=np.abs(i); lg=slog(i); n=len(E); Ec=reg["Ecorr"]
    # Noise
    if n>20:
        lsm=sm(lg,min(21,(n//4)*2-1 or 5)); noise=np.std(lg-lsm)
        if noise>0.5: issues.append(("High noise",f"σ={noise:.2f} dec. Use slower scan rate or average scans.","e"))
        elif noise>0.15: issues.append(("Moderate noise",f"σ={noise:.2f} dec.","w"))
        else: issues.append(("Low noise",f"σ={noise:.3f} dec.","o"))
    if bp is not None:
        if bp[2]*1000>200: issues.append(("Large βa",f"{bp[2]*1000:.0f} mV/dec — possible IR drop or multi-step mechanism.","w"))
        if bp[3]*1000>200: issues.append(("Large βc₁",f"{bp[3]*1000:.0f} mV/dec — possible diffusion in Tafel region or IR drop.","w"))
        if bp[6]*1000>250: issues.append(("Large βc₂ (H₂)",f"{bp[6]*1000:.0f} mV/dec — may indicate coupled reactions.","w"))
    Er=E[-1]-E[0]
    if Er<0.3: issues.append(("Narrow scan",f"{Er*1000:.0f} mV. May miss regions.","w"))
    if n/max(Er,0.01)<50: issues.append(("Low density",f"{n/max(Er,0.01):.0f} pts/V.","w"))
    if np.sum(np.abs(E-Ec)<0.05)<5: issues.append(("Sparse near Ecorr","<5 pts within ±50 mV.","w"))
    cat_range=Ec-E[0]; an_range=E[-1]-Ec
    if cat_range>0 and an_range>0:
        ratio=an_range/cat_range
        if ratio>4 or ratio<0.25: issues.append(("Asymmetric scan",
            f"Cathodic {cat_range*1000:.0f} mV vs Anodic {an_range*1000:.0f} mV.","w"))
    if bp is not None:
        res=lg-slog(gmodel(E,bp))
        if n>10:
            nruns=np.sum(np.abs(np.diff(np.sign(res)))>0)+1
            if nruns<n*0.25: issues.append(("Systematic misfit",
                f"{nruns} sign changes (expect ~{n//2}). Model may miss a feature.","w"))
        for name,mask in [("near Ecorr",np.abs(E-Ec)<0.05),
            ("cathodic",E<Ec-0.1),("passive",(E>reg.get("Epp",99))&(E<reg.get("Eb",E[-1])))]:
            if mask.sum()>3:
                rmse=np.sqrt(np.mean(res[mask]**2))
                if rmse>0.5: issues.append((f"Poor fit: {name}",f"RMSE={rmse:.3f} dec.","w"))
        if bp[1]>0.1: issues.append(("Very high icorr",f"{bp[1]:.2e} — check area/units.","e"))
        if bp[1]<1e-14: issues.append(("Very low icorr",f"{bp[1]:.2e}.","w"))
    if rv is not None and rv<0.90: issues.append(("Poor fit",
        f"R²(log)={rv:.4f}. Possible: coupled reactions, adsorption, roughness, artifacts.","e"))
    return issues

# ================================================================
# PLOTS
# ================================================================
def plot_main(E, i, bp, reg, ct):
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
        Em=np.linspace(E.min(),E.max(),1000)
        try:
            im=gmodel(Em,bp); rv=r2(lg,slog(gmodel(E,bp)))
            fig.add_trace(go.Scatter(x=Em,y=slog(im),mode="lines",
                name=f"Global Fit  R²(log)={rv:.4f}",line=dict(color=CL["fit"],width=3)))
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
    Em=np.linspace(E.min(),E.max(),800); c=gcomp(Em,bp); fig=go.Figure()
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
    fig.add_trace(go.Scatter(x=Em,y=slog(c["itot"]),mode="lines",name="Total",
        line=dict(color=CL["fit"],width=3)))
    fig.update_layout(template="plotly_dark",plot_bgcolor=CL["bg"],paper_bgcolor=CL["paper"],
        title=dict(text="Component Breakdown",font=dict(size=15,color=CL["tx"])),
        xaxis=dict(title="Potential (V)",gridcolor=CL["grid"],color=CL["tx"]),
        yaxis=dict(title="log₁₀|i|",gridcolor=CL["grid"],color=CL["tx"]),
        legend=dict(bgcolor="rgba(19,19,32,0.9)",bordercolor=CL["grid"],font=dict(color=CL["tx"],size=10)),
        height=420,margin=dict(l=70,r=20,t=50,b=60))
    return fig

def plot_res(E, i, bp):
    if bp is None: return None
    ld=slog(i); lp=slog(gmodel(E,bp)); res=ld-lp; rmse=np.sqrt(np.mean(res**2))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.55,0.45],vertical_spacing=0.06,
        subplot_titles=("Overlay","Residuals (log)"))
    fig.add_trace(go.Scatter(x=E,y=ld,mode="lines",name="Measured",line=dict(color=CL["data"],width=2)),row=1,col=1)
    Em=np.linspace(E.min(),E.max(),600)
    fig.add_trace(go.Scatter(x=Em,y=slog(gmodel(Em,bp)),mode="lines",name="Fit",
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

def show_p(bp, reg, ew, rho, ct, rv):
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
                f'<div style="font-size:12px;color:#a6adc8">{desc} ({nf} free params)</div></div>',unsafe_allow_html=True)
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
        else: st.info("Simple Butler-Volmer.")
    with c4:
        st.markdown('<div class="sh">Passivity & Breakdown</div>',unsafe_allow_html=True)
        if ct not in (CT.ACTIVE,CT.ACTIVE_D,CT.ACTIVE_H2):
            pc("Epp",p["Epp"],"V","#a6e3a1"); pc("ipass",p["ipass"],"A cm⁻²","#94e2d5")
            if ct in (CT.PASS_TP,CT.PITTING,CT.PASS_TP_SP,CT.FULL):
                pc("Eb",p["Eb"],"V","#f38ba8")
                pc("Eb−Ecorr",(p["Eb"]-p["Ecorr"])*1000,"mV","#f38ba8")
            if ct in (CT.PASS_TP_SP,CT.FULL):
                pc("Esp",p["Esp"],"V","#cba6f7"); pc("ipass₂",p["ipass2"],"A cm⁻²","#cba6f7")
        else: st.info("No passivation.")

# ================================================================
# MATERIALS & DEMOS
# ================================================================
MATS={"Carbon Steel / Iron":(27.92,7.87),"304 Stainless Steel":(25.10,7.90),
    "316 Stainless Steel":(25.56,8.00),"Copper":(31.77,8.96),"Aluminum":(8.99,2.70),
    "Nickel":(29.36,8.91),"Titanium":(11.99,4.51),"Zinc":(32.69,7.14),"Custom":(27.92,7.87)}

def make_demo(ch):
    np.random.seed(42)
    if "secondary" in ch.lower() or "full" in ch.lower():
        E=np.linspace(-0.65,1.0,600)
        p=[-0.38,3e-6,0.060,0.110,5e-4, 1e-9,0.150,
           -0.10,50.0,4e-6, 0.50,2e-6,10.0, 0.80,40.0,3e-6]
    elif "pitting" in ch.lower():
        E=np.linspace(-0.55,0.55,400)
        p=[-0.35,2e-6,0.055,0.120,1e3, 1e-10,0.150,
           -0.05,60.0,3e-6, 0.35,1e-4,25.0, 1.5,50.0,1e3]
    elif "trans" in ch.lower():
        E=np.linspace(-0.55,0.75,450)
        p=[-0.38,2e-6,0.065,0.110,1e3, 1e-10,0.150,
           -0.10,45.0,5e-6, 0.55,3e-6,8.0, 5.0,50.0,1e3]
    elif "passive" in ch.lower():
        E=np.linspace(-0.55,0.50,400)
        p=[-0.40,2e-6,0.065,0.110,1e3, 1e-10,0.150,
           -0.15,50.0,4e-6, 5.0,1e-30,1.0, 10.0,50.0,1e3]
    elif "dual" in ch.lower() or "h2" in ch.lower():
        E=np.linspace(-0.85,0.50,500)
        p=[-0.30,5e-8,0.065,0.080,3e-6, 1e-11,0.130,
           -0.05,40.0,5e-8, 0.40,1e-7,8.0, 5.0,50.0,1e3]
    elif "diffusion" in ch.lower():
        E=np.linspace(-0.80,0.30,400)
        p=[-0.40,3e-6,0.070,0.120,2e-4, 1e-10,0.150,
           5.0,50.0,1e3, 5.0,1e-30,1.0, 10.0,50.0,1e3]
    else:
        E=np.linspace(-0.65,0.40,350)
        p=[-0.45,5e-6,0.060,0.120,1e3, 1e-15,0.150,
           5.0,50.0,1e3, 5.0,1e-30,1.0, 10.0,50.0,1e3]
    i=gmodel(E,p)
    return E, i+np.random.normal(0,np.abs(i)*0.03+3e-9,len(E))

# ================================================================
# MAIN PIPELINE
# ================================================================
def process(E, i_d, area, ew, rho):
    prog=st.progress(0,text="Detecting regions..."); reg=detect(E,i_d); ct=reg["ct"]
    prog.progress(15,text="Initial estimates..."); p0=init_guess(E,i_d,reg)
    prog.progress(25,text=f"Optimizing ({CT.nfree(ct)} params)...")
    opt=Optimizer(E,i_d,reg,p0); bp,rv=opt.run()
    prog.progress(90,text="Diagnostics..."); diags=diagnose(E,i_d,reg,bp,rv)
    prog.progress(100,text="Done!"); prog.empty()

    st.markdown("---")
    st.plotly_chart(plot_main(E,i_d,bp,reg,ct),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fc=plot_comp(E,bp,ct)
        if fc: st.plotly_chart(fc,use_container_width=True)
    with c2:
        fr=plot_res(E,i_d,bp)
        if fr: st.plotly_chart(fr,use_container_width=True)
    st.markdown("---"); show_p(bp,reg,ew,rho,ct,rv)
    st.markdown("---"); st.markdown("### Data Quality Diagnostics")
    for t,m,s in diags:
        cls={"e":"bx ber","w":"bx bwn","o":"bx bok"}[s]
        st.markdown(f'<div class="{cls}"><b>{t}</b><br>{m}</div>',unsafe_allow_html=True)

    if bp is not None:
        with st.expander("Model Equation"):
            p=dict(zip(PN,bp))
            st.markdown(f"""
**Dual-Cathodic Film-Coverage Model**

`i_net = i_anodic_total − (i_O₂ + i_H₂)`

**Cathodic 1 — O₂ reduction** (diffusion-limited):
`i_O₂ = icorr·exp(−2.303η/βc₁) / (1 + i_kin/iL)`
→ Tafel near Ecorr, saturates at iL = {p['iL']:.3e} A/cm²

**Cathodic 2 — H₂ evolution** (activation-controlled):
`i_H₂ = i₀,c₂·exp(−2.303η/βc₂)`
→ Kicks in at negative potentials where O₂ is depleted.
i₀,c₂ = {p['i0_c2']:.3e}, βc₂ = {p['bc2']*1000:.1f} mV/dec

**Anodic — Active dissolution:**
`i_act = icorr·exp(2.303η/βa)`

**Primary passivation** (film coverage θ₁):
`θ₁ = sigmoid(k₁·(E−Epp))`  →  `i = i_act·(1−θ₁) + ipass·θ₁`
Epp = {p['Epp']:.4f} V, ipass = {p['ipass']:.3e}

**Transpassive:** `i_tp = a_tp·exp(b_tp·(E−Eb))·σ(E−Eb)` | Eb = {p['Eb']:.4f} V

**Secondary passivation** (θ₂):
`i_total = (i_p1+i_tp)·(1−θ₂) + ipass₂·θ₂` | Esp = {p['Esp']:.4f} V

**Fitted:** Ecorr={p['Ecorr']:.4f} V, icorr={p['icorr']:.3e}, βa={p['ba']*1000:.1f}, βc₁={p['bc1']*1000:.1f} mV/dec
""")
    with st.expander("Optimization Log"):
        for msg in opt.log: st.markdown(f"- {msg}")
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
            ("R2(log)",rv)]
        for t,m,s in diags: rows.append((t,m))
        st.download_button("Results CSV",pd.DataFrame(rows,columns=["Parameter","Value"]).to_csv(index=False).encode(),
            "tafel_results.csv","text/csv",use_container_width=True)
    with cd2:
        st.download_button("Data CSV",pd.DataFrame(
            {"E_V":E,"i_Acm2":i_d,"log_abs_i":slog(i_d)}).to_csv(index=False).encode(),
            "tafel_data.csv","text/csv",use_container_width=True)

def main():
    st.markdown("""<div style="background:linear-gradient(135deg,#1e1e2e,#131320);
        border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">
      <h1 style="margin:0;color:#cdd6f4;font-size:26px">⚡ Tafel Fitting Tool</h1>
      <p style="margin:4px 0 0;color:#6c7086;font-size:13px">
        Dual-cathodic global model · Film-coverage physics · All curve types · Fast · Diagnostics
      </p></div>""",unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Settings")
        area=st.number_input("Electrode area (cm²)",0.001,100.0,1.0,0.01)
        mat=st.selectbox("Material",list(MATS.keys()))
        ew0,rho0=MATS[mat]
        ew,rho=(st.number_input("EW",1.0,300.0,ew0),st.number_input("ρ",0.5,25.0,rho0)) if mat=="Custom" else (ew0,rho0)
        st.divider()
        st.markdown("""<div style="font-size:11px;color:#a6adc8;line-height:2.0">
        <b style="color:#89b4fa">Auto-detected types:</b><br>
        Active | +Diffusion | +H₂ Evolution<br>
        Passive | +Transpassive | +2nd Passive<br>
        Pitting | Full multi-region<br><br>
        <b>Dual cathodic:</b> O₂ + H₂<br>
        <b>Adaptive:</b> 4→16 free params
        </div>""",unsafe_allow_html=True)
    up=st.file_uploader("Upload polarization data",type=["csv","txt","xlsx","xls"])
    if up is None:
        c1,c2,c3=st.columns([1,2,1])
        with c2:
            demo=st.selectbox("Or try a demo",[
                "Active only","Diffusion-limited cathodic",
                "Dual cathodic (O₂ + H₂ evolution)",
                "Active → Passive","Passive + Transpassive","Passive + Pitting",
                "Full: Passive + Transpassive + Secondary Passivity"])
            if st.button("Run demo",use_container_width=True,type="primary"):
                st.session_state["demo"]=demo
        return
    with st.spinner("Reading..."):
        try: df=load_file(up)
        except Exception as ex: st.error(f"Read error: {ex}"); return
    with st.spinner("Detecting columns..."):
        try: ec,ic,ifac=auto_cols(df)
        except Exception as ex: st.error(f"Column error: {ex}"); return
    with st.expander(f"Detected: {ec} / {ic}",expanded=False):
        st.dataframe(df[[ec,ic]].head(10),use_container_width=True)
    E=df[ec].values.astype(float); ir=df[ic].values.astype(float)*ifac
    ok=np.isfinite(E)&np.isfinite(ir); E,ir=E[ok],ir[ok]
    idx=np.argsort(E); E,ir=E[idx],ir[idx]
    process(E,ir/area,area,ew,rho)

if __name__=="__main__":
    if "demo" in st.session_state:
        st.markdown('<div style="background:linear-gradient(135deg,#1e1e2e,#131320);'
            'border:1px solid #313244;border-radius:12px;padding:20px 28px;margin-bottom:20px">'
            '<h1 style="margin:0;color:#cdd6f4;font-size:26px">⚡ Tafel Fitting Tool</h1></div>',unsafe_allow_html=True)
        with st.sidebar:
            area=st.number_input("Area",0.001,100.0,1.0)
            mat=st.selectbox("Material",list(MATS.keys()))
            ew,rho=MATS[mat]
            if st.button("Back"): del st.session_state["demo"]; st.rerun()
        E_d,i_d=make_demo(st.session_state["demo"])
        st.info(f"Demo: {st.session_state['demo']}")
        process(E_d,i_d/1.0,1.0,ew,rho)
    else: main()
