# Coupled 4-state inductive-crosstalk model — copy, paste, run.  PURE STANDARD LIBRARY.
#   python3 this_file.py      # no numpy / scipy / matplotlib needed; prints an ASCII waveform
#
# Reproduces the DEFAULT case populated in the Spec Checker (section 7):
#   aggressor PWM_HDRV : dV=5 V, Rout=100 ohm, 30x10 um trace, 4000 um long
#   victims  ISEN_CS1 (Rin=200 ohm) and ISEN_CS2 (Rin=1000 ohm), Cin=1 pF, d=100 um, 150 um long
import math
mu0 = 4e-7*math.pi

def Lself(L,w,t): return mu0*L/(2*math.pi)*(math.log(2*L/(w+t))+0.5+0.2235*(w+t)/L)  # trace self-L
def Mmut(d,L):    return mu0*L/(2*math.pi)*(math.asinh(L/d)-math.sqrt(1+(d/L)**2)+d/L) # parallel mutual

# --- default-populated geometry / drive ---
Vagg, R1, C1 = 5.0, 100.0, 100e-12
La = Lself(4000e-6, 30e-6, 10e-6)          # aggressor self-inductance (4 mm)
Lv = Lself( 150e-6, 30e-6, 10e-6)          # victim self-inductance (150 um)
M  = Mmut(100e-6, min(4000e-6, 150e-6))    # mutual over 150 um overlap at d = 100 um

def simulate(R2, C2):
    """RK4 solve of x=[I1,Vc1,I2,Vc2]; returns (times, victim voltage across C2)."""
    det = La*Lv - M*M
    def f(x):
        I1,Vc1,I2,Vc2 = x
        b1 = Vagg - R1*I1 - Vc1                 # primary branch
        b2 =       - R2*I2 - Vc2                # secondary branch (no source)
        return (( Lv*b1 - M*b2)/det, I1/C1, (-M*b1 + La*b2)/det, I2/C2)
    dt = min(La/R1, Lv/R2, R2*C2, math.sqrt(Lv*C2))/25
    T  = max(8*R2*C2, 40*La/R1, 3e-9); N = min(60000, math.ceil(T/dt)); dt = T/N
    x=[0.0]*4; ts=[0.0]; vs=[0.0]
    for i in range(N):
        k1=f(x); x2=[x[j]+.5*dt*k1[j] for j in range(4)]
        k2=f(x2); x3=[x[j]+.5*dt*k2[j] for j in range(4)]
        k3=f(x3); x4=[x[j]+dt*k3[j] for j in range(4)]; k4=f(x4)
        for j in range(4): x[j]+=dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j])
        ts.append((i+1)*dt); vs.append(x[3])
    return ts, vs

def ascii_plot(ts, vs, title, W=60, H=15):
    """ASCII waveform of V(t) with the peak marked and its value annotated."""
    ip = max(range(len(vs)), key=lambda i: abs(vs[i])); vpk, tpk = vs[ip], ts[ip]
    Tplot = min(max(9*tpk, 300e-12), ts[-1])
    lo, hi = min(0.0, min(vs)), max(0.0, max(vs))
    if hi==lo: hi = lo+1e-9
    row = lambda v: min(H-1, max(0, int(round((hi-v)/(hi-lo)*(H-1)))))
    col = lambda t: min(W-1, max(0, int(round(t/Tplot*(W-1)))))
    grid=[[' ']*W for _ in range(H)]; r0=row(0.0)
    for c in range(W): grid[r0][c]='-'                       # zero axis
    pkc=col(tpk); j=0
    for c in range(W):
        tt=c/(W-1)*Tplot
        while j+1<len(ts) and ts[j+1]<tt: j+=1
        grid[row(vs[j])][c] = '*' if c==pkc else 'o'         # curve, '*' at peak
    ann=' <- peak = %.3f mV'%(vpk*1e3); rp=row(vpk)          # inline annotation
    s=max(0, min(pkc+2, W-len(ann)-1))
    for i,ch in enumerate(ann):
        if s+i<W: grid[rp][s+i]=ch
    ylab={0:'%7.3f'%(hi*1e3), r0:'%7.3f'%0.0, H-1:'%7.3f'%(lo*1e3)}
    out=[title]
    for ri,rc in enumerate(grid): out.append(ylab.get(ri,'       ')+' |'+''.join(rc))
    out.append('        +'+'-'*W)
    out.append('        0'+' '*(W//2-4)+'time [ps]'+' '*(W//2-13)+'%d'%round(Tplot*1e12))
    return '\n'.join(out), vpk

print("La=%.3f nH   Lv=%.4f nH   M=%.2f pH   (Vagg=%.0f V, R1=%.0f ohm, Cin=1 pF)\n"
      % (La*1e9, Lv*1e9, M*1e12, Vagg, R1))
for name, Rin in [("ISEN_CS1", 200.0), ("ISEN_CS2", 1000.0)]:
    ts, vs = simulate(Rin, 1e-12)
    plot, vpk = ascii_plot(ts, vs, "PWM_HDRV -> %s  (Rin=%.0f ohm)  [victim V across C2, mV]"%(name,Rin))
    print(plot); print("        Verr = 1.000 mV   ->   %s\n"%("PASS" if abs(vpk)<=1e-3 else "FAIL"))
