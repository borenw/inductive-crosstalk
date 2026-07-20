# Coupled 4-state inductive-crosstalk model — copy, paste, run (pure standard library).
#   python3 this_file.py          # no numpy / scipy / matplotlib needed; prints an ASCII waveform
#
# Reproduces the DEFAULT case populated in the Spec Checker (section 7):
#   aggressor PWM_HDRV : dV=5 V, Rout=100 ohm, 30x10 um trace, 4000 um long
#   victims  ISEN_CS1 (Rin=200 ohm) and ISEN_CS2 (Rin=1000 ohm), Cin=1 pF, d=100 um, 150 um long
# The waveform is drawn with a Verr spec line; the part of the trace BEYOND spec is red, the
# part within spec is green (ANSI colors — most terminals show them).
import math
mu0 = 4e-7*math.pi
RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

def Lself(L,w,t): return mu0*L/(2*math.pi)*(math.log(2*L/(w+t))+0.5+0.2235*(w+t)/L)  # trace self-L
def Mmut(d,L):    return mu0*L/(2*math.pi)*(math.asinh(L/d)-math.sqrt(1+(d/L)**2)+d/L) # parallel mutual

# --- default-populated geometry / drive ---
Vagg, R1, C1 = 5.0, 100.0, 100e-12
La = Lself(4000e-6, 30e-6, 10e-6)      # aggressor self-inductance (4 mm trace)
Lv = Lself( 150e-6, 30e-6, 10e-6)      # victim self-inductance (150 um trace)
M  = Mmut(100e-6, min(4000e-6, 150e-6))# mutual over the 150 um overlap at d = 100 um

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

def _paint(chars, cols):
    """chars: list of characters; cols: list of None/'r'/'g' -> ANSI string (run-length grouped)."""
    out=""; i=0; n=len(chars)
    while i<n:
        c=cols[i]
        if c is None: out+=chars[i]; i+=1
        else:
            j=i
            while j<n and cols[j]==c: j+=1
            out+=(RED if c=='r' else GREEN)+"".join(chars[i:j])+RESET; i=j
    return out

def ascii_plot(ts, vs, title, Verr, W=60, H=17):
    """ASCII waveform with a Verr spec line; trace beyond +/-Verr is red, within is green."""
    ip = max(range(len(vs)), key=lambda i: abs(vs[i])); vpk, tpk = vs[ip], ts[ip]
    Tplot = min(max(9*tpk, 300e-12), ts[-1])
    lo = min(0.0, min(vs), -Verr*1.15); hi = max(0.0, max(vs), Verr*1.15)  # keep +/-Verr in view
    row = lambda v: min(H-1, max(0, int(round((hi-v)/(hi-lo)*(H-1)))))
    col = lambda t: min(W-1, max(0, int(round(t/Tplot*(W-1)))))
    grid=[[' ']*W for _ in range(H)]; cols=[[None]*W for _ in range(H)]
    rSp, rSm, r0 = row(Verr), row(-Verr), row(0.0)
    for c in range(W):                                   # dashed spec lines at +/-Verr
        if c%3==0:
            if grid[rSp][c]==' ': grid[rSp][c]='.'
            if grid[rSm][c]==' ': grid[rSm][c]='.'
    for c in range(W): grid[r0][c]='-'                   # zero axis
    pkc=col(tpk); j=0
    for c in range(W):                                   # curve (on top), colored by spec
        tt=c/(W-1)*Tplot
        while j+1<len(ts) and ts[j+1]<tt: j+=1
        r=row(vs[j]); beyond = abs(vs[j])>Verr
        grid[r][c] = '*' if c==pkc else 'o'
        cols[r][c] = 'r' if beyond else 'g'
    ann=' <- peak = %.3f mV'%(vpk*1e3); rp=row(vpk); s=max(0,min(pkc+2,W-len(ann)-1))
    for i,ch in enumerate(ann):                          # peak label (color = peak's spec state)
        if s+i<W and grid[rp][s+i]==' ':
            grid[rp][s+i]=ch; cols[rp][s+i] = 'r' if abs(vpk)>Verr else 'g'
    slab=' spec (Verr)'                                  # spec-line tag on the -Verr row
    s2=W-len(slab)
    for i,ch in enumerate(slab):
        if 0<=s2+i<W and grid[rSm][s2+i] in ' .': grid[rSm][s2+i]=ch
    ylab={0:'%7.3f'%(hi*1e3), r0:'%7.3f'%0.0, rSm:'%7.3f'%(-Verr*1e3), H-1:'%7.3f'%(lo*1e3)}
    out=[title]
    for ri in range(H):
        out.append(ylab.get(ri,'       ')+' |'+_paint(grid[ri], cols[ri]))
    out.append('        +'+'-'*W)
    out.append('        0'+' '*(W//2-4)+'time [ps]'+' '*(W//2-13)+'%d'%round(Tplot*1e12))
    return '\n'.join(out), vpk

print("La=%.3f nH   Lv=%.4f nH   M=%.2f pH   (Vagg=%.0f V, R1=%.0f ohm, Cin=1 pF)\n"
      % (La*1e9, Lv*1e9, M*1e12, Vagg, R1))
Verr = 1e-3
for name, Rin in [("ISEN_CS1", 200.0), ("ISEN_CS2", 1000.0)]:
    ts, vs = simulate(Rin, 1e-12)
    plot, vpk = ascii_plot(ts, vs, "PWM_HDRV -> %s  (Rin=%.0f ohm)  [victim V across C2, mV]"%(name,Rin), Verr)
    ok = abs(vpk) <= Verr
    print(plot)
    print("        Verr = %.3f mV   ->   %s%s%s\n" % (Verr*1e3, GREEN if ok else RED, "PASS" if ok else "FAIL", RESET))
