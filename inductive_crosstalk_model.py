# Coupled 4-state inductive-crosstalk model — copy, paste, run.
#   pip install numpy scipy matplotlib   →   python3 this_file.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- geometry: 1 mm wire, 30 um wide, 10 um thick -> self-inductance ---
mu0, l, w, t = 4e-7*np.pi, 1e-3, 30e-6, 10e-6
L1 = L2 = mu0*l/(2*np.pi)*(np.log(2*l/(w+t)) + 0.5 + 0.2235*(w+t)/l)   # ~0.884 nH

# --- fixed circuit values ---
R1, C1, Vagg = 50.0, 100e-12, 1.0      # primary  (aggressor loop)
R2, C2       = 50.0, 1e-12             # victim filter (light: tau2 = 50 ps)

def Mof(d):                            # mutual inductance of two parallel wires, spacing d
    return mu0*l/(2*np.pi)*(np.arcsinh(l/d) - np.sqrt(1+(d/l)**2) + d/l)

def solve(M):                          # returns (t, victim-voltage-across-C2)
    det = L1*L2 - M*M                  # mutual-inductance matrix determinant
    def rhs(t, x):                     # x = [I1, Vc1, I2, Vc2]
        I1, Vc1, I2, Vc2 = x
        b1 = Vagg - R1*I1 - Vc1        # primary branch
        b2 =       - R2*I2 - Vc2       # secondary branch (no source)
        dI1 = ( L2*b1 - M*b2)/det
        dI2 = (-M*b1 + L1*b2)/det
        return [dI1, I1/C1, dI2, I2/C2]
    s = solve_ivp(rhs, (0, 8e-9), [0,0,0,0], rtol=1e-10, atol=1e-15, max_step=2e-12)
    return s.t, s.y[3]                 # victim = voltage across C2

# --- run the two headline cases and show the result ---
cases = {"Case A  (far,  d=23.5 mm)": Mof(23.5e-3),
         "Case B  (near, d=0.088 mm)": Mof(0.088e-3)}
print(f"L1 = L2 = {L1*1e9:.3f} nH   (R1={R1:.0f} Ohm, R2={R2:.0f} Ohm, C2={C2*1e12:.0f} pF, Vagg={Vagg} V)\n")
plt.figure(figsize=(8,4))
for name, M in cases.items():
    tt, Vvic = solve(M)
    Vpk = np.max(np.abs(Vvic))
    print(f"{name:28s}  M = {M*1e12:6.2f} pH   ->   victim peak = {Vpk*1e3:8.3f} mV")
    plt.plot(tt*1e12, Vvic*1e3, label=f"{name}  (peak {Vpk*1e3:.2f} mV)")
plt.xlabel("time  [ps]"); plt.ylabel("V_victim across C2  [mV]")
plt.title("Coupled 4-state inductive crosstalk — filtered victim"); plt.xlim(0,300)
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
