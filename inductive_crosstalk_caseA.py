#!/usr/bin/env python3
# ============================================================================
#  Inductive Crosstalk (Aggressor -> Victim) -- CASE A  ("far", 1 mV victim)
#  Reproduces Section 3/4 (Table 1) of Inductive_Crosstalk rev9.
#
#  One-click reproducible:  python3 inductive_crosstalk_caseA.py
#  Deps: numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)
#
#  Model: primary series-RLC (I1) mutually coupled by M to a secondary
#         L2-R2-C2 loop (I2). Victim = voltage across C2.
#  States x = [I1, Vc1, I2, Vc2].  Coupled inductor matrix inverted exactly.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- Geometry / self-inductance of a 1 mm wire (30 um wide, 10 um thick) ----
mu0, l, w, t = 4e-7*np.pi, 1e-3, 30e-6, 10e-6
L1 = L2 = mu0*l/(2*np.pi)*(np.log(2*l/(w+t)) + 0.5 + 0.2235*(w+t)/l)   # ~0.88 nH

# ---- CASE A parameters (from Table 1: "far", 1 mV victim) -------------------
Vagg = 1.0            # aggressor swing         [V]
R1   = 50.0           # primary resistance      [ohm]
C1   = 100e-12        # primary capacitance     [F]
R2   = 50.0           # victim resistance (light filter)   [ohm]
C2   = 1e-12          # victim capacitance                 [F]
M    = 4.26e-12       # mutual inductance, Case A (d = 23.5 mm)  [H]

TABLE_VICTIM_PK = 1.0e-3     # Table 1 value to check against  [V] (1 mV)
TABLE_EMF_PK    = 4.82e-3    # Table 1 pre-filter EMF peak      [V]
TABLE_I1_PK     = 19.7e-3    # Table 1 primary peak current     [A]

det = L1*L2 - M*M            # determinant of the [[L1,M],[M,L2]] matrix

def rhs(tt, x):              # x = [I1, Vc1, I2, Vc2]
    I1, Vc1, I2, Vc2 = x
    b1 = Vagg - R1*I1 - Vc1          # primary branch voltage
    b2 =      - R2*I2 - Vc2          # secondary branch (no source)
    dI1 = ( L2*b1 - M*b2)/det        # invert 2x2 inductor matrix
    dI2 = (-M*b1 + L1*b2)/det
    return [dI1, I1/C1, dI2, I2/C2]

# ---- Solve (dense output so we get a smooth waveform) ----------------------
T = 40e-9
sol = solve_ivp(rhs, (0, T), [0, 0, 0, 0], method='RK45',
                rtol=1e-11, atol=1e-16, dense_output=True, max_step=1e-12)

tt   = np.linspace(0, T, 400000)
I1, Vc1, I2, Vc2 = sol.sol(tt)
dI1  = np.gradient(I1, tt)      # dI1/dt
EMF  = M*dI1                    # coupled EMF onto the victim loop  (M dI1/dt)
Vvic = Vc2                      # victim = voltage across C2

# ---- Peaks -----------------------------------------------------------------
Vvic_pk = np.max(np.abs(Vvic))
EMF_pk  = np.max(np.abs(EMF))
I1_pk   = np.max(I1)
att     = EMF_pk/Vvic_pk

def row(name, sim, tab, unit, scale):
    err = 100*(sim-tab)/tab
    print(f"  {name:<26s} {sim*scale:9.3f}    {tab*scale:9.3f}   {err:+6.1f}%   {unit}")

print("\n============ CASE A  (far, target 1 mV victim) =====================")
print(f"  L1 = L2 = {L1*1e9:.3f} nH   M = {M*1e12:.2f} pH   tau2 = R2*C2 = {R2*C2*1e12:.0f} ps")
print("  Quantity                     PY-sim       Table    diff")
print("  " + "-"*58)
row("Victim peak (across C2)", Vvic_pk, TABLE_VICTIM_PK, "mV", 1e3)
row("Induced EMF peak",        EMF_pk,  TABLE_EMF_PK,    "mV", 1e3)
row("Primary peak current I1", I1_pk,   TABLE_I1_PK,     "mA", 1e3)
print(f"  Filter attenuation (EMF/victim) = {att:.2f}x   (table: ~5x)")
print("====================================================================\n")

# ---- Plot: I1, EMF, victim  (zoomed to the first ~300 ps of action) --------
tps = tt*1e12                                  # time in ps
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

ax[0].plot(tps, I1*1e3, color='#0052CC')
ax[0].axhline(Vagg/R1*1e3, ls='--', color='#DE350B', lw=1,
              label=f'I1,pk ~ Vagg/R1 = {Vagg/R1*1e3:.1f} mA')
ax[0].set_ylabel('I1  [mA]'); ax[0].legend(loc='lower right', fontsize=8)
ax[0].set_title('Case A  -  Inductive crosstalk transient (d = 23.5 mm, M = 4.26 pH)')

ax[1].plot(tps, EMF*1e3, color='#36B37E')
ax[1].axhline( EMF_pk*1e3, ls=':', color='#7A869A', lw=1)
ax[1].axhline(-EMF_pk*1e3, ls=':', color='#7A869A', lw=1)
ax[1].set_ylabel('EMF = M dI1/dt  [mV]')
ax[1].annotate(f'EMF peak = {EMF_pk*1e3:.2f} mV', (0.98, 0.9),
               xycoords='axes fraction', ha='right', fontsize=8)

ax[2].plot(tps, Vvic*1e3, color='#6554C0')
ax[2].axhline(Vvic_pk*1e3, ls='--', color='#DE350B', lw=1,
              label=f'victim peak = {Vvic_pk*1e3:.3f} mV  (table 1 mV)')
ax[2].set_ylabel('V_victim across C2 [mV]')
ax[2].set_xlabel('time  [ps]')
ax[2].legend(loc='lower right', fontsize=8)

ax[2].set_xlim(0, 300)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
out = 'inductive_crosstalk_caseA.png'
fig.savefig(out, dpi=130)
print(f"Saved waveform figure -> {out}")
plt.show()
