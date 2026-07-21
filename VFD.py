# Updated rectifier, by avoiding abrupt current drop to zero

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # or "TkAgg" if Qt not installed
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import compute_fft

# ---------------- USER COMMANDS ----------------
rpm_des = 280 # motor rpm
n_poles = 48
f_out_des = rpm_des/60 * n_poles / 2         # desired motor electrical frequency [Hz]
print("f engine =", f_out_des)
P_des = 3000e3          # desired average motor electrical power [W] (RL-only motor model)

# Simple optimization settings
N_candidates = 10
f_sw_choices = np.array([2400])  # [Hz]
m_min, m_max = 0.05, 0.95

# ---------------- PARAMETERS ----------------
V_ll_rms = 400.0
f_grid = 60.0
w_grid = 2 * np.pi * f_grid
V_phase_peak = np.sqrt(2) * V_ll_rms / np.sqrt(3)

R_s = 0.05
L_s = 5e-6
C_dc = 5e-2
V_diode = 4

# --- Motor parameters (simple RL delta branches) ---
R_m = 0.2e-1 # 0.2 - 0.5 resistance of each phase of the motor
L_m = 5e-3 # 5e-3 - 50e-3

# ---------------- TIME ----------------
freq_microp = 40e3 # Hz
t_end = 30*1/f_grid
dt = 1/freq_microp
t = np.arange(0, t_end, dt)
thr = 1e-2 # threshold for FFT peak picking
n_fft_cycles = 10          # analyze last N electrical cycles (assumed steady-state)

# ---------------- THREE-PHASE SOURCE ----------------
Va_src = V_phase_peak * np.sin(w_grid * t)
Vb_src = V_phase_peak * np.sin(w_grid * t - 2*np.pi/3)
Vc_src = V_phase_peak * np.sin(w_grid * t - 4*np.pi/3)

# ---------------- PWM helpers ----------------
def tri_carrier(time, f_sw): # comparator signal - triangle wave
    phase = (time * f_sw) % 1.0          # 0 after periods, otherwise linearly varying, with the value of time*f_sw
    return 1.0 - 4.0 * np.abs(phase - 0.5)  # -1..+1 perfect triangle

def spwm_gates(time, f_out, f_sw, m): # controller of the transistor switches
    w = 2*np.pi*f_out
    ref_a = m*np.sin(w*time) # reference
    ref_b = m*np.sin(w*time - 2*np.pi/3)
    ref_c = m*np.sin(w*time - 4*np.pi/3)
    car = tri_carrier(time, f_sw) # comparator

    # switching law
    Sa_p = 1 if ref_a > car else 0
    Sb_p = 1 if ref_b > car else 0
    Sc_p = 1 if ref_c > car else 0
    Sa_n = 1 - Sa_p
    Sb_n = 1 - Sb_p
    Sc_n = 1 - Sc_p
    return Sa_p, Sb_p, Sc_p, Sa_n, Sb_n, Sc_n

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def fundamental_phasor(x, time, f0):
    w0 = 2*np.pi*f0
    return np.mean(x * np.exp(-1j*w0*time))

def thd_like(x, time, f0): # compare actual signal amo and phase to desired
    X1 = fundamental_phasor(x, time, f0) # actual signal
    amp = 2*np.abs(X1)
    phi = np.angle(X1)
    x1 = amp*np.sin(2*np.pi*f0*time + phi) # amp and phi of the actual signal, f0 of the desired
    num = np.sqrt(np.mean((x - x1)**2)) # RMS of the error
    den = np.sqrt(np.mean(x1**2)) + 1e-12 # prevent dividing by zero
    return num / den, amp, phi # amplitude and phase refer to desired

# ---------------- Simulation (optionally log gates for plotting) ----------------
def simulate_once(f_out, f_sw, m, P_target=None, log_gates=False):
    V_dc = np.zeros_like(t)
    I_L  = np.zeros_like(t)

    Iab = np.zeros_like(t)
    Ibc = np.zeros_like(t)
    Ica = np.zeros_like(t)

    Va_inv = np.zeros_like(t) # voltages at motor electrical poles
    Vb_inv = np.zeros_like(t)
    Vc_inv = np.zeros_like(t)

    # plot switches logs if desired
    if log_gates:
        Sa_p_log = np.zeros_like(t, dtype=np.int8)
        Sb_p_log = np.zeros_like(t, dtype=np.int8)
        Sc_p_log = np.zeros_like(t, dtype=np.int8)
        Sa_n_log = np.zeros_like(t, dtype=np.int8)
        Sb_n_log = np.zeros_like(t, dtype=np.int8)
        Sc_n_log = np.zeros_like(t, dtype=np.int8)
        ref_a_log = np.zeros_like(t)
        car_log   = np.zeros_like(t)
    else:
        Sa_p_log = Sb_p_log = Sc_p_log = None
        Sa_n_log = Sb_n_log = Sc_n_log = None
        ref_a_log = car_log = None

    for i in range(len(t)-1):
        # ---------- RECTIFIER ----------
        Vmax = max(Va_src[i], Vb_src[i], Vc_src[i])
        Vmin = min(Va_src[i], Vb_src[i], Vc_src[i])
        V_rect = Vmax - Vmin - 2 * V_diode

        # Diodes conduct if:
        # 1) the bridge voltage is high enough to start conduction, or
        # 2) current is already flowing and must decay continuously through L_s
        if I_L[i] > 0.0 or V_rect > V_dc[i]:
            dI_dt = (V_rect - R_s * I_L[i] - V_dc[i]) / L_s
            I_next = I_L[i] + dI_dt * dt

            if I_next > 0.0:
                I_L[i + 1] = I_next
                I_dc_in = I_L[i + 1]
            else:
                I_L[i + 1] = 0.0
                I_dc_in = 0.0
        else:
            I_L[i + 1] = 0.0
            I_dc_in = 0.0

        # ---------- INVERTER GATES (SPWM) ----------
        Sa_p, Sb_p, Sc_p, Sa_n, Sb_n, Sc_n = spwm_gates(t[i], f_out, f_sw, m)

        if log_gates:
            Sa_p_log[i], Sb_p_log[i], Sc_p_log[i] = Sa_p, Sb_p, Sc_p
            Sa_n_log[i], Sb_n_log[i], Sc_n_log[i] = Sa_n, Sb_n, Sc_n
            ref_a_log[i] = m*np.sin(2*np.pi*f_out*t[i])
            car_log[i]   = tri_carrier(t[i], f_sw)

        # ---------- ELECTRICAL POLES VOLTAGES ----------
        Va = (2*Sa_p - 1) * (V_dc[i] / 2.0) # ref = 0V, V_dc = [Vmax/2, Vmin_2]
        Vb = (2*Sb_p - 1) * (V_dc[i] / 2.0)
        Vc = (2*Sc_p - 1) * (V_dc[i] / 2.0)
        Va_inv[i], Vb_inv[i], Vc_inv[i] = Va, Vb, Vc

        # ---------- DELTA BRANCH VOLTAGES ----------
        Vab = Va - Vb
        Vbc = Vb - Vc
        Vca = Vc - Va

        # ---------- DELTA RL DYNAMICS ----------
        Iab[i+1] = Iab[i] + ((Vab - R_m*Iab[i]) / L_m) * dt
        Ibc[i+1] = Ibc[i] + ((Vbc - R_m*Ibc[i]) / L_m) * dt
        Ica[i+1] = Ica[i] + ((Vca - R_m*Ica[i]) / L_m) * dt

        # ---------- LINE CURRENTS INTO DELTA ----------
        Ia_line = Iab[i+1] - Ica[i+1]
        Ib_line = Ibc[i+1] - Iab[i+1]
        Ic_line = Ica[i+1] - Ibc[i+1]

        # ---------- DC CURRENT DRAWN BY INVERTER (ideal, no induced EMF, no losses) ----------
        I_dc_inv = Sa_p*Ia_line + Sb_p*Ib_line + Sc_p*Ic_line

        # ---------- DC CAPACITOR ----------
        V_dc[i+1] = V_dc[i] + ((I_dc_in - I_dc_inv) / C_dc) * dt
        if V_dc[i+1] < 0.0:
            V_dc[i+1] = 0.0

    Va_inv[-1], Vb_inv[-1], Vc_inv[-1] = Va_inv[-2], Vb_inv[-2], Vc_inv[-2] # the last value would remain zero

    # steady-state metrics (last 3 cycles)
    n_cycles = 3
    T0 = 1.0 / f_out
    idx = t >= (t_end - n_cycles*T0)
    tw = t[idx]

    Iab_w, Ibc_w, Ica_w = Iab[idx], Ibc[idx], Ica[idx]
    Va_w, Vb_w, Vc_w = Va_inv[idx], Vb_inv[idx], Vc_inv[idx]
    Vab_w, Vbc_w, Vca_w = Va_w - Vb_w, Vb_w - Vc_w, Vc_w - Va_w

    p_inst = Vab_w*Iab_w + Vbc_w*Ibc_w + Vca_w*Ica_w
    P_rms = np.sqrt(np.mean(np.square(p_inst)))

    # check metrics comparing actual signal with desired
    thd_a, amp_a, ph_a = thd_like(Iab_w, tw, f_out) # measures how far from amplitudes and phase from desired signal
    thd_b, amp_b, ph_b = thd_like(Ibc_w, tw, f_out)
    thd_c, amp_c, ph_c = thd_like(Ica_w, tw, f_out)

    d_ab = wrap_angle(ph_b - ph_a)
    d_ac = wrap_angle(ph_c - ph_a)
    phase_pen = (wrap_angle(d_ab + 2*np.pi/3)**2 + wrap_angle(d_ac - 2*np.pi/3)**2)

    amps = np.array([amp_a, amp_b, amp_c])
    bal_pen = np.var(amps) / (np.mean(amps)**2 + 1e-12) # signal noise measure

    if P_target is None or P_target <= 0:
        pow_pen = 0.0
    else:
        pow_pen = ((P_rms - P_target) / (P_target + 1e-9))**2

    obj = 3.0*pow_pen + 1.0*(thd_a + thd_b + thd_c)/3.0 + 0.3*phase_pen + 0.2*bal_pen

    return {
        "obj": obj,
        "P_rms": P_rms,
        "thd": (thd_a, thd_b, thd_c),
        "V_dc": V_dc,
        "I_L": I_L,
        "Iab": Iab, "Ibc": Ibc, "Ica": Ica,
        "Sa_p_log": Sa_p_log, "Sb_p_log": Sb_p_log, "Sc_p_log": Sc_p_log,
        "Sa_n_log": Sa_n_log, "Sb_n_log": Sb_n_log, "Sc_n_log": Sc_n_log,
        "ref_a_log": ref_a_log,
        "car_log": car_log,
    }

# ---------------- Optimization (minimal random search) ----------------
rng = np.random.default_rng(0)
best = None

for _ in range(N_candidates):
    f_sw = float(rng.choice(f_sw_choices))
    m = float(rng.uniform(m_min, m_max))
    sim = simulate_once(f_out_des, f_sw, m, P_target=P_des, log_gates=False)
    # update best solution until now
    if best is None or sim["obj"] < best["obj"]:
        best = {**sim, "f_sw": f_sw, "m": m} # it will contain best solution at the end

print("BEST:")
print(f"  f_sw = {best['f_sw']:.0f} Hz")
print(f"  m    = {best['m']:.3f}")
print(f"  Prms = {best['P_rms']:.1f} W (target {P_des:.1f} W)")
print(f"  THD_like: a={best['thd'][0]:.3f}, b={best['thd'][1]:.3f}, c={best['thd'][2]:.3f}")

# Re-simulate best candidate WITH gate logging
best2 = simulate_once(f_out_des, best["f_sw"], best["m"], P_target=P_des, log_gates=True)

V_dc, I_L = best2["V_dc"], best2["I_L"]
Iab, Ibc, Ica = best2["Iab"], best2["Ibc"], best2["Ica"]

Sa_p_log, Sb_p_log, Sc_p_log = best2["Sa_p_log"], best2["Sb_p_log"], best2["Sc_p_log"]
Sa_n_log, Sb_n_log, Sc_n_log = best2["Sa_n_log"], best2["Sb_n_log"], best2["Sc_n_log"]
ref_a_log, car_log = best2["ref_a_log"], best2["car_log"]

# ---------------- PLOTS (non-blocking, multiple windows) ----------------
plt.ion() # interactive mode ON

plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(t, V_dc)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("DC Bus Voltage")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, I_L)
plt.title("Rectifier Inductor Current")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid(True)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

plt.figure(figsize=(12, 5))
plt.plot(t, Iab, label="Iab")
plt.plot(t, Ibc, label="Ibc")
plt.plot(t, Ica, label="Ica")
plt.title(f"Delta Branch Currents | f_out={f_out_des:.0f} Hz, f_sw={best['f_sw']:.0f} Hz, m={best['m']:.3f}")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.001)

# ---------------- Comparator triangle + switching history (stacked) ----------------
t_zoom_start = t_end - 1*1/f_out_des
idxz = t >= t_zoom_start

plt.figure(figsize=(12, 7))

ax1 = plt.subplot(3, 1, 1)
plt.plot(t[idxz], ref_a_log[idxz], label="ref_a = m*sin(wt)")
plt.plot(t[idxz], car_log[idxz], label="carrier (triangle)")
plt.title("SPWM comparator (phase A): Sa_p = 1 when ref_a > carrier")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend(loc="upper right")

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.step(t[idxz], Sa_p_log[idxz], where="post", label="Sa_p")
plt.title ("Sa_p (offset)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend(loc="upper right")

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.step(t[idxz], Sa_p_log[idxz], where="post", label="Sa_p")
plt.step(t[idxz], Sb_p_log[idxz] + 1.2, where="post", label="Sb_p (offset)")
plt.step(t[idxz], Sc_p_log[idxz] + 2.4, where="post", label="Sc_p (offset)")
plt.step(t[idxz], Sa_n_log[idxz] + 3.8, where="post", label="Sa_n (offset)")
plt.step(t[idxz], Sb_n_log[idxz] + 5.0, where="post", label="Sb_n (offset)")
plt.step(t[idxz], Sc_n_log[idxz] + 6.2, where="post", label="Sc_n (offset)")
plt.title("Switching history (upper and lower gates, offset)")
plt.yticks([0,1, 1.2,2.2, 2.4,3.4, 3.8,4.8, 5.0,6.0, 6.2,7.2],
           ["0","1","0","1","0","1","0","1","0","1","0","1"])
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend(loc="upper right", ncol=3, fontsize=9)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

# --- COMPUTE FFT --- #
# Current

t_fft_start = t_end - (n_fft_cycles / f_out_des)
idx_fft = t >= t_fft_start
x_I = Iab[idx_fft].copy()
compute_fft.compute_fft(t[idx_fft].copy(), x_I, thr, freq_microp/2, "Current", "[A]", "Current")

# Squared current
F = Iab**2
x_F = F[idx_fft].copy()
compute_fft.compute_fft(t[idx_fft].copy(), x_F, thr, freq_microp/2, "Squared current", "[A^2]", "Squared current")

plt.ioff()
plt.show()
