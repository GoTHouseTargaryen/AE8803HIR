import numpy as np
import matplotlib.pyplot as plt
import sys

"""
HW5 SIMULATION: CRTBP + TFRBP (Lecture 23/24 Splitting)

METHODOLOGY:

1. CRTBP:
   - Split into Kinetic (Drift), Potential (Kick), and Coriolis (Rotation).
   - Normalized units used for stability.

2. TFRBP (Rigid Body):
   - Implements Hamiltonian Splitting from Lecture 24: H' = H1' + H2'
   - H1' (Kinematic): Evolves Orientation ([q] or DCM) with constant Momentum (Pi).
     Solution: R(t) = R(0) * exp(skew(w)*t)
   - H2' (Dynamic): Evolves Momentum (Pi) with constant Orientation.
     Solution: Euler's equations, solved via further splitting into 3 axis rotations.

3. Integrator:
   - Yoshida 6th-Order Composition of a symmetric 2nd-order base stepper.
"""

# ============================================================================
# 1. Physical Constants & Normalization
# ============================================================================

class SystemConstants:
    """Constants from Table 1."""
    # SI Units
    M_earth = 5.9722e24  # kg
    M_moon  = 7.3477e22  # kg
    R_EM    = 384400.0e3 # m (Distance)
    G       = 6.67430e-11 

    # Derived SI
    mu_SI = G * (M_earth + M_moon)
    n_mean = np.sqrt(mu_SI / R_EM**3) # Mean motion (rad/s)
    T_orbit = 2 * np.pi / n_mean

    # Rigid Body Constants (Table 1)
    # Ellipsoid 2a=20, 2b=10, 2c=8 => a=10, b=5, c=4
    rb_a, rb_b, rb_c = 10.0, 5.0, 4.0
    rb_rho = 1000.0 # kg/m^3
    
    # Mass and Inertia
    vol = (4.0/3.0) * np.pi * rb_a * rb_b * rb_c
    rb_mass = vol * rb_rho
    Ix = (1.0/5.0) * rb_mass * (rb_b**2 + rb_c**2)
    Iy = (1.0/5.0) * rb_mass * (rb_a**2 + rb_c**2)
    Iz = (1.0/5.0) * rb_mass * (rb_a**2 + rb_b**2)
    
    # Initial Body Angular Velocity (Table 1)
    w0_body = np.array([1e-7, 1e-8, 1e-5])

    # CRTBP Normalization Factors
    LU = R_EM
    TU = 1.0 / n_mean 
    MU = M_earth + M_moon
    
    # Normalized Parameters
    mu = M_moon / (M_earth + M_moon) 
    
    @staticmethod
    def to_years(t_normalized):
        t_seconds = t_normalized * SystemConstants.TU
        return t_seconds / (365.25 * 86400.0)

# ============================================================================
# 2. Dynamics Classes
# ============================================================================

class CRTBP_Normalized:
    """Normalized CRTBP with Drift-Kick-Rotation splitting."""
    def __init__(self, mu):
        self.mu = mu
        self.mu1 = 1 - mu
        self.xi1 = -mu
        self.xi2 = 1 - mu

    def get_energy(self, q, p):
        ξ, η = q[0], q[1]
        p_ξ, p_η = p[0], p[1]
        r1 = np.sqrt((ξ - self.xi1)**2 + η**2)
        r2 = np.sqrt((ξ - self.xi2)**2 + η**2)
        T = 0.5 * (p_ξ**2 + p_η**2)
        C = (η*p_ξ - ξ*p_η)
        V = -self.mu1/r1 - self.mu/r2
        return T + C + V

    def drift(self, q, p, dt):
        q_new = q + p * dt
        return q_new, p

    def kick(self, q, p, dt):
        ξ, η = q[0], q[1]
        r1 = np.sqrt((ξ - self.xi1)**2 + η**2)
        r2 = np.sqrt((ξ - self.xi2)**2 + η**2)
        fac1 = self.mu1 / (r1**3)
        fac2 = self.mu / (r2**3)
        dVdξ = fac1*(ξ - self.xi1) + fac2*(ξ - self.xi2)
        dVdη = fac1*η + fac2*η
        p_new = np.zeros_like(p)
        p_new[0] = p[0] - dVdξ * dt
        p_new[1] = p[1] - dVdη * dt
        return q, p_new

    def rot(self, q, p, dt):
        c = np.cos(dt)
        s = np.sin(dt)
        ξ, η = q[0], q[1]
        p_ξ, p_η = p[0], p[1]
        ξ_new  = ξ*c + η*s
        η_new  = -ξ*s + η*c
        p_ξ_new = p_ξ*c + p_η*s
        p_η_new = -p_ξ*s + p_η*c
        return np.array([ξ_new, η_new]), np.array([p_ξ_new, p_η_new])


class RigidBody_Split:
    """
    Torque-Free Rigid Body using Lecture 24 Splitting Method.
    H' = H1' (Kinematic) + H2' (Dynamic).
    """
    def __init__(self, I, w0):
        self.I = I # Principal Moments (Ix, Iy, Iz)
        self.w = w0 # Body angular velocity
        self.DCM = np.eye(3) # Body-to-Inertial Matrix
        
    def get_angular_momentum_inertial(self):
        # L_body = I * w
        L_body = self.I * self.w
        # L_inertial = DCM @ L_body
        return self.DCM @ L_body

    def get_energy(self):
        return 0.5 * np.sum(self.I * self.w**2)

    def kinematic_step(self, dt):
        """
        H1': Evolve Orientation (DCM) with constant Momentum (w).
        Solution: R(t) = R(0) * exp(skew(w)*t)
        """
        # Magnitude of rotation
        w_norm = np.linalg.norm(self.w)
        if w_norm < 1e-16:
            return
            
        angle = w_norm * dt
        axis = self.w / w_norm
        
        # Rodrigues Rotation Formula
        # K is skew-symmetric matrix of axis
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        I = np.eye(3)
        R_step = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Update DCM (Post-multiply because w is in Body frame)
        self.DCM = self.DCM @ R_step

    def dynamic_step_axis(self, dt, axis_idx):
        """
        H2' Sub-step: Evolve Momentum (w) about one principal axis.
        Orientation (DCM) is CONSTANT during this step.
        """
        i = axis_idx
        j = (i + 1) % 3
        k = (i + 2) % 3
        
        # Exact rotation of the angular momentum vector L
        # L_j, L_k rotate with rate omega_i = L_i / I_i = w[i]
        # But we track w directly.
        # w_j * I_j  --> new L_j
        
        # Current L components
        Li = self.I[i] * self.w[i]
        Lj = self.I[j] * self.w[j]
        Lk = self.I[k] * self.w[k]
        
        # Constant rate for this step
        rate = self.w[i] 
        
        # Rotate Lj, Lk
        angle = rate * dt
        c = np.cos(angle)
        s = np.sin(angle)
        
        # The flow preserves kinetic energy and L^2.
        # Equations:
        # dot_Lj = (Ik - Ii)/(Ij) * wk * wi ? No, using the splitting exact solution.
        # The splitting Hamiltonian H = 0.5 * Li^2 / Ii generates rotation of (Lj, Lk).
        # Specifically: dot_Lj = {Lj, H} = Lk * (Li/Ii) = Lk * wi
        #               dot_Lk = {Lk, H} = -Lj * (Li/Ii) = -Lj * wi
        # This is a rotation of (Lj, Lk) by angle (wi * dt).
        
        Lj_new = Lj * c + Lk * s
        Lk_new = -Lj * s + Lk * c
        
        # Update w
        self.w[j] = Lj_new / self.I[j]
        self.w[k] = Lk_new / self.I[k]
        # w[i] remains constant

# ============================================================================
# 3. Integrator
# ============================================================================

class YoshidaIntegrator:
    def __init__(self):
        self.w1 = 1.0 / (2.0 - 2.0**(1.0/5.0))
        self.w0 = 1.0 - 2.0 * self.w1

    def step(self, dt, crtbp_system, crtbp_state, rb_system):
        # Stage 1
        crtbp_state = self.s2_step_crtbp(self.w1 * dt, crtbp_system, crtbp_state)
        self.s2_step_rb(self.w1 * dt, rb_system) 
        
        # Stage 2
        crtbp_state = self.s2_step_crtbp(self.w0 * dt, crtbp_system, crtbp_state)
        self.s2_step_rb(self.w0 * dt, rb_system)
        
        # Stage 3
        crtbp_state = self.s2_step_crtbp(self.w1 * dt, crtbp_system, crtbp_state)
        self.s2_step_rb(self.w1 * dt, rb_system)
        
        return crtbp_state

    def s2_step_crtbp(self, dt, sys, state):
        q, p = state
        q, p = sys.rot(q, p, 0.5 * dt)
        q, p = sys.kick(q, p, 0.5 * dt)
        q, p = sys.drift(q, p, dt)
        q, p = sys.kick(q, p, 0.5 * dt)
        q, p = sys.rot(q, p, 0.5 * dt)
        return q, p

    def s2_step_rb(self, dt, sys):
        """
        Symmetric 2nd order step for Rigid Body (Lecture 24).
        Sequence: Kinematic(d/2) -> Dynamic(d) -> Kinematic(d/2)
        Dynamic(d) is further split: 1(d/2) 2(d/2) 3(d) 2(d/2) 1(d/2)
        """
        dt_sec = dt * SystemConstants.TU
        
        # 1. Half Kinematic (Attitude)
        sys.kinematic_step(0.5 * dt_sec)
        
        # 2. Full Dynamic (Momentum) - Split symmetrically
        sys.dynamic_step_axis(0.5 * dt_sec, 0)
        sys.dynamic_step_axis(0.5 * dt_sec, 1)
        sys.dynamic_step_axis(dt_sec, 2)
        sys.dynamic_step_axis(0.5 * dt_sec, 1)
        sys.dynamic_step_axis(0.5 * dt_sec, 0)
        
        # 3. Half Kinematic (Attitude)
        sys.kinematic_step(0.5 * dt_sec)

# ============================================================================
# 4. Main
# ============================================================================

def main():
    duration_years = 25.0
    dt_days = 0.01 
    dt_norm = dt_days * 86400.0 / SystemConstants.TU
    total_steps = int((duration_years * 365.25 * 86400.0) / (dt_norm * SystemConstants.TU))
    
    print(f"HW5 Simulation (Lecture 24 Splitting) | Duration: {duration_years} years")
    
    crtbp = CRTBP_Normalized(SystemConstants.mu)
    
    # L4 Initial State
    l4_ξ = 0.5 - SystemConstants.mu
    l4_η = np.sqrt(3.0) / 2.0
    q0 = np.array([l4_ξ, l4_η])
    p0 = np.array([-l4_η, l4_ξ])
    state_crtbp = (q0, p0)
    E0_crtbp = crtbp.get_energy(q0, p0)
    
    I_tens = np.array([SystemConstants.Ix, SystemConstants.Iy, SystemConstants.Iz])
    rb = RigidBody_Split(I_tens, SystemConstants.w0_body.copy())
    
    L0_inertial = rb.get_angular_momentum_inertial()
    E0_rb = rb.get_energy()
    
    # Storage
    store_interval = max(1, total_steps // 5000)
    t_hist, traj_ξ, traj_η, dE_crtbp_hist = [], [], [], []
    rb_L_inertial_hist, rb_L_diff_hist, rb_dE_hist = [], [], []
    
    integrator = YoshidaIntegrator()
    
    print("Starting integration...")
    for i in range(total_steps):
        state_crtbp = integrator.step(dt_norm, crtbp, state_crtbp, rb)
        
        if i % store_interval == 0:
            t_yr = SystemConstants.to_years((i+1) * dt_norm)
            q, p = state_crtbp
            
            dE = abs(crtbp.get_energy(q, p) - E0_crtbp)
            L_curr = rb.get_angular_momentum_inertial()
            dL_vec = np.abs(L_curr - L0_inertial)
            dE_rb = rb.get_energy() - E0_rb
            
            t_hist.append(t_yr)
            traj_ξ.append(q[0])
            traj_η.append(q[1])
            dE_crtbp_hist.append(dE)
            rb_L_inertial_hist.append(L_curr)
            rb_L_diff_hist.append(dL_vec)
            rb_dE_hist.append(dE_rb)
            
        if (i+1) % (total_steps // 10) == 0:
            print(f"Progress: {(i+1)/total_steps*100:.0f}%")

    # Plotting
    t_arr = np.array(t_hist)
    dE_arr = np.array(dE_crtbp_hist)
    dL_arr = np.array(rb_L_diff_hist)
    
    print("\nGenerating plots...")
    
    # 1. CRTBP Trajectory
    plt.figure()
    plt.plot(traj_ξ, traj_η, 'b-', linewidth=0.5)
    plt.plot(l4_ξ, l4_η, 'rx', label='L4')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title('CRTBP Trajectory')
    plt.xlabel('ξ')
    plt.ylabel('η')
    plt.legend()
    plt.savefig('hw5_crtbp_traj.png', dpi=150)
    plt.close()

    # 2. CRTBP Energy
    plt.figure()
    plt.semilogy(t_arr, dE_arr, 'r-', linewidth=0.5)
    plt.axhline(1e-12, color='k', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.title('CRTBP Energy Error')
    plt.xlabel('Years')
    plt.ylabel('|dE|')
    plt.savefig('hw5_crtbp_energy.png', dpi=150)
    plt.close()

    # 3. TFRBP L
    L_inertial_arr = np.array(rb_L_inertial_hist)
    plt.figure()
    plt.plot(t_arr, L_inertial_arr)
    plt.grid(True, alpha=0.3)
    plt.title('TFBRP Inertial Angular Momentum')
    plt.xlabel('Years')
    plt.ylabel('L')
    plt.savefig('hw5_tfbrp_L.png', dpi=150)
    plt.close()

    # 4. TFBRP Energy
    plt.figure()
    plt.plot(t_arr, rb_dE_hist)
    plt.grid(True, alpha=0.3)
    plt.title('TFBRP Energy Error')
    plt.xlabel('Years')
    plt.ylabel('dE')
    plt.savefig('hw5_tfbrp_E.png', dpi=150)
    plt.close()

    # 5. TFBRP dL
    plt.figure()
    plt.plot(t_arr, dL_arr)
    plt.axhline(1e-8, color='k', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.title('TFBRP dL (Inertial)')
    plt.xlabel('Years')
    plt.ylabel('|dL|')
    plt.savefig('hw5_tfbrp_dL.png', dpi=150)
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()