"""
HW5: Circular Restricted Three-Body Problem (CRTBP) Simulation
Using Yoshida 6th Order Symplectic Integrator

Hamiltonian:
    H = H1 + H2 + H3
    
    H1 = 0.5*px^2 + 0.5*py^2                  (kinetic energy)
    H2 = Omega*(px*qy - py*qx)                 (Coriolis, solved analytically)
    H3 = -mu1/r1 - mu2/r2                      (gravitational potential)
    
where:
    - qx, qy are positions
    - px, py are momenta
    - Omega is the angular velocity of the rotating frame (= 1 in normalized units)
    - mu1 = 1 - mu, mu2 = mu (mass parameters)
    - r1, r2 are distances to the two primaries
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import argparse
import sys
from typing import Optional


# ============================================================================
# Physical Constants
# ============================================================================

class PhysicalConstants:
    """Physical constants for Earth-Moon system."""
    
    # Gravitational parameters (km^3/s^2)
    mu_earth = 398600.4418
    mu_moon = 4902.8000
    
    # Earth-Moon distance (km)
    distance_earth_moon = 384400.0
    
    # Mass parameter
    mu = mu_moon / (mu_earth + mu_moon)
    mu1 = 1.0 - mu  # Primary mass
    mu2 = mu         # Secondary mass
    
    # Mean motion (rad/s)
    n = np.sqrt((mu_earth + mu_moon) / (distance_earth_moon ** 3))
    
    # Characteristic scales
    L_star = distance_earth_moon
    T_star = 1.0 / n
    V_star = L_star / T_star
    
    # Time conversions
    seconds_per_day = 86400.0
    days_per_year = 365.25
    
    # Angular velocity in rotating frame (normalized = 1)
    Omega = 1.0
    
    @classmethod
    def time_to_normalized(cls, time_days: float) -> float:
        """Convert time in days to normalized time."""
        time_seconds = time_days * cls.seconds_per_day
        return time_seconds / cls.T_star
    
    @classmethod
    def normalized_to_time(cls, time_normalized: float) -> float:
        """Convert normalized time to days."""
        time_seconds = time_normalized * cls.T_star
        return time_seconds / cls.seconds_per_day


# ============================================================================
# Yoshida Coefficients
# ============================================================================

def get_yoshida6_coefficients():
    """
    Yoshida 6th order symplectic integrator coefficients.
    
    Analytically derived using recursive composition:
    - Start with 2nd-order leapfrog: S_2
    - Build 4th-order via triple composition: S_4 = S_2(w_1*τ) ∘ S_2(w_0*τ) ∘ S_2(w_1*τ)
    - Build 6th-order via triple composition: S_6 = S_4(w_1*τ) ∘ S_4(w_0*τ) ∘ S_4(w_1*τ)
    
    Composition weights:
        w_1 = 1 / (2 - 2^(1/3)) ≈ 1.351207
        w_0 = -2^(1/3) / (2 - 2^(1/3)) ≈ -1.702414
    
    These satisfy: 2*w_1 + w_0 = 1
    
    Reference: H. Yoshida, "Construction of higher order symplectic integrators",
               Physics Letters A, 150(5-7), 262-268 (1990)
    
    Returns:
        c: drift coefficients for position updates (12 stages)
        d: kick coefficients for momentum updates (12 stages)
    """
    # Composition weights (analytical)
    cbrt_2 = 2.0**(1.0/3.0)  # 2^(1/3)
    w_1 = 1.0 / (2.0 - cbrt_2)
    w_0 = -cbrt_2 / (2.0 - cbrt_2)
    
    # Build S_4 coefficients (from 4th-order method)
    c4 = np.array([w_1/2.0, (w_1+w_0)/2.0, (w_0+w_1)/2.0, w_1/2.0])
    d4 = np.array([w_1, w_0, w_1, 0.0])
    
    # Build S_6 by composing S_4(w_1*τ) ∘ S_4(w_0*τ) ∘ S_4(w_1*τ)
    # This gives us 3*4 = 12 stages
    c = np.concatenate([
        c4 * w_1,  # S_4(w_1*τ)
        c4 * w_0,  # S_4(w_0*τ)
        c4 * w_1   # S_4(w_1*τ)
    ])
    
    d = np.concatenate([
        d4 * w_1,  # S_4(w_1*τ)
        d4 * w_0,  # S_4(w_0*τ)
        d4 * w_1   # S_4(w_1*τ)
    ])
    
    # Verify the sums
    sum_c = np.sum(c)
    sum_d = np.sum(d)
    
    if abs(sum_c - 1.0) > 1e-10 or abs(sum_d - 1.0) > 1e-10:
        print(f"WARNING: Yoshida coefficients may be incorrect!")
        print(f"  sum(c) = {sum_c:.15f} (should be 1.0)")
        print(f"  sum(d) = {sum_d:.15f} (should be 1.0)")
        print(f"  w_1 = {w_1:.15f}, w_0 = {w_0:.15f}")
        print(f"  2*w_1 + w_0 = {2*w_1 + w_0:.15f}")
    
    return c, d


# ============================================================================
# Hamiltonian Components
# ============================================================================

class CRTBP:
    """Circular Restricted Three-Body Problem."""
    
    def __init__(self, mu: float, Omega: float = 1.0):
        """
        Initialize CRTBP.
        
        Args:
            mu: mass parameter (mu2/(mu1+mu2))
            Omega: angular velocity of rotating frame (default 1.0 in normalized units)
        """
        self.mu = mu
        self.mu1 = 1.0 - mu
        self.mu2 = mu
        self.Omega = Omega
    
    # ------------------------------------------------------------------------
    # H1: Kinetic Energy
    # ------------------------------------------------------------------------
    
    def H1(self, px: float, py: float) -> float:
        """H1 = 0.5*px^2 + 0.5*py^2"""
        return 0.5 * (px**2 + py**2)
    
    def dH1_dpx(self, px: float) -> float:
        """dH1/dpx = px"""
        return px
    
    def dH1_dpy(self, py: float) -> float:
        """dH1/dpy = py"""
        return py
    
    # ------------------------------------------------------------------------
    # H2: Coriolis (Analytical Solution)
    # ------------------------------------------------------------------------
    
    def H2(self, qx: float, qy: float, px: float, py: float) -> float:
        """H2 = Omega*(px*qy - py*qx)"""
        return self.Omega * (px * qy - py * qx)
    
    def analytical_flow_H2(self, qx: float, qy: float, px: float, py: float, 
                           dt: float) -> tuple:
        """
        Analytical solution of H2 flow.
        
        Equations:
            dqx/dt = Omega * qy
            dqy/dt = -Omega * qx
            dpx/dt = Omega * py
            dpy/dt = -Omega * px
            
        Solution (rotation by angle Omega*dt):
            qx_new = qx*cos(Omega*dt) + qy*sin(Omega*dt)
            qy_new = -qx*sin(Omega*dt) + qy*cos(Omega*dt)
            px_new = px*cos(Omega*dt) + py*sin(Omega*dt)
            py_new = -px*sin(Omega*dt) + py*cos(Omega*dt)
            
        Args:
            qx, qy: positions
            px, py: momenta
            dt: time step
            
        Returns:
            (qx_new, qy_new, px_new, py_new)
        """
        theta = self.Omega * dt
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        qx_new = qx * cos_t + qy * sin_t
        qy_new = -qx * sin_t + qy * cos_t
        px_new = px * cos_t + py * sin_t
        py_new = -px * sin_t + py * cos_t
        
        return qx_new, qy_new, px_new, py_new
    
    # ------------------------------------------------------------------------
    # H3: Gravitational Potential
    # ------------------------------------------------------------------------
    
    def H3(self, qx: float, qy: float) -> float:
        """H3 = -mu1/r1 - mu2/r2"""
        r1, r2 = self._compute_distances(qx, qy)
        return -self.mu1 / r1 - self.mu2 / r2
    
    def dH3_dqx(self, qx: float, qy: float) -> float:
        """dH3/dqx = mu1*(qx + mu)/r1^3 + mu2*(qx - mu1)/r2^3"""
        r1, r2 = self._compute_distances(qx, qy)
        return self.mu1 * (qx + self.mu) / r1**3 + self.mu2 * (qx - self.mu1) / r2**3
    
    def dH3_dqy(self, qx: float, qy: float) -> float:
        """dH3/dqy = mu1*qy/r1^3 + mu2*qy/r2^3"""
        r1, r2 = self._compute_distances(qx, qy)
        return qy * (self.mu1 / r1**3 + self.mu2 / r2**3)
    
    def _compute_distances(self, qx: float, qy: float) -> tuple:
        """
        Compute distances to primaries.
        
        Primary 1 (larger) at (-mu, 0)
        Primary 2 (smaller) at (1-mu, 0)
        """
        r1 = np.sqrt((qx + self.mu)**2 + qy**2)
        r2 = np.sqrt((qx - self.mu1)**2 + qy**2)
        return r1, r2
    
    # ------------------------------------------------------------------------
    # Total Energy
    # ------------------------------------------------------------------------
    
    def total_energy(self, qx: float, qy: float, px: float, py: float) -> float:
        """Compute total Hamiltonian H = H1 + H2 + H3"""
        return self.H1(px, py) + self.H2(qx, qy, px, py) + self.H3(qx, qy)

    def angular_momentum_z(self, qx: float, qy: float, px: float, py: float) -> float:
        """Planar angular momentum (z-component) Lz = qx*py - qy*px."""
        return qx * py - qy * px


# ============================================================================
# Torque-Free Rigid Body (3D) Dynamics (decoupled system)
# ============================================================================

class RigidBody:
    """Torque-free rigid body using Euler's equations and quaternion orientation.

    State variables:
        w1, w2, w3 : angular velocity components in body frame
        q0, q1, q2, q3 : orientation quaternion (body->inertial)

    Equations (Euler's torque-free):
        I1 * dw1/dt = (I2 - I3) * w2 * w3
        I2 * dw2/dt = (I3 - I1) * w3 * w1
        I3 * dw3/dt = (I1 - I2) * w1 * w2

    Quaternion update:
        dq/dt = 0.5 * Omega(w) * q,  Omega(w) = [0, w1, w2, w3]

    Integrated with RK4 (not strictly symplectic but acceptable for small dt
    given no coupling to translational CRTBP dynamics in torque-free case).
    """

    def __init__(self, I1: float, I2: float, I3: float,
                 w1: float, w2: float, w3: float):
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        # Unit quaternion initial orientation
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0

    def energy(self) -> float:
        return 0.5 * (self.I1 * self.w1**2 + self.I2 * self.w2**2 + self.I3 * self.w3**2)

    def angular_momentum_body(self):
        return (self.I1 * self.w1, self.I2 * self.w2, self.I3 * self.w3)

    def _euler_rhs(self, w1, w2, w3):
        dw1 = (self.I2 - self.I3) * w2 * w3 / self.I1
        dw2 = (self.I3 - self.I1) * w3 * w1 / self.I2
        dw3 = (self.I1 - self.I2) * w1 * w2 / self.I3
        return dw1, dw2, dw3

    def _quat_rhs(self, q0, q1, q2, q3, w1, w2, w3):
        dq0 = -0.5 * (w1 * q1 + w2 * q2 + w3 * q3)
        dq1 = 0.5 * (w1 * q0 + w2 * q3 - w3 * q2)
        dq2 = 0.5 * (-w1 * q3 + w2 * q0 + w3 * q1)
        dq3 = 0.5 * (w1 * q2 - w2 * q1 + w3 * q0)
        return dq0, dq1, dq2, dq3

    def step(self, dt_seconds: float):
        """Advance rigid body by one RK4 step in physical seconds.

        Args:
            dt_seconds: timestep in seconds (Table 1 angular velocities are in s^-1)
        """
        w1_0, w2_0, w3_0 = self.w1, self.w2, self.w3
        q0_0, q1_0, q2_0, q3_0 = self.q0, self.q1, self.q2, self.q3

        k1_w = self._euler_rhs(w1_0, w2_0, w3_0)
        k2_w = self._euler_rhs(w1_0 + 0.5*dt_seconds*k1_w[0], w2_0 + 0.5*dt_seconds*k1_w[1], w3_0 + 0.5*dt_seconds*k1_w[2])
        k3_w = self._euler_rhs(w1_0 + 0.5*dt_seconds*k2_w[0], w2_0 + 0.5*dt_seconds*k2_w[1], w3_0 + 0.5*dt_seconds*k2_w[2])
        k4_w = self._euler_rhs(w1_0 + dt_seconds*k3_w[0], w2_0 + dt_seconds*k3_w[1], w3_0 + dt_seconds*k3_w[2])

        self.w1 += dt_seconds * (k1_w[0] + 2*k2_w[0] + 2*k3_w[0] + k4_w[0]) / 6.0
        self.w2 += dt_seconds * (k1_w[1] + 2*k2_w[1] + 2*k3_w[1] + k4_w[1]) / 6.0
        self.w3 += dt_seconds * (k1_w[2] + 2*k2_w[2] + 2*k3_w[2] + k4_w[2]) / 6.0

        k1_q = self._quat_rhs(q0_0, q1_0, q2_0, q3_0, w1_0, w2_0, w3_0)
        k2_q = self._quat_rhs(q0_0 + 0.5*dt_seconds*k1_q[0], q1_0 + 0.5*dt_seconds*k1_q[1], q2_0 + 0.5*dt_seconds*k1_q[2], q3_0 + 0.5*dt_seconds*k1_q[3],
                               self.w1, self.w2, self.w3)
        k3_q = self._quat_rhs(q0_0 + 0.5*dt_seconds*k2_q[0], q1_0 + 0.5*dt_seconds*k2_q[1], q2_0 + 0.5*dt_seconds*k2_q[2], q3_0 + 0.5*dt_seconds*k2_q[3],
                               self.w1, self.w2, self.w3)
        k4_q = self._quat_rhs(q0_0 + dt_seconds*k3_q[0], q1_0 + dt_seconds*k3_q[1], q2_0 + dt_seconds*k3_q[2], q3_0 + dt_seconds*k3_q[3],
                               self.w1, self.w2, self.w3)

        self.q0 += dt_seconds * (k1_q[0] + 2*k2_q[0] + 2*k3_q[0] + k4_q[0]) / 6.0
        self.q1 += dt_seconds * (k1_q[1] + 2*k2_q[1] + 2*k3_q[1] + k4_q[1]) / 6.0
        self.q2 += dt_seconds * (k1_q[2] + 2*k2_q[2] + 2*k3_q[2] + k4_q[2]) / 6.0
        self.q3 += dt_seconds * (k1_q[3] + 2*k2_q[3] + 2*k3_q[3] + k4_q[3]) / 6.0

        norm_q = np.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        if norm_q != 0.0:
            self.q0 /= norm_q
            self.q1 /= norm_q
            self.q2 /= norm_q
            self.q3 /= norm_q


def create_rigid_body(rb_I1: Optional[float] = None, rb_I2: Optional[float] = None, rb_I3: Optional[float] = None,
                      rb_w1: Optional[float] = None, rb_w2: Optional[float] = None, rb_w3: Optional[float] = None) -> tuple:
    """Instantiate a rigid body using Table 1 defaults unless overrides are provided."""
    # Table 1 baseline ellipsoid and density (used if overrides are None)
    a = 10.0  # m
    b = 5.0   # m
    c = 4.0   # m
    density = 1000.0  # kg/m^3
    mass = (4.0/3.0) * np.pi * a * b * c * density
    I_x = 0.2 * mass * (b**2 + c**2)
    I_y = 0.2 * mass * (a**2 + c**2)
    I_z = 0.2 * mass * (a**2 + b**2)
    w1_base = 1e-7
    w2_base = 1e-8
    w3_base = 1e-5

    I1_use = rb_I1 if rb_I1 is not None else I_x
    I2_use = rb_I2 if rb_I2 is not None else I_y
    I3_use = rb_I3 if rb_I3 is not None else I_z
    w1_use = rb_w1 if rb_w1 is not None else w1_base
    w2_use = rb_w2 if rb_w2 is not None else w2_base
    w3_use = rb_w3 if rb_w3 is not None else w3_base

    rb = RigidBody(I1_use, I2_use, I3_use, w1_use, w2_use, w3_use)
    rb_energy0 = rb.energy()
    rb_L0 = rb.angular_momentum_body()
    return rb, rb_energy0, rb_L0


# ============================================================================
# Yoshida 6th Order Integrator
# ============================================================================

class Yoshida6Integrator:
    """
    Yoshida 6th order symplectic integrator for CRTBP.
    
    Splits H = H1 + H2 + H3 where:
    - H1 + H3 are integrated symplectically (drift-kick)
    - H2 is solved analytically at each stage
    """
    
    def __init__(self, crtbp: CRTBP, dt_days: float):
        """
        Initialize integrator.
        
        Args:
            crtbp: CRTBP instance
            dt_days: time step in days
        """
        self.crtbp = crtbp
        self.dt_days = dt_days
        self.dt_normalized = PhysicalConstants.time_to_normalized(dt_days)
        
        # Get Yoshida-6 coefficients
        self.c_coeffs, self.d_coeffs = get_yoshida6_coefficients()
    
    def step(self, state: dict) -> dict:
        """
        Advance state by one timestep using Yoshida-6.
        
        For each stage i with coefficients (c_i, d_i):
            Symmetric sequence to reduce secular drift:
            1. Apply half H2 analytical flow by (c_i*dt)/2
            2. Drift H1 by c_i*dt: qx += c_i*dt*px, qy += c_i*dt*py
            3. Apply half H2 analytical flow by (c_i*dt)/2
            4. Kick H3 by d_i*dt: px -= d_i*dt*dH3/dqx, py -= d_i*dt*dH3/dqy
            
        Args:
            state: dict with keys 'qx', 'qy', 'px', 'py'
            
        Returns:
            new_state: updated state dict
        """
        qx = state['qx']
        qy = state['qy']
        px = state['px']
        py = state['py']
        
        # Apply Yoshida-6 composition
        for c_i, d_i in zip(self.c_coeffs, self.d_coeffs):
            # Symmetric H2 around H1 drift
            if c_i != 0.0:
                dt_drift = c_i * self.dt_normalized
                half = 0.5 * dt_drift
                # Half H2 flow
                qx, qy, px, py = self.crtbp.analytical_flow_H2(qx, qy, px, py, half)
                # Drift H1
                qx, qy = self._drift_H1(qx, qy, px, py, dt_drift)
                # Half H2 flow
                qx, qy, px, py = self.crtbp.analytical_flow_H2(qx, qy, px, py, half)

            # Kick H3 by d_i * dt
            if d_i != 0.0:
                dt_kick = d_i * self.dt_normalized
                px, py = self._kick_H3(qx, qy, px, py, dt_kick)
        
        return {'qx': qx, 'qy': qy, 'px': px, 'py': py}
    
    def _drift_H1(self, qx: float, qy: float, px: float, py: float, 
                  dt: float) -> tuple:
        """
        Drift step for H1: dq/dt = dH1/dp = p
        
        qx_new = qx + dt * px
        qy_new = qy + dt * py
        """
        qx_new = qx + dt * self.crtbp.dH1_dpx(px)
        qy_new = qy + dt * self.crtbp.dH1_dpy(py)
        return qx_new, qy_new
    
    def _kick_H3(self, qx: float, qy: float, px: float, py: float, 
                 dt: float) -> tuple:
        """
        Kick step for H3: dp/dt = -dH3/dq
        
        px_new = px - dt * dH3/dqx
        py_new = py - dt * dH3/dqy
        """
        px_new = px - dt * self.crtbp.dH3_dqx(qx, qy)
        py_new = py - dt * self.crtbp.dH3_dqy(qx, qy)
        return px_new, py_new


# ============================================================================
# Simulation
# ============================================================================

def initialize_state():
    """Return initial state for halo orbit around L1 (original configuration)."""
    qx0 = 1.2
    qy0 = 0.0
    px0 = 0.0
    py0 = -1.04935750983031990726  # Halo orbit initial momentum (unchanged)
    return {'qx': qx0, 'qy': qy0, 'px': px0, 'py': py0}


def run_simulation(duration_years: float, dt_days: float,
                   verbose: bool = True,
                   rigid_body: bool = False,
                   rb_I1: Optional[float] = None, rb_I2: Optional[float] = None, rb_I3: Optional[float] = None,
                   rb_w1: Optional[float] = None, rb_w2: Optional[float] = None, rb_w3: Optional[float] = None) -> dict:
    """
    Run CRTBP simulation.
    
    Args:
        duration_years: simulation duration in years
        dt_days: time step in days
        verbose: print progress
        
    Returns:
        results dict with trajectory and diagnostics
    """
    # Initialize
    crtbp = CRTBP(PhysicalConstants.mu, PhysicalConstants.Omega)
    integrator = Yoshida6Integrator(crtbp, dt_days)
    state = initialize_state()
    
    # Initial energy
    E0 = crtbp.total_energy(state['qx'], state['qy'], state['px'], state['py'])
    Lz0 = crtbp.angular_momentum_z(state['qx'], state['qy'], state['px'], state['py'])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CRTBP Simulation with Yoshida-6 Integrator")
        print(f"{'='*70}")
        print(f"Parameters:")
        print(f"  Duration: {duration_years:.2f} years")
        print(f"  Time step: {dt_days:.6f} days")
        print(f"  Mass parameter mu: {PhysicalConstants.mu:.10f}")
        print(f"\nInitial Conditions:")
        print(f"  Position: qx={state['qx']:.6f}, qy={state['qy']:.6f}")
        print(f"  Momenta:  px={state['px']:.6f}, py={state['py']:.6f}")
        print(f"  Initial energy: E0 = {E0:.15f}")
        print(f"{'='*70}\n")
    
    # Storage
    n_steps = int(duration_years * PhysicalConstants.days_per_year / dt_days)
    store_interval = max(1, n_steps // 10000)
    
    times = [0.0]
    trajectory = [(state['qx'], state['qy'])]
    energies = [E0]
    energy_abs_diff = [0.0]
    angmom_z = [Lz0]
    angmom_abs_diff = [0.0]

    rb = None
    rb_energy = []
    rb_energy0 = None
    rb_L_body = []
    rb_L0 = None
    if rigid_body:
        rb, rb_energy0, rb_L0 = create_rigid_body(
            rb_I1=rb_I1, rb_I2=rb_I2, rb_I3=rb_I3,
            rb_w1=rb_w1, rb_w2=rb_w2, rb_w3=rb_w3
        )
    
    # Integration loop
    rb_dt_seconds = dt_days * PhysicalConstants.seconds_per_day

    for step in range(n_steps):
        state = integrator.step(state)
        if rb is not None:
            rb.step(rb_dt_seconds)
        
        if step % store_interval == 0:
            t = (step + 1) * dt_days / PhysicalConstants.days_per_year
            times.append(t)
            trajectory.append((state['qx'], state['qy']))
            
            E = crtbp.total_energy(state['qx'], state['qy'], state['px'], state['py'])
            energies.append(E)
            energy_abs_diff.append(abs(E - E0))
            Lz = crtbp.angular_momentum_z(state['qx'], state['qy'], state['px'], state['py'])
            angmom_z.append(Lz)
            angmom_abs_diff.append(abs(Lz - Lz0))
            if rb is not None:
                rb_energy.append(rb.energy())
                rb_L_body.append(rb.angular_momentum_body())
        
        # Progress updates
        if verbose and (step + 1) % max(1, n_steps // 20) == 0:
            progress = 100.0 * (step + 1) / n_steps
            t = (step + 1) * dt_days / PhysicalConstants.days_per_year
            E = crtbp.total_energy(state['qx'], state['qy'], state['px'], state['py'])
            dE = abs(E - E0)
            print(f"  Progress: {progress:5.1f}% | t = {t:6.2f} yr | |ΔE| = {dE:.3e}")
    
    # Final diagnostics
    E_final = crtbp.total_energy(state['qx'], state['qy'], state['px'], state['py'])
    dE_abs = abs(E_final - E0)
    dE_rel = dE_abs / abs(E0)
    Lz_final = angmom_z[-1]
    dLz_abs = abs(Lz_final - Lz0)
    
    # Compute maximum absolute energy error over entire trajectory
    energy_errors_abs = np.abs(np.array(energies) - E0)
    max_dE = np.max(energy_errors_abs)
    max_dE_rel = max_dE / abs(E0)
    max_dLz_abs = max(angmom_abs_diff)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Simulation Complete!")
        print(f"\nFinal State:")
        print(f"  Time: {duration_years:.2f} years")
        print(f"  Position: qx={state['qx']:.6f}, qy={state['qy']:.6f}")
        print(f"  Momenta:  px={state['px']:.6f}, py={state['py']:.6f}")
        print(f"\nConservation:")
        print(f"  |ΔE| final = {dE_abs:.6e} (target < 1e-12?) {'PASS' if dE_abs < 1e-12 else 'FAIL'}")
        print(f"  |ΔE| max   = {max_dE:.6e}")
        print(f"  |ΔLz| final = {dLz_abs:.6e} (target < 1e-8?) {'PASS' if dLz_abs < 1e-8 else 'FAIL'}")
        print(f"  |ΔLz| max   = {max_dLz_abs:.6e}")
        if rb is not None:
            if rb_energy0 is not None and rb_energy:
                rb_E_final = rb_energy[-1]
                rb_dE_abs = abs(rb_E_final - rb_energy0)
                print("  RB |ΔE| final = {:.6e}".format(rb_dE_abs))
            if rb_L0 is not None and rb_L_body:
                rb_L_final = rb_L_body[-1]
                rb_L_diff = tuple(abs(rb_L_final[i] - rb_L0[i]) for i in range(3))
                print("  RB |ΔL_body| components = ({:.3e}, {:.3e}, {:.3e})".format(*rb_L_diff))
        print(f"{'='*70}\n")
    
    results = {
        'times': np.array(times),
        'trajectory': np.array(trajectory),
        'energies': np.array(energies),
        'energy_abs_diff': np.array(energy_abs_diff),
        'angular_momentum_z': np.array(angmom_z),
        'angmom_abs_diff': np.array(angmom_abs_diff),
        'E0': E0,
        'Lz0': Lz0,
        'dE_abs': dE_abs,
        'dE_rel': dE_rel,
        'max_dE_rel': max_dE_rel,
        'dLz_abs': dLz_abs,
        'max_dLz_abs': max_dLz_abs,
        'final_state': state,
        'rigid_body_enabled': rb is not None
    }
    if rb is not None:
        results.update({
            'rb_energy': np.array(rb_energy),
            'rb_energy0': rb_energy0,
            'rb_L_body': np.array(rb_L_body),
            'rb_L0': np.array(rb_L0)
        })
    return results


def generate_plots(results: dict, output_file: str = "hw5_results.png"):
    """Generate requested plots: CRTBP trajectory, CRTBP |ΔE| semilogy,
    TFBRP angular momentum components, TFBRP differential energy, TFBRP differential angular momentum."""
    times = results['times']
    trajectory = results['trajectory']
    energies = results['energies']
    E0 = results['E0']
    angmom_abs_diff = results.get('angmom_abs_diff', None)
    rb_enabled = results.get('rigid_body_enabled', False)
    rb_energy = results.get('rb_energy') if rb_enabled else None
    rb_energy0 = results.get('rb_energy0') if rb_enabled else None
    rb_L_body = results.get('rb_L_body') if rb_enabled else None
    rb_L0 = results.get('rb_L0') if rb_enabled else None
    
    # Create figure with 3x2 layout (top: trajectory & CRTBP |ΔE|, middle: TFBRP L components & TFBRP ΔE, bottom: TFBRP ΔL)
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)
    
    # Trajectory (zoomed to spacecraft)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('qx (normalized)')
    ax.set_ylabel('qy (normalized)')
    ax.set_title('Spacecraft Trajectory (Rotating Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # CRTBP differential energy |ΔE| in semilogy
    ax = fig.add_subplot(gs[0, 1])
    energy_abs_error = np.abs(energies - E0)
    ax.semilogy(times, energy_abs_error, 'b-', linewidth=1)
    ax.axhline(1e-12, color='orange', linestyle=':', alpha=0.5, label='Target |ΔE| 1e-12')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('|ΔE| (CRTBP)')
    ax.set_title('CRTBP Energy Drift |ΔE| (semilogy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TFBRP: body angular momentum components over time
    ax = fig.add_subplot(gs[1, 0])
    if rb_enabled and rb_L_body is not None and len(rb_L_body) > 0:
        rb_L_body_arr = np.array(rb_L_body)
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 0], label='Lx (TFBRP)')
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 1], label='Ly (TFBRP)')
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 2], label='Lz (TFBRP)')
        ax.legend(fontsize=8)
        ax.set_title('TFBRP Body Angular Momentum Components')
        ax.set_ylabel('Angular Momentum (kg·m²/s)')
    else:
        ax.text(0.5, 0.5, 'Rigid body disabled or unavailable', ha='center', va='center')
        ax.set_title('TFBRP Body Angular Momentum Components')
    ax.set_xlabel('Time (years)')
    ax.grid(True, alpha=0.3)

    # TFBRP: differential energy over time
    ax = fig.add_subplot(gs[1, 1])
    if rb_enabled and rb_energy is not None and rb_energy0 is not None:
        rb_dE = np.array(rb_energy) - rb_energy0
        ax.plot(times[:len(rb_dE)], rb_dE, 'm-', linewidth=1, label='ΔE_rb')
        ax.set_title('TFBRP Differential Energy ΔE')
        ax.set_ylabel('ΔE (J)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Rigid body disabled or unavailable', ha='center', va='center')
        ax.set_title('TFBRP Differential Energy ΔE')
    ax.set_xlabel('Time (years)')
    ax.grid(True, alpha=0.3)
    # TFBRP: differential angular momentum over time
    ax = fig.add_subplot(gs[2, :])
    if rb_enabled and rb_L_body is not None and len(rb_L_body) > 0:
        rb_L_body_arr = np.array(rb_L_body)
        L0 = rb_L_body_arr[0]
        dL = rb_L_body_arr - L0
        ax.plot(times[:len(dL)], dL[:, 0], label='ΔLx (TFBRP)')
        ax.plot(times[:len(dL)], dL[:, 1], label='ΔLy (TFBRP)')
        ax.plot(times[:len(dL)], dL[:, 2], label='ΔLz (TFBRP)')
        ax.legend(fontsize=8)
        ax.set_title('TFBRP Differential Angular Momentum ΔL')
        ax.set_ylabel('ΔL (kg·m²/s)')
    else:
        ax.text(0.5, 0.5, 'Rigid body disabled or unavailable', ha='center', va='center')
        ax.set_title('TFBRP Differential Angular Momentum ΔL')
    ax.set_xlabel('Time (years)')
    ax.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close()


def generate_individual_plots(results: dict, prefix: str = "hw5"):
    """Generate individual plots matching requested outputs."""
    times = results['times']
    trajectory = results['trajectory']
    energies = results['energies']
    E0 = results['E0']
    angmom_abs_diff = results.get('angmom_abs_diff', None)
    rb_enabled = results.get('rigid_body_enabled', False)
    rb_energy = results.get('rb_energy') if rb_enabled else None
    rb_energy0 = results.get('rb_energy0') if rb_enabled else None
    rb_L_body = results.get('rb_L_body') if rb_enabled else None
    rb_L0 = results.get('rb_L0') if rb_enabled else None
    mu = PhysicalConstants.mu

    # 1. Trajectory (zoomed to spacecraft)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('qx (normalized)')
    ax.set_ylabel('qy (normalized)')
    ax.set_title('Spacecraft Trajectory (Rotating Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.savefig(f"{prefix}_trajectory.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_trajectory.png")
    plt.close()

    # 1a. 3D Trajectory with Earth and Moon (z=0 plane)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = np.zeros_like(x)
    ax3d.plot3D(x, y, z, 'b-', linewidth=0.6, label='Spacecraft')
    # Start/end markers
    ax3d.plot3D([x[0]], [y[0]], [0.0], 'go', label='Start')
    ax3d.plot3D([x[-1]], [y[-1]], [0.0], 'ro', label='End')
    # Earth and Moon positions in rotating frame
    earth_pos = (-mu, 0.0, 0.0)
    moon_pos = (1.0 - mu, 0.0, 0.0)
    ax3d.plot3D([earth_pos[0]], [earth_pos[1]], [earth_pos[2]], 'co', label='Earth')
    ax3d.plot3D([moon_pos[0]], [moon_pos[1]], [moon_pos[2]], 'mo', label='Moon')
    # Aesthetic bounds
    rng = max(np.ptp(x), np.ptp(y)) if len(x) > 1 else 1.0
    cx = np.mean(x); cy = np.mean(y)
    ax3d.set_xlim(cx - 0.6*rng, cx + 0.6*rng)
    ax3d.set_ylim(cy - 0.6*rng, cy + 0.6*rng)
    ax3d.set_zlim(-0.2*rng, 0.2*rng)
    ax3d.set_xlabel('qx (normalized)')
    ax3d.set_ylabel('qy (normalized)')
    ax3d.set_zlabel('qz (normalized)')
    ax3d.set_title('Spacecraft Trajectory in 3D with Earth and Moon')
    ax3d.legend(loc='upper right', fontsize=8)
    plt.savefig(f"{prefix}_trajectory_3d.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_trajectory_3d.png")
    plt.close()

    # 2) Time evolution of differential energy (CRTBP) in semilogy
    fig, ax = plt.subplots(figsize=(8, 6))
    energy_abs_error = np.abs(energies - E0)
    ax.semilogy(times, energy_abs_error, 'b-', linewidth=1)
    ax.axhline(1e-12, color='orange', linestyle=':', alpha=0.5, label='Target |ΔE| 1e-12')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('|ΔE| (CRTBP)')
    ax.set_title('CRTBP Energy Drift |ΔE| (semilogy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_energy_relative.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_energy_relative.png")
    plt.close()

    # 3) TFBRP: derived angular momentum components over time
    rb_L_body = results.get('rb_L_body') if rb_enabled else None
    fig, ax = plt.subplots(figsize=(8, 6))
    if rb_enabled and rb_L_body is not None and len(rb_L_body) > 0:
        rb_L_body_arr = np.array(rb_L_body)
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 0], label='Lx (TFBRP)')
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 1], label='Ly (TFBRP)')
        ax.plot(times[:len(rb_L_body_arr)], rb_L_body_arr[:, 2], label='Lz (TFBRP)')
        ax.legend()
        ax.set_title('TFBRP Body Angular Momentum Components')
    else:
        ax.text(0.5, 0.5, 'Rigid body angular momentum unavailable', ha='center', va='center')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Angular Momentum (kg·m²/s)')
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_rb_angular_momentum.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_rb_angular_momentum.png")
    plt.close()

    # 4) TFBRP: differential energy over time
    rb_energy = results.get('rb_energy') if rb_enabled else None
    rb_energy0 = results.get('rb_energy0') if rb_enabled else None
    fig, ax = plt.subplots(figsize=(8, 6))
    if rb_enabled and rb_energy is not None and rb_energy0 is not None:
        rb_dE = np.array(rb_energy) - rb_energy0
        ax.plot(times[:len(rb_dE)], rb_dE, 'm-', linewidth=1)
        ax.set_title('TFBRP Differential Energy ΔE')
    else:
        ax.text(0.5, 0.5, 'Rigid body energy unavailable', ha='center', va='center')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('ΔE (J)')
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_rb_energy_diff.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_rb_energy_diff.png")
    plt.close()

        # 5. Removed L4 visualization (reverted to restricted 3-body halo-only analysis)

    # 5) TFBRP: differential angular momentum over time
    fig, ax = plt.subplots(figsize=(8, 6))
    if rb_enabled and rb_L_body is not None and len(rb_L_body) > 0:
        rb_L_body_arr = np.array(rb_L_body)
        L0 = rb_L_body_arr[0]
        dL = rb_L_body_arr - L0
        ax.plot(times[:len(dL)], dL[:, 0], label='ΔLx (TFBRP)')
        ax.plot(times[:len(dL)], dL[:, 1], label='ΔLy (TFBRP)')
        ax.plot(times[:len(dL)], dL[:, 2], label='ΔLz (TFBRP)')
        ax.legend()
        ax.set_title('TFBRP Differential Angular Momentum ΔL')
    else:
        ax.text(0.5, 0.5, 'Rigid body angular momentum unavailable', ha='center', va='center')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('ΔL (kg·m²/s)')
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_rb_angular_momentum_diff.png", dpi=180, bbox_inches='tight')
    print(f"  Saved plot: {prefix}_rb_angular_momentum_diff.png")
    plt.close()

    # 7. Rigid body diagnostics (if enabled)
    if rb_enabled and rb_energy is not None and rb_energy0 is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        rb_dE = np.abs(np.array(rb_energy) - rb_energy0)
        ax.semilogy(times[:len(rb_dE)], rb_dE, 'c-', linewidth=1, label='|ΔE_rb|')
        if rb_L_body is not None and rb_L0 is not None and len(rb_L_body) > 0:
            rb_L_body_arr = np.array(rb_L_body)
            rb_L0_arr = np.array(rb_L0)
            dL1 = np.abs(rb_L_body_arr[:, 0] - rb_L0_arr[0])
            dL2 = np.abs(rb_L_body_arr[:, 1] - rb_L0_arr[1])
            dL3 = np.abs(rb_L_body_arr[:, 2] - rb_L0_arr[2])
            ax.semilogy(times[:len(dL1)], dL1, 'r--', linewidth=0.8, label='|ΔL1|')
            ax.semilogy(times[:len(dL2)], dL2, 'g--', linewidth=0.8, label='|ΔL2|')
            ax.semilogy(times[:len(dL3)], dL3, 'b--', linewidth=0.8, label='|ΔL3|')
        ax.set_title('Rigid Body Conservation')
        ax.set_ylabel('Absolute Differences')
        ax.set_xlabel('Time (years)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{prefix}_rigid_body.png", dpi=180, bbox_inches='tight')
        print(f"  Saved plot: {prefix}_rigid_body.png")
        plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='HW5: CRTBP simulation with Yoshida-6 integrator'
    )
    
    parser.add_argument(
        '--duration-years',
        type=float,
        default=1.0,
        help='Simulation duration in years (default: 1.0)'
    )
    
    parser.add_argument(
        '--dt-days',
        type=float,
        default=0.001,
        help='Time step in days (default: 0.001)'
    )

    parser.add_argument(
        '--rigid-body',
        action='store_true',
        help='Enable torque-free rigid body dynamics'
    )
    parser.add_argument('--I1', type=float, default=None, help='Override I1 (kg m^2); default = Table 1 ellipsoid')
    parser.add_argument('--I2', type=float, default=None, help='Override I2 (kg m^2); default = Table 1 ellipsoid')
    parser.add_argument('--I3', type=float, default=None, help='Override I3 (kg m^2); default = Table 1 ellipsoid')
    parser.add_argument('--w1', type=float, default=None, help='Override initial ω1 (rad/s); default=1e-7')
    parser.add_argument('--w2', type=float, default=None, help='Override initial ω2 (rad/s); default=1e-8')
    parser.add_argument('--w3', type=float, default=None, help='Override initial ω3 (rad/s); default=1e-5')
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run simulation
    results = run_simulation(
        duration_years=args.duration_years,
        dt_days=args.dt_days,
        verbose=not args.quiet,
        rigid_body=args.rigid_body,
        rb_I1=args.I1,
        rb_I2=args.I2,
        rb_I3=args.I3,
        rb_w1=args.w1,
        rb_w2=args.w2,
        rb_w3=args.w3,
        
    )
    
    # Generate plots
    if not args.skip_plots:
        print("\nGenerating plots...")
        generate_plots(results)
        print("\nGenerating individual plots...")
        generate_individual_plots(results)
    
    print("\nSimulation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
