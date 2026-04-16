#!/usr/bin/env python3
"""
Energy vs. Velocity: Why Parameterization Matters for Composition
=================================================================

Diffusion and flow-based models can be parameterized in two fundamentally
different ways:

  ENERGY MODELS  output a scalar energy V(x). The dynamics come from its
                 gradient: the "force" is -∇V(x). To compose two energy
                 models, just add the scalars: V_total = V₁ + V₂. The
                 gradient of a sum IS the sum of gradients — always exact.

  VELOCITY MODELS  output a vector field v(x) directly — the "force" is
                    just v(x). To compose them, you add vectors:
                    v_total = v₁ + v₂. But there's no guarantee this sum
                    represents the gradient of any physical energy landscape.

This script demonstrates the difference using two potentials with OPPOSING
ROTATIONAL DYNAMICS that produce a ring-shaped equilibrium:

    V₁(x,y) = -a·x·y + (r² - R²)²    — particles spiral CCW onto a ring
    V₂(x,y) = +a·x·y + (r² - R²)²    — particles spiral CW onto a ring

Their sum V₁+V₂ = 2(r² - R²)² has NO rotation — particles converge
radially onto the ring. The key insight: WHERE each particle lands on the
ring depends on whether it went straight (correct) or spiraled (wrong).

Energy composition gets the radial-only dynamics by construction — the
±a·xy terms cancel in the scalar sum before the gradient is ever computed.
Velocity composition must cancel the rotation through imperfect vector
addition. Any error creates residual spiraling that shifts where particles
land on the ring, producing a visibly different distribution.

RUNNING
-------
    $ uv run energy_vs_velocity.py     # generates energy_vs_velocity.png

READING GUIDE
-------------
    potential_v1 / v2     — Opposing spiral potentials (cancel when summed).
    generate_training_data — Run particles downhill to create training data.
    EnergyModel           — Scalar output; drift computed via autograd.
    VelocityModel         — Vector output; drift is the output directly.
    train_energy / vel    — Fit each model type to trajectory data.
    simulate              — Push particles forward with a model's drift.
    measure_curl          — Quantify the curl (rotation) in a vector field.
    visualize             — Compare ground truth vs. energy vs. velocity.
"""

import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: TWO ENERGY LANDSCAPES WITH OPPOSING ROTATIONS
# ============================================================================
# V₁ and V₂ each have a cross term (±a·xy) that creates rotation. When you
# add V₁ + V₂, these cancel, leaving a pure ring potential 2(r² - R²)².
#
# Individual dynamics:
#   V₁: drift has BOTH radial (→ ring) and tangential (CCW rotation)
#   V₂: drift has BOTH radial (→ ring) and tangential (CW rotation)
#
# Combined (V₁+V₂): drift is PURELY RADIAL (→ ring, no rotation)
#
# This is a fragile cancellation. Energy models get it for free (add scalars
# → xy terms cancel → gradient is purely radial). Velocity models must
# somehow cancel the rotation through vector addition, and any error in the
# learned rotation shows up as angular displacement on the ring.

ROTATION_STRENGTH = 3.0  # how strongly particles spiral
RING_RADIUS = 1.2        # radius of the equilibrium ring


def potential_v1(xy: torch.Tensor) -> torch.Tensor:
    """
    V₁(x,y) = -a·xy + (r² - R²)²

    The -a·xy term creates counterclockwise rotation. The (r²-R²)² term
    confines particles to a ring at radius R. Particles spiral inward (from
    outside) or outward (from inside) while rotating CCW, eventually settling
    on the ring.
    """
    x, y = xy[:, 0], xy[:, 1]
    r_sq = x**2 + y**2
    return -ROTATION_STRENGTH * x * y + (r_sq - RING_RADIUS**2) ** 2


def potential_v2(xy: torch.Tensor) -> torch.Tensor:
    """
    V₂(x,y) = +a·xy + (r² - R²)²

    Same as V₁ but with OPPOSITE sign on the cross term.
    Particles spiral CW onto the same ring — mirror image of V₁.
    """
    x, y = xy[:, 0], xy[:, 1]
    r_sq = x**2 + y**2
    return ROTATION_STRENGTH * x * y + (r_sq - RING_RADIUS**2) ** 2


def potential_combined(xy: torch.Tensor) -> torch.Tensor:
    """
    V₁ + V₂ = 2(r² - R²)²

    The ±a·xy terms cancel exactly. What remains is a pure ring potential.
    The gradient 8r̂(r²-R²) is purely radial — particles go straight to the
    ring with no rotation at all.

    This means WHERE a particle lands on the ring equals its initial angle.
    If composition introduces any rotation, particles land at WRONG angles,
    producing a visibly different distribution.
    """
    return potential_v1(xy) + potential_v2(xy)


def compute_gradient(potential_fn, xy: torch.Tensor) -> torch.Tensor:
    """Compute ∇V(x,y) using PyTorch's autograd."""
    xy = xy.detach().requires_grad_(True)
    V = potential_fn(xy)
    grad = torch.autograd.grad(V.sum(), xy)[0]
    return grad.detach()


# ============================================================================
# STEP 2: TRAINING DATA (PARTICLE TRAJECTORIES)
# ============================================================================
# We simulate particles under each potential and record (position, drift)
# pairs. V₁ trajectories are CCW spirals; V₂ trajectories are CW spirals.
# The training distributions are different because spiraling in opposite
# directions visits different parts of phase space.


def generate_training_data(
    potential_fn,
    n_particles: int = 2_000,
    n_steps: int = 80,
    dt: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run gradient flow to create (position, drift) training pairs.

    Returns stacked tensors of all positions and corresponding drifts
    from the trajectory. Data is concentrated near the ring (where particles
    spend the most time) with sparser coverage far from it.
    """
    x = (torch.rand(n_particles, 2) - 0.5) * 4.0  # uniform in [-2, 2]²

    all_positions = []
    all_drifts = []

    for _ in range(n_steps):
        grad = compute_gradient(potential_fn, x)
        drift = -grad

        all_positions.append(x.clone())
        all_drifts.append(drift.clone())

        x = x + dt * drift

    return torch.cat(all_positions), torch.cat(all_drifts)


# ============================================================================
# STEP 3: THE TWO MODEL TYPES
# ============================================================================
# Same architecture, different output structure:
#   Energy:   2D input → scalar output → gradient via autograd = drift
#   Velocity: 2D input → 2D output = drift directly
#
# The energy model's drift is ALWAYS curl-free (∂²V/∂x∂y = ∂²V/∂y∂x).
# The velocity model's output has no such constraint.


class EnergyModel(nn.Module):
    """
    Learns a scalar energy V̂(x,y). Drift = -∇V̂ is always curl-free.

    When two energy models are added, V̂₁ + V̂₂ is still a scalar.
    The combined drift -(∇V̂₁ + ∇V̂₂) = -∇(V̂₁ + V̂₂) is still curl-free.
    Rotation CANNOT leak through an energy composition.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy).squeeze(-1)

    def drift(self, xy: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            xy_g = xy.detach().requires_grad_(True)
            V = self.forward(xy_g)
            grad = torch.autograd.grad(V.sum(), xy_g)[0]
        return -grad.detach()


class VelocityModel(nn.Module):
    """
    Learns a vector field v̂(x,y) directly. No structural constraint.

    The network can output any 2D vector — including vectors with "curl"
    (rotational components). When two velocity models are summed, any
    curl in each model adds up rather than canceling.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)

    def drift(self, xy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(xy)


# ============================================================================
# STEP 4: TRAINING
# ============================================================================


def train_energy(
    model, positions, drifts, n_iters=3_000, batch_size=512, lr=1e-3,
):
    """
    Train so that -∇V̂(x) ≈ drift. Loss: ||∇V̂(x) + drift||².

    Requires create_graph=True so backprop reaches model weights through
    the autograd.grad computation.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(n_iters):
        idx = torch.randint(0, len(positions), (batch_size,))
        x = positions[idx].detach().requires_grad_(True)
        target = drifts[idx]

        V = model(x)
        grad_V = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

        loss = nn.functional.mse_loss(-grad_V, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"    Step {i:5d}/{n_iters} | Loss: {loss.item():.6f}")

    print(f"    Done. Final loss: {loss.item():.6f}")


def train_velocity(
    model, positions, drifts, n_iters=3_000, batch_size=512, lr=1e-3,
):
    """Train so that v̂(x) ≈ drift. Loss: ||v̂(x) - drift||²."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(n_iters):
        idx = torch.randint(0, len(positions), (batch_size,))
        x = positions[idx]
        target = drifts[idx]

        loss = nn.functional.mse_loss(model(x), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"    Step {i:5d}/{n_iters} | Loss: {loss.item():.6f}")

    print(f"    Done. Final loss: {loss.item():.6f}")


# ============================================================================
# STEP 5: SIMULATION & COMPOSITION
# ============================================================================


def simulate(drift_fn, x_init, n_steps=300, dt=0.005):
    """
    x_{t+1} = x_t + dt · drift(x_t). Returns list of snapshots.
    """
    trajectory = [x_init.clone()]
    x = x_init.clone()
    for _ in range(n_steps):
        v = drift_fn(x)
        x = (x + dt * v).detach()
        trajectory.append(x.clone())
    return trajectory


def make_composed_drift(models):
    def drift_fn(xy):
        total = torch.zeros_like(xy)
        for m in models:
            total = total + m.drift(xy)
        return total
    return drift_fn


def make_truth_drift(potential_fn):
    def drift_fn(xy):
        return -compute_gradient(potential_fn, xy)
    return drift_fn


# ============================================================================
# STEP 6: MEASURING CURL
# ============================================================================
# curl(v) = ∂v_y/∂x - ∂v_x/∂y
#
# For a gradient field v = ∇V: curl = ∂²V/∂x∂y - ∂²V/∂y∂x = 0 always.
# For an arbitrary vector field: curl can be anything.


def measure_curl(drift_fn, grid_range=2.0, grid_n=30):
    """Compute curl of a drift function on a grid via finite differences."""
    xs = torch.linspace(-grid_range, grid_range, grid_n)
    ys = torch.linspace(-grid_range, grid_range, grid_n)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    eps = 0.01
    vy_xp = drift_fn(points + torch.tensor([eps, 0.0]))[:, 1]
    vy_xm = drift_fn(points - torch.tensor([eps, 0.0]))[:, 1]
    vx_yp = drift_fn(points + torch.tensor([0.0, eps]))[:, 0]
    vx_ym = drift_fn(points - torch.tensor([0.0, eps]))[:, 0]

    curl = ((vy_xp - vy_xm) - (vx_yp - vx_ym)) / (2 * eps)
    return gx.numpy(), gy.numpy(), curl.reshape(grid_n, grid_n).detach().numpy()


# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================


def visualize(
    traj_v1_truth, traj_v2_truth, traj_combined_truth,
    traj_energy_composed, traj_velocity_composed,
    composed_energy_fn, composed_velocity_fn,
):
    """
    3×3 grid:
      Row 0: Individual spiral dynamics (CCW, CW) and ground truth (radial).
      Row 1: Composition comparison — particles on the ring.
      Row 2: Curl heatmaps — the structural proof.
    """
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    lim = 2.5

    def make_ax(row, col):
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def draw_ring(ax):
        """Draw the equilibrium ring for reference."""
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta),
                "k--", alpha=0.2, linewidth=1)

    def add_scatter(ax, points, color, **kw):
        pts = points.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=2, alpha=0.5, **kw)

    def add_traces(ax, trajectory, n=12, color="gray"):
        idxs = np.linspace(0, len(trajectory[0]) - 1, n, dtype=int)
        for i in idxs:
            path = torch.stack([t[i] for t in trajectory]).numpy()
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.25, lw=0.7)

    # --- Row 0: individual dynamics ---
    for col, (traj, title, clr) in enumerate([
        (traj_v1_truth, "V₁: spiral CCW onto ring", "steelblue"),
        (traj_v2_truth, "V₂: spiral CW onto ring", "steelblue"),
        (traj_combined_truth, "V₁+V₂ truth: straight to ring\n(no rotation)", "steelblue"),
    ]):
        ax = make_ax(0, col)
        draw_ring(ax)
        add_traces(ax, traj, color=clr)
        add_scatter(ax, traj[-1], clr)
        ax.set_title(title, fontsize=10)

    fig.text(0.02, 0.82, "Individual\nPotentials", fontsize=11,
             fontweight="bold", ha="left", va="center")

    # --- Row 1: composition comparison ---
    truth_final = traj_combined_truth[-1]
    energy_final = traj_energy_composed[-1]
    velocity_final = traj_velocity_composed[-1]

    mse_energy = ((energy_final - truth_final) ** 2).mean().item()
    mse_velocity = ((velocity_final - truth_final) ** 2).mean().item()

    for col, (traj, title, clr) in enumerate([
        (traj_combined_truth, "Ground truth V₁+V₂\n(radial convergence)", "seagreen"),
        (traj_energy_composed,
         f"Energy composed (Ê₁+Ê₂)\nMSE: {mse_energy:.4f}", "seagreen"),
        (traj_velocity_composed,
         f"Velocity composed (v̂₁+v̂₂)\nMSE: {mse_velocity:.4f}", "coral"),
    ]):
        ax = make_ax(1, col)
        draw_ring(ax)
        add_traces(ax, traj, n=15, color=clr)
        add_scatter(ax, traj[-1], clr)
        ax.set_title(title, fontsize=10)

    fig.text(0.02, 0.50, "Composed\nModels", fontsize=11,
             fontweight="bold", ha="left", va="center")

    # --- Row 2: curl heatmaps ---
    _, _, curl_truth = measure_curl(make_truth_drift(potential_combined))
    gx, gy, curl_energy = measure_curl(composed_energy_fn)
    _, _, curl_velocity = measure_curl(composed_velocity_fn)

    vmax = max(np.abs(curl_truth).max(), np.abs(curl_energy).max(),
               np.abs(curl_velocity).max(), 0.01)
    curl_kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")

    curl_data = [
        (curl_truth, "Curl: ground truth\n(zero by construction)"),
        (curl_energy, f"Curl: energy composed\n(mean |curl|: {np.abs(curl_energy).mean():.4f})"),
        (curl_velocity, f"Curl: velocity composed\n(mean |curl|: {np.abs(curl_velocity).mean():.4f})"),
    ]
    for col, (curl, title) in enumerate(curl_data):
        ax = make_ax(2, col)
        im = ax.pcolormesh(gx, gy, curl, **curl_kw)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta),
                "k--", alpha=0.3, linewidth=1)
        ax.set_title(title, fontsize=10)

    fig.text(0.02, 0.17, "Curl\n(rotation)", fontsize=11,
             fontweight="bold", ha="left", va="center")

    # Colorbar for curl row
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.25])
    fig.colorbar(im, cax=cbar_ax, label="curl (∂v_y/∂x − ∂v_x/∂y)")

    fig.suptitle(
        "Energy vs. Velocity Composition:\n"
        "Opposing Rotations Cancel in Energy Sum, Leak Through Velocity Sum",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.savefig("energy_vs_velocity.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to energy_vs_velocity.png")
    plt.show()


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print(" Energy vs. Velocity: Why Parameterization Matters")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Generate training data
    # ------------------------------------------------------------------
    print("\n① Generating training data (particle trajectories)...")

    t0 = time.time()
    pos_v1, drift_v1 = generate_training_data(potential_v1)
    pos_v2, drift_v2 = generate_training_data(potential_v2)
    print(f"   V₁: {len(pos_v1):,} training pairs  (CCW spirals)")
    print(f"   V₂: {len(pos_v2):,} training pairs  (CW spirals)")
    print(f"   ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Train four models
    # ------------------------------------------------------------------
    N_ITERS = 3_000

    print(f"\n② Training models ({N_ITERS} iterations each)...")

    print("\n  Energy model for V₁:")
    energy_1 = EnergyModel()
    t0 = time.time()
    train_energy(energy_1, pos_v1, drift_v1, n_iters=N_ITERS)
    time_e1 = time.time() - t0

    print("\n  Velocity model for V₁:")
    velocity_1 = VelocityModel()
    t0 = time.time()
    train_velocity(velocity_1, pos_v1, drift_v1, n_iters=N_ITERS)
    time_v1 = time.time() - t0

    print("\n  Energy model for V₂:")
    energy_2 = EnergyModel()
    t0 = time.time()
    train_energy(energy_2, pos_v2, drift_v2, n_iters=N_ITERS)
    time_e2 = time.time() - t0

    print("\n  Velocity model for V₂:")
    velocity_2 = VelocityModel()
    t0 = time.time()
    train_velocity(velocity_2, pos_v2, drift_v2, n_iters=N_ITERS)
    time_v2 = time.time() - t0

    print(f"\n   Training time: energy {time_e1 + time_e2:.1f}s, "
          f"velocity {time_v1 + time_v2:.1f}s")

    # ------------------------------------------------------------------
    # 3. Test individual models
    # ------------------------------------------------------------------
    print("\n③ Testing individual models...")

    torch.manual_seed(42)
    x_init = (torch.rand(1_500, 2) - 0.5) * 4.0

    traj_v1_truth = simulate(make_truth_drift(potential_v1), x_init)
    traj_v2_truth = simulate(make_truth_drift(potential_v2), x_init)

    traj_v1_energy = simulate(energy_1.drift, x_init)
    traj_v1_velocity = simulate(velocity_1.drift, x_init)
    traj_v2_energy = simulate(energy_2.drift, x_init)
    traj_v2_velocity = simulate(velocity_2.drift, x_init)

    mse_v1_e = ((traj_v1_energy[-1] - traj_v1_truth[-1]) ** 2).mean().item()
    mse_v1_v = ((traj_v1_velocity[-1] - traj_v1_truth[-1]) ** 2).mean().item()
    mse_v2_e = ((traj_v2_energy[-1] - traj_v2_truth[-1]) ** 2).mean().item()
    mse_v2_v = ((traj_v2_velocity[-1] - traj_v2_truth[-1]) ** 2).mean().item()

    print(f"   V₁ individual MSE — energy: {mse_v1_e:.6f}  "
          f"velocity: {mse_v1_v:.6f}")
    print(f"   V₂ individual MSE — energy: {mse_v2_e:.6f}  "
          f"velocity: {mse_v2_v:.6f}")
    print("   (Both model types should work well individually.)")

    # ------------------------------------------------------------------
    # 4. The composition test
    # ------------------------------------------------------------------
    print("\n④ Composing models (the main event)...")

    traj_combined_truth = simulate(make_truth_drift(potential_combined), x_init)

    composed_energy_fn = make_composed_drift([energy_1, energy_2])
    traj_energy_composed = simulate(composed_energy_fn, x_init)

    composed_velocity_fn = make_composed_drift([velocity_1, velocity_2])
    traj_velocity_composed = simulate(composed_velocity_fn, x_init)

    truth_final = traj_combined_truth[-1]
    mse_energy = ((traj_energy_composed[-1] - truth_final) ** 2).mean().item()
    mse_velocity = ((traj_velocity_composed[-1] - truth_final) ** 2).mean().item()

    print(f"\n   Composition MSE vs. ground truth:")
    print(f"   Energy  (Ê₁ + Ê₂):   {mse_energy:.4f}")
    print(f"   Velocity (v̂₁ + v̂₂):  {mse_velocity:.4f}")
    if mse_energy > 0:
        print(f"   Velocity is {mse_velocity / mse_energy:.1f}× worse.")

    # ------------------------------------------------------------------
    # 5. Measure curl
    # ------------------------------------------------------------------
    print("\n⑤ Measuring curl of composed vector fields...")

    _, _, curl_energy = measure_curl(composed_energy_fn)
    _, _, curl_velocity = measure_curl(composed_velocity_fn)

    print(f"   Energy composed:   mean |curl| = {np.abs(curl_energy).mean():.6f}")
    print(f"   Velocity composed: mean |curl| = {np.abs(curl_velocity).mean():.6f}")
    if np.abs(curl_energy).mean() > 0:
        print(f"   Velocity curl is {np.abs(curl_velocity).mean() / np.abs(curl_energy).mean():.0f}× "
              f"higher than energy curl.")
    print("   (Energy is near zero by construction; velocity has residual rotation.)")

    # ------------------------------------------------------------------
    # 6. Visualize
    # ------------------------------------------------------------------
    print("\n⑥ Plotting...")
    visualize(
        traj_v1_truth, traj_v2_truth, traj_combined_truth,
        traj_energy_composed, traj_velocity_composed,
        composed_energy_fn, composed_velocity_fn,
    )
