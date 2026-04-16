#!/usr/bin/env python3
"""
Diffusion Models from Scratch — A Minimal 2D Particle Example
=============================================================

Diffusion models generate data (images, audio, 3D models...) by learning to
reverse a noise-adding process. This script demonstrates the core idea using
simple 2D particles so you can see exactly what's happening.

THE BIG IDEA
------------
    1. FORWARD:  Take real data. Gradually add random noise until it's static.
    2. TRAIN:    Teach a neural network to predict the noise that was added.
    3. REVERSE:  Start from pure static. Use the network to remove noise,
                 one small step at a time. Out comes new data.

Why does this work? Because "predict what noise was added to this" is a much
easier problem than "generate a realistic sample from scratch." And if you can
predict the noise, you can subtract it to get something slightly cleaner.
Chain enough small denoising steps together, and you go all the way from random
static to a realistic sample.

RUNNING
-------
    $ uv run main.py            # generates diffusion_process.png

READING GUIDE
-------------
The code is organized as a pipeline. Read top-to-bottom:

    create_training_data   — The real data we want to learn to generate.
    create_noise_schedule  — A recipe for how much noise to add at each step.
    add_noise              — The forward process: destroy data by adding noise.
    NoisePredictor         — A neural net that learns to predict noise.
    train                  — Teach the net to predict noise at every level.
    generate               — The reverse process: turn static into data.
    visualize              — Plot both directions so you can see it working.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: THE TARGET DISTRIBUTION
# ============================================================================


def create_training_data(n_samples: int = 10_000) -> torch.Tensor:
    """
    Generate the "real" data distribution: points arranged in a circle.

    We use a circle because:
      - It's instantly recognizable (easy to tell if the model learned it).
      - It's NOT a Gaussian blob, so the model must learn actual structure.
      - It's simple enough to work with a tiny neural network.

    In a real application, this would be your dataset of images, audio, etc.

    Args:
        n_samples: How many 2D points to generate.

    Returns:
        Tensor of shape (n_samples, 2) — each row is an (x, y) point on/near
        the unit circle.
    """
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radius = 1.0 + np.random.normal(0, 0.1, n_samples)  # slight wobble
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


# ============================================================================
# STEP 2: THE NOISE SCHEDULE
# ============================================================================


def create_noise_schedule(n_steps: int = 100) -> dict:
    """
    Define how much noise to add at each timestep.

    Think of this like a volume knob that goes from "barely any static" to
    "nothing but static" over n_steps increments.

    Key terms (don't worry about memorizing these — the code is what matters):

        beta[t]       How much noise to add at step t.
                      Starts tiny (~0.0001), ends moderate (~0.05).

        alpha[t]      = 1 - beta[t].
                      The fraction of signal that survives one step.

        alpha_bar[t]  = alpha[0] × alpha[1] × ... × alpha[t].
                      The fraction of ORIGINAL signal surviving after all steps
                      up to t. This is the important one — it lets us jump
                      directly from clean data to any noise level in one shot,
                      instead of adding noise one step at a time.

    Args:
        n_steps: Number of diffusion steps (more steps = finer-grained noise
                 removal, but slower generation).

    Returns:
        Dictionary with schedule tensors, keyed by name.
    """
    # Linear ramp: noise increases steadily from step 0 to step T
    betas = torch.linspace(1e-4, 0.05, n_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # running product

    return {
        "betas": betas,            # β_t  — noise added at step t
        "alphas": alphas,          # α_t  — signal kept at step t
        "alpha_bars": alpha_bars,  # ᾱ_t  — total signal remaining after t steps
        "n_steps": n_steps,
    }


# ============================================================================
# STEP 3: THE FORWARD PROCESS (DESTROYING DATA)
# ============================================================================


def add_noise(
    x_clean: torch.Tensor, t: torch.Tensor, schedule: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Noise up clean data, jumping directly to noise level t.

    Instead of adding noise one step at a time, we use a shortcut formula:

        x_noisy  =  √(ᾱ_t) · x_clean   +   √(1 - ᾱ_t) · noise
                     ─────────────────       ────────────────────
                     original signal          random noise
                     (fading out)             (fading in)

    The two weights always sum (in a squared sense) to 1, so this is a smooth
    blend from "all signal" to "all noise":

        t ≈ 0:   ᾱ ≈ 1   →  x_noisy ≈ x_clean       (barely any noise)
        t ≈ T:   ᾱ ≈ 0   →  x_noisy ≈ pure noise     (signal is gone)

    Args:
        x_clean:  Original data points, shape (batch_size, 2).
        t:        Timestep for each point, shape (batch_size,). Integers in
                  [0, n_steps).
        schedule: Noise schedule from create_noise_schedule().

    Returns:
        x_noisy:  The noised-up data at timestep t, shape (batch_size, 2).
        noise:    The exact noise vector that was added — we'll train the
                  network to predict this.
    """
    # Look up how much original signal survives at timestep t
    alpha_bar_t = schedule["alpha_bars"][t].unsqueeze(-1)  # (batch_size, 1)

    # Sample fresh noise
    noise = torch.randn_like(x_clean)

    # Blend signal and noise
    x_noisy = torch.sqrt(alpha_bar_t) * x_clean + torch.sqrt(1 - alpha_bar_t) * noise

    return x_noisy, noise


# ============================================================================
# STEP 4: THE NEURAL NETWORK
# ============================================================================


class NoisePredictor(nn.Module):
    """
    A neural network that looks at a noisy point and predicts the noise.

    Given:
        - A noisy 2D point (could be anywhere — depends on noise level)
        - Which timestep t it was noised to (tells the net how noisy it is)

    It outputs: the predicted 2D noise vector that was added.

    If it predicts correctly, we can subtract the noise to recover something
    cleaner. That's the whole trick behind the reverse process.

    In production diffusion models (Stable Diffusion, DALL·E, etc.), this is
    a massive U-Net or Transformer with hundreds of millions of parameters.
    Here it's a simple 3-layer MLP, because our data is just 2D points.
    """

    def __init__(self, n_steps: int, hidden_dim: int = 128):
        super().__init__()
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),          # input:  (x, y, t_normalized)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),          # output: (noise_x, noise_y)
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise present in a noisy sample.

        Args:
            x_noisy: Noisy 2D points, shape (batch_size, 2).
            t:       Timestep indices, shape (batch_size,). Integers.

        Returns:
            Predicted noise vectors, shape (batch_size, 2).
        """
        # Normalize timestep to [0, 1] so the network sees a stable input range
        t_norm = (t.float() / self.n_steps).unsqueeze(-1)  # (batch_size, 1)

        # Feed [x, y, t] through the network
        net_input = torch.cat([x_noisy, t_norm], dim=-1)   # (batch_size, 3)
        return self.net(net_input)


# ============================================================================
# STEP 5: TRAINING
# ============================================================================


def train(
    model: NoisePredictor,
    data: torch.Tensor,
    schedule: dict,
    n_iterations: int = 5_000,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    """
    Train the noise predictor.

    The training loop is almost comically simple:

        1. Grab a random batch of clean data points.
        2. Pick a random noise level (timestep) for each point.
        3. Add that much noise (forward process).
        4. Ask the network: "what noise was added?"
        5. Penalize wrong answers (mean squared error).
        6. Update the network weights.
        7. Repeat until the network gets good at predicting noise.

    That's the entire training procedure for a diffusion model. Everything
    else is architecture, scale, and engineering.

    Args:
        model:        The NoisePredictor to train.
        data:         Clean training data, shape (n_samples, 2).
        schedule:     Noise schedule from create_noise_schedule().
        n_iterations: Number of gradient update steps.
        batch_size:   How many points per training step.
        lr:           Learning rate for the Adam optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps = schedule["n_steps"]

    for i in range(n_iterations):
        # 1. Random batch of clean data
        idx = torch.randint(0, len(data), (batch_size,))
        x_clean = data[idx]

        # 2. Random timestep for each point
        t = torch.randint(0, n_steps, (batch_size,))

        # 3. Add noise (forward process)
        x_noisy, true_noise = add_noise(x_clean, t, schedule)

        # 4. Ask the network to predict the noise
        predicted_noise = model(x_noisy, t)

        # 5. How wrong was the prediction?
        loss = nn.functional.mse_loss(predicted_noise, true_noise)

        # 6. Update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"  Step {i:5d}/{n_iterations} | Loss: {loss.item():.4f}")

    print(f"  Done. Final loss: {loss.item():.4f}")


# ============================================================================
# STEP 6: SAMPLING (GENERATING NEW DATA)
# ============================================================================


@torch.no_grad()
def generate(
    model: NoisePredictor,
    schedule: dict,
    n_samples: int = 1_000,
) -> list[torch.Tensor]:
    """
    Generate new data by reversing the diffusion process.

    This is where it all pays off:

        1. Start with pure random noise (like TV static).
        2. Ask the network: "what noise do you see?"
        3. Subtract (a scaled version of) the predicted noise.
        4. Add a tiny bit of fresh randomness (keeps the process stochastic).
        5. Repeat from step 2 at the next lower noise level.
        6. After all steps: you have new, clean data points.

    Each step makes the data a little less noisy, like slowly turning down
    the static on a TV until the picture comes through.

    The formula for each reverse step (you do NOT need to memorize this):

        x_{t-1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · predicted_noise) + σ_t · z

    In English: "take the noisy point, subtract a scaled version of the
    predicted noise to partially denoise it, then add a small amount of
    fresh randomness." The network plus this formula together undo one step
    of the forward noise process.

    Args:
        model:     A trained NoisePredictor.
        schedule:  Noise schedule (same one used for training).
        n_samples: How many new points to generate.

    Returns:
        Trajectory: a list of tensors showing the data at each step, from
        pure noise (index 0) to the final generated data (last index).
        Useful for visualizing the reverse process in action.
    """
    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alpha_bars = schedule["alpha_bars"]
    n_steps = schedule["n_steps"]

    # Start from pure random noise
    x = torch.randn(n_samples, 2)
    trajectory = [x.clone()]

    # Walk backwards through the noise levels
    for t in reversed(range(n_steps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long)

        # "What noise do you see?"
        predicted_noise = model(x, t_batch)

        # Partially subtract the predicted noise (the reverse-step formula)
        coeff = betas[t] / torch.sqrt(1.0 - alpha_bars[t])
        x = (1.0 / torch.sqrt(alphas[t])) * (x - coeff * predicted_noise)

        # Add a tiny bit of fresh randomness (except at the very last step,
        # where we want a clean final result)
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

        trajectory.append(x.clone())

    return trajectory


# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================


def visualize(
    training_data: torch.Tensor,
    schedule: dict,
    trajectory: list[torch.Tensor],
) -> None:
    """
    Plot the full diffusion story in two rows:

        Top row:    FORWARD — clean data dissolving into noise (left → right).
        Bottom row: REVERSE — noise crystallizing into data (left → right).

    The rightmost panel in the bottom row should look like the leftmost panel
    in the top row — that's the visual proof that generation worked.

    Args:
        training_data: The original clean data (for the forward demo).
        schedule:      Noise schedule.
        trajectory:    Output of generate() — snapshots at each reverse step.
    """
    n_steps = schedule["n_steps"]
    n_cols = 6
    fig, axes = plt.subplots(2, n_cols, figsize=(18, 6))
    plot_kwargs = dict(s=1, alpha=0.5)
    axis_limit = 3.5

    # --- Top row: forward process (data → noise) ---
    forward_times = np.linspace(0, n_steps - 1, n_cols, dtype=int)
    sample = training_data[:1000]

    for col, t in enumerate(forward_times):
        t_batch = torch.full((len(sample),), t, dtype=torch.long)
        x_noisy, _ = add_noise(sample, t_batch, schedule)

        ax = axes[0, col]
        ax.scatter(x_noisy[:, 0], x_noisy[:, 1], c="steelblue", **plot_kwargs)
        alpha_bar = schedule["alpha_bars"][t].item()
        ax.set_title(f"t={t}  (signal: {alpha_bar:.0%})", fontsize=9)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel(
        "Forward\n(adding noise)", fontsize=11, fontweight="bold"
    )

    # --- Bottom row: reverse process (noise → data) ---
    reverse_indices = np.linspace(0, len(trajectory) - 1, n_cols, dtype=int)

    for col, idx in enumerate(reverse_indices):
        points = trajectory[idx].numpy()

        ax = axes[1, col]
        ax.scatter(points[:, 0], points[:, 1], c="coral", **plot_kwargs)
        ax.set_title(f"step {idx}/{n_steps}", fontsize=9)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].set_ylabel(
        "Reverse\n(removing noise)", fontsize=11, fontweight="bold"
    )

    fig.suptitle(
        "Diffusion in Action: 2D Particles",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig("diffusion_process.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to diffusion_process.png")
    plt.show()


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Diffusion Models from Scratch — 2D Particle Demo")
    print("=" * 60)

    # 1. The data we want to learn to generate
    print("\n① Creating training data (circle of 2D points)...")
    data = create_training_data(n_samples=10_000)
    print(f"   Generated {len(data)} points.")

    # 2. The noise recipe
    print("\n② Setting up noise schedule...")
    schedule = create_noise_schedule(n_steps=100)
    print(f"   {schedule['n_steps']} diffusion steps.")
    print(f"   Signal remaining after all steps: "
          f"{schedule['alpha_bars'][-1]:.4f}  (≈0 means fully destroyed)")

    # 3. Train the noise predictor
    print("\n③ Training the noise predictor...")
    model = NoisePredictor(n_steps=schedule["n_steps"])
    train(model, data, schedule, n_iterations=8_000)

    # 4. Generate new samples by reversing the diffusion
    print("\n④ Generating new samples (reversing the diffusion)...")
    trajectory = generate(model, schedule, n_samples=1_000)
    print(f"   Created {len(trajectory[-1])} new points in {schedule['n_steps']} "
          f"reverse steps.")

    # 5. Visualize
    print("\n⑤ Plotting...")
    visualize(data, schedule, trajectory)
