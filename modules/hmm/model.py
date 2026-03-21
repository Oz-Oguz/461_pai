"""HMM algorithms: Forward (filtering), Viterbi, and Robot Localization.

Pedagogical implementations for teaching Hidden Markov Models:
- Forward algorithm for belief updating and likelihood calculation
- Viterbi algorithm for most likely state sequence
- 2D robot localization with movement and sensor models
"""

from __future__ import annotations

import numpy as np
from typing import Literal


# ══════════════════════════════════════════════════════════════════════
# Example 1: Weather-Mood (Forward Algorithm)
# ══════════════════════════════════════════════════════════════════════

def forward_algorithm(
    observations: list[str],
    states: list[str],
    transition: dict[str, dict[str, float]],
    emission: dict[str, dict[str, float]],
    prior: dict[str, float],
) -> dict:
    """
    Run the Forward algorithm to compute filtered beliefs P(X_t | e_{1:t}).
    
    Args:
        observations: Sequence of observations (e.g., ["Happy", "Sad", "Happy"])
        states: List of state names (e.g., ["Sun", "Rain"])
        transition: P(X_t | X_{t-1}) as nested dict
        emission: P(E_t | X_t) as nested dict
        prior: P(X_0) as dict
    
    Returns:
        Dictionary with beliefs, likelihoods, and step-by-step math
    """
    n_steps = len(observations)
    n_states = len(states)
    
    # Store beliefs at each timestep
    beliefs = np.zeros((n_steps + 1, n_states))
    beliefs[0] = [prior[s] for s in states]
    
    # Store log-likelihoods
    log_likelihood = 0.0
    likelihoods = []
    
    # Store step details for visualization
    steps = []
    
    for t in range(n_steps):
        obs = observations[t]
        
        # Prediction step: B'(X_{t+1}) = Σ_x_t P(X_{t+1} | x_t) B(x_t)
        predicted = np.zeros(n_states)
        for j, s_next in enumerate(states):
            for i, s_prev in enumerate(states):
                predicted[j] += transition[s_prev][s_next] * beliefs[t, i]
        
        # Update step: B(X_{t+1}) ∝ P(e_{t+1} | X_{t+1}) B'(X_{t+1})
        updated = np.array([emission[s][obs] * predicted[i] for i, s in enumerate(states)])
        
        # Normalize
        evidence = updated.sum()
        if evidence > 0:
            updated /= evidence
        
        beliefs[t + 1] = updated
        
        # Track likelihood
        likelihoods.append(evidence)
        log_likelihood += np.log(evidence) if evidence > 0 else -np.inf
        
        # Store step details with LaTeX
        steps.append({
            "timestep": t + 1,
            "observation": obs,
            "predicted_belief": {s: float(predicted[i]) for i, s in enumerate(states)},
            "updated_belief": {s: float(updated[i]) for i, s in enumerate(states)},
            "evidence": float(evidence),
            "latex": _forward_step_latex(t, obs, states, predicted, updated, evidence, transition, emission),
        })
    
    return {
        "states": states,
        "observations": observations,
        "beliefs": [[float(beliefs[t, i]) for i in range(n_states)] for t in range(n_steps + 1)],
        "state_labels": states,
        "log_likelihood": float(log_likelihood),
        "steps": steps,
    }


def _forward_step_latex(
    t: int,
    obs: str,
    states: list[str],
    predicted: np.ndarray,
    updated: np.ndarray,
    evidence: float,
    transition: dict,
    emission: dict,
) -> str:
    """Generate LaTeX for one forward step."""
    lines = [
        f"\\text{{Step }} t = {t + 1}, \\quad e_{{{t + 1}}} = \\text{{{obs}}}",
        "",
        "\\text{Predict:} \\quad B'(X_{" + str(t + 1) + "}) = \\sum_{x_" + str(t) + "} P(X_{" + str(t + 1) + "} | x_" + str(t) + "}) B(x_" + str(t) + "})",
    ]
    
    for i, s in enumerate(states):
        lines.append(f"B'(\\text{{{s}}}) = {predicted[i]:.4f}")
    
    lines.extend([
        "",
        f"\\text{{Update:}} \\quad B(X_{{{t + 1}}}) \\propto P(e_{{{t + 1}}} | X_{{{t + 1}}}) B'(X_{{{t + 1}}})",
    ])
    
    for i, s in enumerate(states):
        emission_prob = emission[s][obs]
        lines.append(f"B(\\text{{{s}}}) \\propto {emission_prob:.2f} \\times {predicted[i]:.4f} = {emission_prob * predicted[i]:.4f}")
    
    lines.extend([
        "",
        f"\\text{{Evidence:}} \\quad P(e_{{{t + 1}}} | e_{{1:{t}}}) = {evidence:.4f}",
        "",
        "\\text{Normalized belief:}",
    ])
    
    for i, s in enumerate(states):
        lines.append(f"B(\\text{{{s}}}) = {updated[i]:.4f}")
    
    return "\\\\".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Example 2: Viterbi (Most Likely Path)
# ══════════════════════════════════════════════════════════════════════

def viterbi_algorithm(
    observations: list[str],
    states: list[str],
    transition: dict[str, dict[str, float]],
    emission: dict[str, dict[str, float]],
    prior: dict[str, float],
) -> dict:
    """
    Viterbi algorithm to find the most likely state sequence.
    
    Returns the maximum probability path through the HMM.
    """
    n_steps = len(observations)
    n_states = len(states)
    
    # Viterbi table: max probability of reaching each state at each time
    viterbi = np.zeros((n_steps + 1, n_states))
    viterbi[0] = [prior[s] for s in states]
    
    # Backpointers for traceback
    backpointers = np.zeros((n_steps, n_states), dtype=int)
    
    # Forward pass: maximize over previous states
    for t in range(n_steps):
        obs = observations[t]
        for j, s_next in enumerate(states):
            max_prob = -1
            max_prev = 0
            for i, s_prev in enumerate(states):
                prob = viterbi[t, i] * transition[s_prev][s_next] * emission[s_next][obs]
                if prob > max_prob:
                    max_prob = prob
                    max_prev = i
            viterbi[t + 1, j] = max_prob
            backpointers[t, j] = max_prev
    
    # Backward pass: traceback to find most likely path
    path_indices = np.zeros(n_steps + 1, dtype=int)
    path_indices[-1] = np.argmax(viterbi[-1])
    
    for t in range(n_steps - 1, -1, -1):
        path_indices[t] = backpointers[t, path_indices[t + 1]]
    
    path = [states[i] for i in path_indices]
    max_prob = viterbi[-1, path_indices[-1]]
    
    # Also compute total likelihood (sum over all paths) for comparison
    beliefs = forward_algorithm(observations, states, transition, emission, prior)
    total_likelihood = np.exp(beliefs["log_likelihood"])
    
    return {
        "states": states,
        "observations": observations,
        "most_likely_path": path,
        "path_probability": float(max_prob),
        "total_likelihood": float(total_likelihood),
        "viterbi_table": viterbi.tolist(),
        "backpointers": backpointers.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════
# Example 3: Robot Localization (2D Grid)
# ══════════════════════════════════════════════════════════════════════

def create_robot_world(
    grid_size: tuple[int, int] = (10, 10),
    walls: list[tuple[int, int]] | None = None,
) -> dict:
    """
    Create a 2D grid world for robot localization.
    
    Args:
        grid_size: (rows, cols) dimensions
        walls: List of (r, c) coordinates that are walls
    
    Returns:
        World configuration
    """
    if walls is None:
        # Default: create a simple room with walls around perimeter
        rows, cols = grid_size
        walls = []
        for r in range(rows):
            walls.append((r, 0))
            walls.append((r, cols - 1))
        for c in range(cols):
            walls.append((0, c))
            walls.append((rows - 1, c))
    
    return {
        "grid_size": grid_size,
        "walls": walls,
        "n_cells": grid_size[0] * grid_size[1],
    }


def robot_localization_step(
    belief: np.ndarray,
    grid_size: tuple[int, int],
    walls: list[tuple[int, int]],
    action: Literal["move_up", "move_down", "move_left", "move_right"] | None = None,
    observation: dict[str, bool] | None = None,
    action_noise: float = 0.1,
) -> dict:
    """
    Perform one step of robot localization (either prediction or update).
    
    Args:
        belief: Current belief state as 2D array
        grid_size: (rows, cols)
        walls: List of wall coordinates
        action: Movement action (for prediction step)
        observation: Wall sensor readings (for update step)
        action_noise: Probability that action fails
    
    Returns:
        Updated belief and step details
    """
    rows, cols = grid_size
    new_belief = belief.copy()
    
    step_type = None
    details = {}
    
    if action is not None:
        # Prediction step: apply transition model
        step_type = "predict"
        predicted = np.zeros_like(belief)
        
        # Define movement deltas
        deltas = {
            "move_up": (-1, 0),
            "move_down": (1, 0),
            "move_left": (0, -1),
            "move_right": (0, 1),
        }
        dr, dc = deltas[action]
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) in walls:
                    continue
                
                # Try to move in intended direction
                new_r, new_c = r + dr, c + dc
                
                # Check if move is valid
                if (new_r, new_c) in walls or new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
                    # Action fails, stay in place
                    predicted[r, c] += belief[r, c]
                else:
                    # With probability (1 - noise), move succeeds
                    predicted[new_r, new_c] += belief[r, c] * (1 - action_noise)
                    # With probability noise, stay in place
                    predicted[r, c] += belief[r, c] * action_noise
        
        new_belief = predicted
        details = {
            "action": action,
            "action_noise": action_noise,
            "entropy_before": float(_entropy(belief)),
            "entropy_after": float(_entropy(predicted)),
        }
    
    elif observation is not None:
        # Update step: apply sensor model
        step_type = "update"
        
        # Generate likelihood for each cell based on wall observations
        likelihood = np.ones_like(belief)
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) in walls:
                    likelihood[r, c] = 0
                    continue
                
                # Check if walls are adjacent
                has_wall = {
                    "north": (r - 1, c) in walls or r == 0,
                    "south": (r + 1, c) in walls or r == rows - 1,
                    "east": (r, c + 1) in walls or c == cols - 1,
                    "west": (r, c - 1) in walls or c == 0,
                }
                
                # Compute likelihood based on sensor readings
                sensor_accuracy = 0.9
                cell_likelihood = 1.0
                for direction, sensed in observation.items():
                    if has_wall[direction] == sensed:
                        cell_likelihood *= sensor_accuracy
                    else:
                        cell_likelihood *= (1 - sensor_accuracy)
                
                likelihood[r, c] = cell_likelihood
        
        # Multiply belief by likelihood and normalize
        new_belief = belief * likelihood
        evidence = new_belief.sum()
        if evidence > 0:
            new_belief /= evidence
        
        details = {
            "observation": observation,
            "entropy_before": float(_entropy(belief)),
            "entropy_after": float(_entropy(new_belief)),
            "information_gain": float(_entropy(belief) - _entropy(new_belief)),
        }
    
    return {
        "belief": new_belief.tolist(),
        "step_type": step_type,
        "details": details,
    }


def _entropy(belief: np.ndarray) -> float:
    """Compute Shannon entropy of belief distribution."""
    p = belief.flatten()
    p = p[p > 1e-10]  # Avoid log(0)
    return float(-np.sum(p * np.log2(p)))


def initialize_robot_belief(
    grid_size: tuple[int, int],
    walls: list[tuple[int, int]],
    uniform: bool = True,
    position: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Initialize robot belief state.
    
    Args:
        grid_size: (rows, cols)
        walls: Wall coordinates
        uniform: If True, uniform over free cells. If False, use position.
        position: Known starting position (if uniform=False)
    
    Returns:
        Initial belief as 2D array
    """
    rows, cols = grid_size
    belief = np.zeros((rows, cols))
    
    if uniform:
        # Uniform distribution over all non-wall cells
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in walls:
                    belief[r, c] = 1.0
        belief /= belief.sum()
    elif position is not None:
        # Known starting position
        r, c = position
        if (r, c) not in walls:
            belief[r, c] = 1.0
    
    return belief


def solver_steps_forward() -> list[dict]:
    """Generate step-by-step derivation of the Forward algorithm."""
    return [
        {
            "title": "Forward Algorithm: Recursive Belief Update",
            "text": "The Forward algorithm computes P(X_t | e_{1:t}) recursively by alternating prediction and update steps.",
            "latex": "B(X_t) = P(X_t | e_{1:t})",
        },
        {
            "title": "Step 1: Prediction",
            "text": "Propagate the current belief forward through the transition model by summing over all previous states.",
            "latex": "B'(X_{t+1}) = \\sum_{x_t} P(X_{t+1} | x_t) B(x_t)",
        },
        {
            "title": "Step 2: Observation Update",
            "text": "Multiply the predicted belief by the observation likelihood to obtain the posterior (unnormalized).",
            "latex": "B(X_{t+1}) \\propto P(e_{t+1} | X_{t+1}) B'(X_{t+1})",
        },
        {
            "title": "Step 3: Normalization",
            "text": "Normalize the posterior to ensure it sums to 1. The normalization constant is the evidence (likelihood of observation).",
            "latex": "B(X_{t+1}) = \\frac{P(e_{t+1} | X_{t+1}) B'(X_{t+1})}{\\sum_{x} P(e_{t+1} | x) B'(x)} = \\frac{P(e_{t+1} | X_{t+1}) B'(X_{t+1})}{P(e_{t+1} | e_{1:t})}",
        },
        {
            "title": "Evidence (Likelihood)",
            "text": "The normalization constant at each step is the likelihood of the new observation given past evidence. Multiplying these gives the total likelihood P(e_{1:T}).",
            "latex": "P(e_{1:T}) = \\prod_{t=1}^T P(e_t | e_{1:t-1})",
        },
    ]


def solver_steps_viterbi() -> list[dict]:
    """Generate step-by-step derivation of the Viterbi algorithm."""
    return [
        {
            "title": "Viterbi Algorithm: Most Likely Path",
            "text": "The Viterbi algorithm finds the single most likely state sequence x_{1:T}^* that explains the observations e_{1:T}.",
            "latex": "x_{1:T}^* = \\arg\\max_{x_{1:T}} P(x_{1:T} | e_{1:T})",
        },
        {
            "title": "Key Difference from Forward",
            "text": "Forward sums over all paths (∑), Viterbi maximizes over paths (max). This changes the recursion.",
            "latex": "\\text{Forward: } B'(X_t) = \\sum_{x_{t-1}} P(X_t | x_{t-1}) B(x_{t-1}) \\\\[6pt] \\text{Viterbi: } V(X_t) = \\max_{x_{t-1}} P(X_t | x_{t-1}) V(x_{t-1})",
        },
        {
            "title": "Step 1: Forward Pass (Max)",
            "text": "Compute the maximum probability of reaching each state at each timestep, storing backpointers.",
            "latex": "V_t(x_t) = \\max_{x_{t-1}} P(e_t | x_t) P(x_t | x_{t-1}) V_{t-1}(x_{t-1})",
        },
        {
            "title": "Step 2: Backward Traceback",
            "text": "Starting from the most likely final state, follow backpointers to reconstruct the optimal path.",
            "latex": "x_T^* = \\arg\\max_{x_T} V_T(x_T) \\\\[6pt] x_t^* = \\text{backpointer}[t+1, x_{t+1}^*]",
        },
        {
            "title": "Path Probability vs Total Likelihood",
            "text": "The Viterbi path probability is always less than or equal to the total likelihood (which sums over all paths).",
            "latex": "P(x_{1:T}^* | e_{1:T}) \\leq P(e_{1:T}) = \\sum_{\\text{all paths}} P(x_{1:T}, e_{1:T})",
        },
    ]
