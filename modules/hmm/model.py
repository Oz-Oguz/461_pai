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
    observation: dict | None = None,
    action_noise: float = 0.1,
    sensor_noise: float = 0.2,
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
        likelihood = np.zeros_like(belief)

        # ── UCB directional sensor model ─────────────────────────────────
        # Sensor reports 4 bits: N/S/E/W wall-or-not.  At most 1 bit wrong.
        #   P(e | X)  =  1 - sensor_noise   if hamming(e, true) == 0
        #             =  sensor_noise / 4   if hamming(e, true) == 1
        #             =  0                  if hamming(e, true) >= 2
        obs_n = bool(observation.get("N", False))
        obs_s = bool(observation.get("S", False))
        obs_e = bool(observation.get("E", False))
        obs_w = bool(observation.get("W", False))

        for r in range(rows):
            for c in range(cols):
                if (r, c) in walls:
                    continue
                true_n = (r - 1, c) in walls or r == 0
                true_s = (r + 1, c) in walls or r == rows - 1
                true_e = (r, c + 1) in walls or c == cols - 1
                true_w = (r, c - 1) in walls or c == 0
                hamming = sum([
                    obs_n != true_n,
                    obs_s != true_s,
                    obs_e != true_e,
                    obs_w != true_w,
                ])
                if hamming == 0:
                    likelihood[r, c] = 1.0 - sensor_noise
                elif hamming == 1:
                    likelihood[r, c] = sensor_noise / 4.0
                # hamming >= 2 → likelihood stays 0

        details_extra = {
            "observation": {"N": obs_n, "S": obs_s, "E": obs_e, "W": obs_w},
            "sensor_noise": sensor_noise,
        }

        # Multiply belief by likelihood and normalize
        new_belief = belief * likelihood
        evidence = new_belief.sum()
        if evidence > 0:
            new_belief /= evidence

        details = {
            **details_extra,
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


# ══════════════════════════════════════════════════════════════════════
# Example 4: Bayesian Filtering — Two-Phase (Passage of Time + Observation)
# ══════════════════════════════════════════════════════════════════════

# ── Transition model helpers ──────────────────────────────────────────

def _generate_spiral_order(rows: int, cols: int, wall_set: set) -> list[tuple[int, int]]:
    """Clockwise inward spiral traversal, skipping walls."""
    order: list[tuple[int, int]] = []
    top, bottom, left, right = 0, rows - 1, 0, cols - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            if (top, c) not in wall_set:
                order.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            if (r, right) not in wall_set:
                order.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                if (bottom, c) not in wall_set:
                    order.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                if (r, left) not in wall_set:
                    order.append((r, left))
            left += 1
    return order


def _clockwise_dir(r: int, c: int, rows: int, cols: int) -> tuple[int, int]:
    """Preferred clockwise direction for (r, c) within its concentric ring."""
    k = min(r, c, rows - 1 - r, cols - 1 - c)
    if r == k and c < cols - 1 - k:          # top side → right
        return (0, 1)
    elif c == cols - 1 - k and r < rows - 1 - k:  # right side → down
        return (1, 0)
    elif r == rows - 1 - k and c > k:        # bottom side → left
        return (0, -1)
    elif c == k and r > k:                   # left side → up
        return (-1, 0)
    return (0, 1)                            # fallback (single-cell ring)


def _build_preferred_dirs(
    rows: int, cols: int, wall_set: set, model: str
) -> dict[tuple[int, int], tuple[int, int] | None]:
    """
    Return preferred next-direction for every free cell.

    None  → spread uniformly (no preference).
    (dr, dc) → move in that direction with probability (1-noise), spread
               noise evenly over remaining free neighbours.

    Models
    ------
    'uniform'   – spread uniformly (no preferred direction).
    'clockwise' – each concentric ring rotates clockwise independently.
    'vortex'    – full inward spiral; cells on the same ring move to the
                  NEXT inner ring at the spiral's "corner" transitions.
                  The innermost ring falls back to clockwise rotation.
    """
    free = {(r, c) for r in range(rows) for c in range(cols) if (r, c) not in wall_set}
    pref: dict[tuple[int, int], tuple[int, int] | None] = {}

    if model == 'uniform':
        return {cell: None for cell in free}

    if model == 'clockwise':
        for r, c in free:
            pref[(r, c)] = _clockwise_dir(r, c, rows, cols)
        return pref

    if model == 'vortex':
        spiral = _generate_spiral_order(rows, cols, wall_set)
        for i, cell in enumerate(spiral):
            next_cell = spiral[(i + 1) % len(spiral)]
            dr = next_cell[0] - cell[0]
            dc = next_cell[1] - cell[1]
            if abs(dr) + abs(dc) == 1:          # adjacent → use it
                pref[cell] = (dr, dc)
            else:                                # wrap-around: fall back to clockwise
                r, c = cell
                pref[cell] = _clockwise_dir(r, c, rows, cols)
        # Fill any free cells missed by the spiral (e.g. isolated wall pockets)
        for cell in free:
            if cell not in pref:
                pref[cell] = None
        return pref

    return {cell: None for cell in free}


def bayesian_filtering_time_step(
    belief: np.ndarray,
    grid_size: tuple[int, int],
    walls: list[tuple[int, int]],
    transition_noise: float = 0.2,
    transition_model: str = 'uniform',
) -> dict:
    """
    Apply the Passage-of-Time phase of Bayesian Filtering.

    B'(X_{t+1}) = Σ_{x_t} P(X_{t+1} | x_t) · B(x_t)

    transition_model options:
        'uniform'   – spread to all free neighbours equally.
        'clockwise' – each concentric ring rotates clockwise.
        'vortex'    – full inward clockwise spiral (ghost whirlpool).

    Returns:
        belief_after, entropy_before, entropy_after, latex
    """
    rows, cols = grid_size
    wall_set = set(map(tuple, walls))
    pref = _build_preferred_dirs(rows, cols, wall_set, transition_model)
    predicted = np.zeros_like(belief)
    adjacents = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) in wall_set or belief[r, c] < 1e-12:
                continue
            b = belief[r, c]

            free_nbrs = [
                (r + dr, c + dc)
                for dr, dc in adjacents
                if 0 <= r + dr < rows and 0 <= c + dc < cols
                and (r + dr, c + dc) not in wall_set
            ]

            if not free_nbrs:
                predicted[r, c] += b
                continue

            preferred = pref.get((r, c))

            if preferred is None:
                # Uniform: robot picks one of 4 directions at random (with 1-noise);
                # blocked directions (walls/boundary) fold back into staying put.
                blocked = 4 - len(free_nbrs)
                predicted[r, c] += b * (transition_noise + blocked * (1 - transition_noise) / 4)
                for nr, nc in free_nbrs:
                    predicted[nr, nc] += b * (1 - transition_noise) / 4
            else:
                pr, pc = r + preferred[0], c + preferred[1]
                if (pr, pc) in wall_set or pr < 0 or pr >= rows or pc < 0 or pc >= cols:
                    # Preferred direction blocked: fall back to uniform
                    predicted[r, c] += b * transition_noise
                    for nr, nc in free_nbrs:
                        predicted[nr, nc] += b * (1 - transition_noise) / len(free_nbrs)
                else:
                    # Move to preferred neighbour with (1-noise), noise to others
                    predicted[pr, pc] += b * (1 - transition_noise)
                    other_nbrs = [n for n in free_nbrs if n != (pr, pc)]
                    if other_nbrs:
                        for nr, nc in other_nbrs:
                            predicted[nr, nc] += b * transition_noise / len(other_nbrs)
                    else:
                        predicted[pr, pc] += b * transition_noise  # stay if no alternatives

    total = predicted.sum()
    if total > 0:
        predicted /= total

    entropy_before = _entropy(belief)
    entropy_after = _entropy(predicted)

    model_desc = {
        'uniform':   'uniform spread to free neighbours',
        'clockwise': 'clockwise rotation within each concentric ring',
        'vortex':    'inward clockwise spiral (vortex)',
    }.get(transition_model, transition_model)

    latex = (
        "B'(X_{t+1}) = \\sum_{x_t} P(X_{t+1} | x_t)\\, B(x_t)"
        "\\\\"
        f"\\text{{Model: {model_desc}}}"
        "\\\\"
        f"\\text{{Noise: }} \\epsilon = {transition_noise:.2f}"
        "\\\\"
        f"H(B) = {entropy_before:.3f} \\to H(B') = {entropy_after:.3f} \\text{{ bits}}"
    )

    return {
        "belief_after": predicted.tolist(),
        "entropy_before": float(entropy_before),
        "entropy_after": float(entropy_after),
        "latex": latex,
    }


def build_transition_matrix(
    grid_size: tuple[int, int],
    walls: list[tuple[int, int]],
    transition_noise: float = 0.2,
    transition_model: str = 'uniform',
) -> dict:
    """
    Build the n×n row-stochastic transition matrix T where
    T[i][j] = P(X_{t+1}=j | X_t=i), indexed over free cells in row-major order.

    Returns:
        T       – n×n list-of-lists
        states  – [[r,c], ...] mapping linear index → grid cell
        n       – number of free cells
    """
    rows, cols = grid_size
    wall_set = set(map(tuple, walls))
    pref = _build_preferred_dirs(rows, cols, wall_set, transition_model)

    free_cells = [(r, c) for r in range(rows) for c in range(cols)
                  if (r, c) not in wall_set]
    n = len(free_cells)
    cell_to_idx = {cell: idx for idx, cell in enumerate(free_cells)}

    T = np.zeros((n, n))
    adjacents = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r, c in free_cells:
        i = cell_to_idx[(r, c)]
        free_nbrs = [
            (r + dr, c + dc)
            for dr, dc in adjacents
            if 0 <= r + dr < rows and 0 <= c + dc < cols
            and (r + dr, c + dc) not in wall_set
        ]
        if not free_nbrs:
            T[i][i] = 1.0
            continue

        preferred = pref.get((r, c))

        if preferred is None:          # uniform model
            # Robot picks one of 4 directions at random (with 1-noise);
            # blocked directions (walls/boundary) fold back into staying put.
            blocked = 4 - len(free_nbrs)
            T[i][i] += transition_noise + blocked * (1 - transition_noise) / 4
            for nr, nc in free_nbrs:
                T[i][cell_to_idx[(nr, nc)]] += (1 - transition_noise) / 4
        else:
            pr, pc = r + preferred[0], c + preferred[1]
            if (pr, pc) in wall_set or not (0 <= pr < rows and 0 <= pc < cols):
                # preferred direction blocked → uniform fallback
                T[i][i] += transition_noise
                for nr, nc in free_nbrs:
                    T[i][cell_to_idx[(nr, nc)]] += (1 - transition_noise) / len(free_nbrs)
            else:
                pj = cell_to_idx[(pr, pc)]
                T[i][pj] += (1 - transition_noise)
                other_nbrs = [nb for nb in free_nbrs if nb != (pr, pc)]
                if other_nbrs:
                    for nr, nc in other_nbrs:
                        T[i][cell_to_idx[(nr, nc)]] += transition_noise / len(other_nbrs)
                else:
                    T[i][pj] += transition_noise   # no alternatives

    return {
        "T": T.tolist(),
        "states": [[r, c] for r, c in free_cells],
        "n": n,
    }


def bayesian_filtering_observe_step(
    belief: np.ndarray,
    grid_size: tuple[int, int],
    walls: list[tuple[int, int]],
    observation: tuple[int, int],
    sensor_noise: float = 0.3,
) -> dict:
    """
    Apply the Observation phase of Bayesian Filtering.

    B(X_{t+1}) ∝ P(e_{t+1} | X_{t+1}) · B'(X_{t+1})

    Sensor model: Gaussian likelihood centered at the observed cell.
        P(e | X=(r,c)) ∝ exp(-manhattan_dist(X, obs) / sigma)
    where sigma = 0.5 + sensor_noise * 2.5 (sensor_noise ∈ [0,1]).

    Returns:
        belief_after, likelihood_map, entropy_before, entropy_after, latex
    """
    rows, cols = grid_size
    wall_set = set(map(tuple, walls))
    r_obs, c_obs = observation

    sigma = 0.5 + sensor_noise * 2.5

    # Build likelihood map
    likelihood = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in wall_set:
                continue
            dist = abs(r - r_obs) + abs(c - c_obs)
            likelihood[r, c] = np.exp(-dist / sigma)

    # Unnormalised posterior
    updated = belief * likelihood
    evidence = updated.sum()
    if evidence > 0:
        updated /= evidence

    entropy_before = _entropy(belief)
    entropy_after = _entropy(updated)

    latex = (
        f"B(X_{{t+1}}) \\propto P(e_{{t+1}} | X_{{t+1}}) \\cdot B'(X_{{t+1}})"
        "\\\\"
        f"\\text{{Observed position: }}({r_obs}, {c_obs})"
        "\\\\"
        f"P(e | X) \\propto \\exp\\!\\left(-\\frac{{d_{{\\text{{Manhattan}}}}}}{{\\sigma}}\\right),"
        f"\\quad \\sigma = {sigma:.2f}"
        "\\\\"
        f"H(B') = {entropy_before:.3f} \\to H(B) = {entropy_after:.3f} \\text{{ bits}}"
    )

    return {
        "belief_after": updated.tolist(),
        "likelihood_map": likelihood.tolist(),
        "entropy_before": float(entropy_before),
        "entropy_after": float(entropy_after),
        "latex": latex,
    }


def solver_steps_filtering() -> list[dict]:
    """Step-by-step derivation of Bayesian Filtering (two-phase view)."""
    return [
        {
            "title": "Base Case: Observation",
            "text": (
                "At time t=1, we can compute the posterior over X₁ given the first "
                "observation e₁ using Bayes' rule. The prior B(X₁) = P(X₁) is "
                "reweighted by the likelihood P(e₁|X₁) and renormalised."
            ),
            "latex": (
                "P(X_1 | e_1) = P(X_1,\\, e_1) / P(e_1)"
                "\\\\[4pt]"
                "\\propto_{X_1}\\, P(X_1)\\, P(e_1 | X_1)"
            ),
        },
        {
            "title": "Base Case: Passage of Time",
            "text": (
                "To predict X₂ before seeing any new evidence, we marginalise out X₁ "
                "using the transition model. The belief 'spreads' through the "
                "transition — uncertainty accumulates."
            ),
            "latex": (
                "P(X_2) = \\sum_{x_1} P(X_2,\\, x_1)"
                "\\\\[4pt]"
                "= \\sum_{x_1} P(X_2 | x_1)\\, P(x_1)"
            ),
        },
        {
            "title": "Passage of Time (General)",
            "text": (
                "Given the current filtered belief B(Xₜ) = P(Xₜ | e₁..ₜ), "
                "one time step later (before the next observation) we obtain the "
                "predicted belief B'(Xₜ₊₁). Beliefs are pushed through the transition; "
                "entropy increases."
            ),
            "latex": (
                "B'(X_{t+1}) = \\sum_{x_t} P(X_{t+1} | x_t)\\, B(x_t)"
                "\\\\[6pt]"
                "\\text{Compact: }\\; B'(X') = \\sum_x P(X'|x)\\, B(x)"
            ),
        },
        {
            "title": "Observation Update",
            "text": (
                "After receiving new evidence eₜ₊₁, we reweight the predicted "
                "belief by the likelihood of the observation. We must renormalise "
                "(unlike the passage-of-time step). Entropy decreases."
            ),
            "latex": (
                "B(X_{t+1}) \\propto_{X_{t+1}} P(e_{t+1} | X_{t+1})\\, B'(X_{t+1})"
                "\\\\[6pt]"
                "\\text{Full recursion: }"
                "\\;B(X_{t+1}) = \\alpha\\, P(e_{t+1}|X_{t+1}) "
                "\\sum_{x_t} P(X_{t+1}|x_t)\\, B(x_t)"
            ),
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
