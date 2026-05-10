from flask import Flask, render_template, jsonify, request
from optimization.optimization_problem import Optimization_Problem

app = Flask(__name__)

# We compute the full 100x100 utility grid once and cache it here.
# It's 10,000 calls but runs in under a second — no need to redo it on every request.
_heatmap_cache = None


def compute_heatmap():
    global _heatmap_cache
    if _heatmap_cache is not None:
        return _heatmap_cache

    # Build a 2D grid: rows = wait_duration (1..100), cols = toast_duration (1..100)
    # Plotly heatmap expects z[row][col], so z[0] = wait=1, z[99] = wait=100
    grid = []
    for w in range(1, 101):
        row = []
        for t in range(1, 101):
            row.append(round(Optimization_Problem.utility(toast_duration=t, wait_duration=w), 3))
        grid.append(row)

    _heatmap_cache = grid
    return grid


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/heatmap')
def heatmap():
    """Return the precomputed 100×100 utility grid."""
    return jsonify(compute_heatmap())


@app.route('/api/run', methods=['POST'])
def run():
    """Run one of the three algorithms and return convergence + path data."""
    algo = request.json.get('algorithm')
    runners = {
        'exhaustive_search': _run_exhaustive,
        'hill_climbing':     _run_hill_climbing,
        'gradient_ascent':   _run_gradient_ascent,
    }
    if algo not in runners:
        return jsonify({'error': 'unknown algorithm'}), 400
    return jsonify(runners[algo]())


# ─── Instrumented algorithm runners ──────────────────────────────────────────
# These mirror the logic in optimization_algorithms.py exactly, but record
# each step so the frontend can animate or chart the search process.

def _run_exhaustive():
    """
    Try every (toast, wait, power) combo.
    power is discretized to 21 values (0.0, 0.1 … 2.0) → 100×100×21 = 210,000 calls.
    We only record a convergence point when we find a NEW best score so the
    chart doesn't become 210,000 near-identical dots.
    """
    convergence = []
    path_t = []
    path_w = []

    best = float('-inf')
    best_sol = None
    call = 0

    power_values = [round(p * 0.1, 1) for p in range(21)]

    for t in range(1, 101):
        for w in range(1, 101):
            for power in power_values:
                call += 1
                u = Optimization_Problem.utility(toast_duration=t, wait_duration=w, power=power)

                if u > best:
                    best = u
                    best_sol = (t, w, power)
                    convergence.append({'x': call, 'y': round(best, 4)})
                    path_t.append(t)
                    path_w.append(w)

    convergence.append({'x': call, 'y': round(best, 4)})

    return {
        'algorithm': 'exhaustive_search',
        'result': {
            'toast': best_sol[0],
            'wait':  best_sol[1],
            'power': best_sol[2],
            'utility': round(best, 4),
        },
        'convergence': convergence,
        'path': {'toast': path_t, 'wait': path_w},
        'total_calls': call,
    }


def _run_hill_climbing():
    """
    Start at (50, 50, 1.0) and step to whichever of 6 neighbors scores higher.
    Neighbors: toast±1, wait±1, power±0.1 — all three variables are now optimized.
    """
    toast, wait = 50, 50
    power = 1.0
    best = Optimization_Problem.utility(toast_duration=toast, wait_duration=wait, power=power)
    step = 0

    path_t = [toast]
    path_w = [wait]
    convergence = [{'x': step, 'y': round(best, 4)}]

    POWER_STEP = 0.1

    while True:
        best_neighbor = None
        best_neighbor_u = best

        for dt, dw, dp in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-POWER_STEP),(0,0,POWER_STEP)]:
            nt  = toast + dt
            nw  = wait  + dw
            np_ = round(power + dp, 10)
            if 1 <= nt <= 100 and 1 <= nw <= 100 and 0.0 <= np_ <= 2.0:
                u = Optimization_Problem.utility(toast_duration=nt, wait_duration=nw, power=np_)
                if u > best_neighbor_u:
                    best_neighbor_u = u
                    best_neighbor = (nt, nw, np_)

        if best_neighbor is None:
            break

        toast, wait, power = best_neighbor
        best = best_neighbor_u
        step += 1

        path_t.append(toast)
        path_w.append(wait)
        convergence.append({'x': step, 'y': round(best, 4)})

    return {
        'algorithm': 'hill_climbing',
        'result': {'toast': toast, 'wait': wait, 'power': round(power, 4), 'utility': round(best, 4)},
        'convergence': convergence,
        'path': {'toast': path_t, 'wait': path_w},
        'total_calls': step * 6 + 1,
    }


def _run_gradient_ascent():
    """
    Same discrete hill-climbing for (toast, wait), PLUS a gradient step for `power`.
    The gradient is estimated numerically: slope ≈ (U(p+h) − U(p−h)) / 2h
    """
    toast, wait = 50, 50
    power = 1.0
    best = Optimization_Problem.utility(toast_duration=toast, wait_duration=wait, power=power)
    step = 0

    path_t  = [toast]
    path_w  = [wait]
    convergence = [{'x': step, 'y': round(best, 4)}]

    lr = 0.1    # learning rate — size of each power nudge
    h  = 1e-5   # tiny delta used to estimate the slope of utility w.r.t. power

    while True:
        improved = False

        # -- Part 1: hill-climb on (toast, wait), keeping power fixed --
        best_discrete = None
        best_discrete_u = best
        for dt, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nt, nw = toast + dt, wait + dw
            if 1 <= nt <= 100 and 1 <= nw <= 100:
                u = Optimization_Problem.utility(toast_duration=nt, wait_duration=nw, power=power)
                if u > best_discrete_u:
                    best_discrete_u = u
                    best_discrete = (nt, nw)

        if best_discrete is not None:
            toast, wait = best_discrete
            best = best_discrete_u
            improved = True

        # -- Part 2: gradient step on power --
        p_hi = min(power + h, 2.0)
        p_lo = max(power - h, 0.0)
        grad = (
            Optimization_Problem.utility(toast_duration=toast, wait_duration=wait, power=p_hi) -
            Optimization_Problem.utility(toast_duration=toast, wait_duration=wait, power=p_lo)
        ) / (p_hi - p_lo)

        new_power   = max(0.0, min(2.0, power + lr * grad))
        new_utility = Optimization_Problem.utility(toast_duration=toast, wait_duration=wait, power=new_power)

        if new_utility > best:
            power = new_power
            best  = new_utility
            improved = True

        step += 1
        path_t.append(toast)
        path_w.append(wait)
        convergence.append({'x': step, 'y': round(best, 4)})

        if not improved:
            break

    return {
        'algorithm': 'gradient_ascent',
        'result': {
            'toast': toast,
            'wait':  wait,
            'power': round(power, 4),
            'utility': round(best, 4),
        },
        'convergence': convergence,
        'path': {'toast': path_t, 'wait': path_w},
        'total_calls': step * 6 + 1,   # approx: 4 discrete + 2 power probes per step
    }


if __name__ == '__main__':
    print('\n  Toast Optimizer running at http://localhost:5050\n')
    app.run(debug=True, port=5050)
