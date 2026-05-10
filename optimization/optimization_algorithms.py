

from optimization.optimization_problem import Optimization_Problem


class Optimization_Algorithms():
    """
        Optimization class with different optimization method implementations.

        All three methods treat Optimization_Problem.utility() as a black box,
        meaning they only query it with candidate inputs.

        Think of utility() as a "how good is this toast?" score.
        We don't know what's inside it — we just pass in numbers and get back a score.
        Our job is to find the inputs that maximize that score.

        All three methods optimize the same three variables so their results are comparable:
          - toast_duration  (integer, 1–100)
          - wait_duration   (integer, 1–100)
          - power           (float,   0.0–2.0)
    """

    @classmethod
    def exhaustive_search(cls):
        """
            THE BRUTE FORCE APPROACH — try literally every combination.

            We loop over every possible (toast, wait, power) triple and keep the best.
            power is continuous so we can't try infinite values — instead we discretize
            it into 21 evenly-spaced steps: 0.0, 0.1, 0.2 ... 2.0.

            Total calls: 100 × 100 × 21 = 210,000

            Pros: guaranteed to find the global best within the discrete grid we define.
            Cons: always makes 210,000 calls no matter what — no shortcuts.
                  The power resolution (0.1 steps) means we might miss a slightly
                  better value between two grid points, but it's close enough.
        """
        # Start with the worst possible score so any real result beats it
        best_utility = float('-inf')
        best_solution = None

        # Discretize power into 21 evenly-spaced values: 0.0, 0.1, 0.2 ... 2.0
        # round() prevents floating-point drift (e.g. 0.1+0.1+0.1 = 0.30000000000004)
        power_values = [round(p * 0.1, 1) for p in range(21)]

        # Three nested loops — one per variable we're optimizing
        for toast_duration in range(1, 101):
            for wait_duration in range(1, 101):
                for power in power_values:

                    # Ask: "how good is this combination of all three settings?"
                    utility = Optimization_Problem.utility(
                        toast_duration=toast_duration,
                        wait_duration=wait_duration,
                        power=power)

                    # If it beats everything we've seen so far, save it
                    if utility > best_utility:
                        best_utility = utility
                        best_solution = (toast_duration, wait_duration, power)

        # After all 210,000 combinations, return the winner
        return best_solution, best_utility


    @classmethod
    def hill_climbing(cls):
        """
            THE "WALK UPHILL" APPROACH — smarter than brute force, but can get stuck.

            Imagine you're blindfolded on a 3D landscape and want to reach the peak.
            You can only feel the ground right around your feet.
            Strategy: take a step in whichever direction feels higher. Repeat.
            Stop when every direction around you is lower — you're at a peak!

            We now optimize all three variables: toast, wait, AND power.
            At each step we check 6 neighbors:
              - toast ± 1  (keeping wait and power the same)
              - wait  ± 1  (keeping toast and power the same)
              - power ± 0.1 (keeping toast and wait the same)

            We move to whichever of those 6 neighbors scores highest.
            We stop when none of them beat the current position.

            Risk: we might get stuck on a local peak, not the global one.
                  (Hill climbing has no way to "jump" to a different hill.)
        """
        # Start in the middle of each variable's valid range
        toast, wait = 50, 50
        power = 1.0  # midpoint of [0.0, 2.0]
        best_utility = Optimization_Problem.utility(
            toast_duration=toast, wait_duration=wait, power=power)

        POWER_STEP = 0.1  # how much to nudge power up or down each step

        # Keep looping until we decide to stop
        while True:
            # Build the 6 neighbors: dt=toast change, dw=wait change, dp=power change
            neighbors = []
            for dt, dw, dp in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-POWER_STEP), (0,0,POWER_STEP)]:
                nt = toast + dt
                nw = wait  + dw
                # round() prevents floating-point drift when adding 0.1 repeatedly
                np_ = round(power + dp, 10)
                # Only include the neighbor if it's inside all valid ranges
                if 1 <= nt <= 100 and 1 <= nw <= 100 and 0.0 <= np_ <= 2.0:
                    neighbors.append((nt, nw, np_))

            # Find which neighbor has the highest utility (must strictly beat current)
            best_neighbor = None
            best_neighbor_utility = best_utility
            for nt, nw, np_ in neighbors:
                u = Optimization_Problem.utility(
                    toast_duration=nt, wait_duration=nw, power=np_)
                if u > best_neighbor_utility:
                    best_neighbor_utility = u
                    best_neighbor = (nt, nw, np_)

            # No neighbor is better → we're at a local peak, stop
            if best_neighbor is None:
                return (toast, wait, power), best_utility

            # Otherwise, move to the best neighbor and repeat from there
            toast, wait, power = best_neighbor
            best_utility = best_neighbor_utility

    @classmethod
    def gradient_ascent(cls):
        """
            THE "FOLLOW THE SLOPE" APPROACH — like hill climbing, but smarter about power.

            This method also optimizes all three variables, but handles power differently:
            instead of just trying power±0.1 like hill climbing does, it estimates the
            *mathematical slope* of the utility curve at the current power value and takes
            a proportional step in the uphill direction. This is called the "gradient."

            Each loop iteration does two sub-steps (this is called "coordinate descent"):
              1. Hill-climb on (toast, wait) — same as hill_climbing() above.
              2. Gradient step on power — estimate slope, nudge power uphill.
            Then check if either sub-step helped. If neither did, we've converged.

            Note: the two sub-steps happen sequentially, not simultaneously.
            Step 2 uses the (toast, wait) updated by Step 1.
            This means they're slightly "out of phase" per iteration, but that's
            a known property of coordinate descent — it still converges correctly.

            Note on power=0 as an answer: if the algorithm lands at power=0.0,
            that's the hard lower boundary of the search space. The utility function
            may reward even lower power values, but we can't go there — the clamp
            silently stops us. Worth keeping in mind when interpreting the result.
        """
        # Start in the middle of each variable's valid range
        toast, wait = 50, 50
        power = 1.0  # midpoint of [0.0, 2.0]
        best_utility = Optimization_Problem.utility(
            toast_duration=toast, wait_duration=wait, power=power)

        # lr = learning rate: how big a step to take along the gradient each iteration.
        # Too large → overshoot the peak. Too small → takes forever.
        lr = 0.1

        # h = a tiny nudge used to estimate the slope (gradient) numerically.
        # We probe utility at power+h and power-h, then divide by the gap.
        h = 1e-5    # that's 0.00001 — very small

        # Keep looping until neither step makes any progress
        while True:
            improved = False  # tracks whether anything improved this round

            # --- STEP 1: Hill climbing on (toast, wait), keeping power fixed ---
            best_discrete = None
            best_discrete_utility = best_utility
            for dt, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nt, nw = toast + dt, wait + dw
                if 1 <= nt <= 100 and 1 <= nw <= 100:
                    u = Optimization_Problem.utility(
                        toast_duration=nt, wait_duration=nw, power=power)
                    if u > best_discrete_utility:
                        best_discrete_utility = u
                        best_discrete = (nt, nw)

            if best_discrete is not None:
                toast, wait = best_discrete
                best_utility = best_discrete_utility
                improved = True

            # --- STEP 2: Gradient ascent on power, using updated (toast, wait) ---
            # Clamp the probe points so we don't evaluate outside [0.0, 2.0]
            p_hi = min(power + h, 2.0)
            p_lo = max(power - h, 0.0)

            # Central-difference slope estimate:
            # slope ≈ (score just above power  −  score just below power) / tiny gap
            # Positive slope → power should increase. Negative → power should decrease.
            grad = (Optimization_Problem.utility(
                        toast_duration=toast, wait_duration=wait, power=p_hi) -
                    Optimization_Problem.utility(
                        toast_duration=toast, wait_duration=wait, power=p_lo)) / (p_hi - p_lo)

            # Move power in the uphill direction, scaled by the learning rate,
            # then clamp to stay inside [0.0, 2.0]
            new_power = max(0.0, min(2.0, power + lr * grad))
            new_utility = Optimization_Problem.utility(
                toast_duration=toast, wait_duration=wait, power=new_power)

            if new_utility > best_utility:
                power = new_power
                best_utility = new_utility
                improved = True

            # If neither step improved anything, we've converged — stop
            if not improved:
                break

        return (toast, wait, power), best_utility
