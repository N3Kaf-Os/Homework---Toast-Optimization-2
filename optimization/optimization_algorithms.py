

from optimization.optimization_problem import Optimization_Problem


class Optimization_Algorithms():
    """
        Optimization class with different optimization method implementations.

        All three methods treat Optimization_Problem.utility() as a black box,
        meaning they only query it with candidate inputs.
    """

    @classmethod
    def exhaustive_search(cls):
        """
            Find the best (toast_duration, wait_duration) by evaluating every
            possible combination in the 100x100 discrete search space (10 000 calls).

            Guaranteed to find the global optimum for the two integer variables,
            but does not optimize the continuous `power` parameter (left at default 1.0).
        """
        best_utility = float('-inf')
        best_solution = None

        # Outer + inner loops cover every integer point in [1,100] x [1,100]
        for toast_duration in range(1, 101):
            for wait_duration in range(1, 101):
                utility = Optimization_Problem.utility(
                    toast_duration=toast_duration,
                    wait_duration=wait_duration)

                if utility > best_utility:
                    best_utility = utility
                    best_solution = (toast_duration, wait_duration)

        return best_solution, best_utility


    @classmethod
    def hill_climbing(cls):
        """
            Find the best (toast_duration, wait_duration) using steepest-ascent
            hill climbing on the discrete integer grid.

            Starting from the midpoint (50, 50), the algorithm repeatedly moves
            to whichever axis-aligned neighbor improves utility most, stopping
            when no neighbor is better (local maximum reached).

        """
        # Midpoint is a neutral, valid starting position that avoids boundary bias
        toast, wait = 50, 50
        best_utility = Optimization_Problem.utility(
            toast_duration=toast, wait_duration=wait)

        while True:
            # Generate the four cardinal neighbors, clipped to the valid domain [1,100]
            neighbors = []
            for dt, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nt, nw = toast + dt, wait + dw
                if 1 <= nt <= 100 and 1 <= nw <= 100:
                    neighbors.append((nt, nw))

            # Select the neighbor with the highest utility (steepest ascent)
            best_neighbor = None
            best_neighbor_utility = best_utility     # threshold: must strictly improve
            for nt, nw in neighbors:
                u = Optimization_Problem.utility(
                    toast_duration=nt, wait_duration=nw)
                if u > best_neighbor_utility:
                    best_neighbor_utility = u
                    best_neighbor = (nt, nw)

            # No neighbor improved utility → we are at a local (or global) maximum
            if best_neighbor is None:
                return (toast, wait), best_utility

            # Step to the best neighbor and repeat
            toast, wait = best_neighbor
            best_utility = best_neighbor_utility

    @classmethod
    def gradient_ascent(cls):
        """
            Find the best (toast_duration, wait_duration, power) using a hybrid
            of hill climbing (for the discrete integer variables) and gradient
            ascent (for the continuous power variable).

            Each iteration alternates between:
              1. A hill-climbing step : on (toast, wait) => checks 4 discrete neighbors.
              2. A gradient ascent step : on power estimates then nudges power in the uphill direction.

            Convergence: stops when neither step produced an improvement in a full
            iteration => we have reached a joint local maximum in all three variables.
        """
        # Start at the midpoint of each variable's valid range
        toast, wait = 50, 50
        power = 1.0
        best_utility = Optimization_Problem.utility(
            toast_duration=toast, wait_duration=wait, power=power)

        lr = 0.1    # learning rate: how far to step along the power gradient each iteration
        h = 1e-5    # finite-difference for numerically estimating the gradient

        while True:
            improved = False  # tracks whether either sub-step made progress this iteration

            # --- Step 1: Hill climbing on discrete variables (toast, wait) ---
            # Check all four axis-aligned integer neighbors, keeping power fixed
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

            # --- Step 2: Gradient ascent on continuous variable (power) ---
            # Central-difference approximation: grad ≈ (U(p+h) - U(p-h)) / (2h)
            # Clamp evaluation points to keep power inside [0.0, 2.0]
            p_hi = min(power + h, 2.0)
            p_lo = max(power - h, 0.0)
            grad = (Optimization_Problem.utility(
                        toast_duration=toast, wait_duration=wait, power=p_hi) -
                    Optimization_Problem.utility(
                        toast_duration=toast, wait_duration=wait, power=p_lo)) / (p_hi - p_lo)

            # Take a gradient step and clamp the result to the valid power range
            new_power = max(0.0, min(2.0, power + lr * grad))
            new_utility = Optimization_Problem.utility(
                toast_duration=toast, wait_duration=wait, power=new_power)

            if new_utility > best_utility:
                power = new_power
                best_utility = new_utility
                improved = True

            # If neither step improved utility, we have converged to a local maximum
            if not improved:
                break

        return (toast, wait, power), best_utility


