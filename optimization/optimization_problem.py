
import math


class Optimization_Problem():
    """
        This class represents the optimization problem — the "black box" we're trying to maximize.

        Think of it as a fancy toaster simulator. You give it settings,
        it gives back a score telling you how good the toast would be.
        Your optimization algorithms in optimization_algorithms.py try to find
        the settings that produce the highest possible score.
    """

    @classmethod
    def utility(cls, toast_duration, wait_duration, power = 1.0, toaster = 1):
        """
            THE SCORING FUNCTION — this is what we're trying to maximize.

            You pass in toaster settings, it returns a single number (the "utility" = quality score).
            Higher is better. Your job as an optimizer is to find the inputs that make this as high as possible.

            IMPORTANT: the optimization algorithms are supposed to treat this as a black box.
            That means they should NOT look at the math inside here to cheat —
            they should only call it and react to the score it returns.

            Parameters:
            - toast_duration: how long to toast, in seconds (integer 1–100)
            - wait_duration:  how long to wait after toasting, in seconds (integer 1–100)
            - power:          how much electricity the toaster uses (float 0.0–2.0, default 1.0)
            - toaster:        which toaster machine to use (integer 1–10, default 1)
        """
        # --- Input validation: crash early if the caller passed bad values ---
        if (not type(toast_duration) is int) or not (1 <= toast_duration <= 100):
            raise ValueError("toast_duration is not an integer or is not in a valid range")
        if (not type(wait_duration) is int) or not (1 <= wait_duration <= 100):
            raise ValueError("wait_duration is not an integer or is not in a valid range")
        if (not type(toaster) is int) or not (1 <= toaster <= 10):
            raise ValueError("toaster is not an integer or is not in a valid range")
        if (not type(power) is float) or not (0.0 <= power <= 2.0):
            raise ValueError("power is not a float or not in the valid range")

        # --- Each toaster has its own "sweet spot" settings (hidden from the optimizer) ---
        # hpt = "happy point for toast" — the ideal toast_duration for this toaster
        # hpw = "happy point for wait" — the ideal wait_duration for this toaster
        # toaster_utility = a multiplier for how good this toaster is in general
        hpt = [10,8,15,7,9,2,9,19,92,32][toaster-1]
        hpw = [1,4,19,3,20,3,1,4,1,62][toaster-1]
        toaster_utility = [1,0.9,0.7,1.3,0.3,0.8,0.5,0.8,3,0.2][toaster-1]

        # --- Score how close the inputs are to the toaster's sweet spot ---
        # This is an "inverted parabola": score is 1.0 at the sweet spot,
        # and drops off the further you are from it.
        # e.g. if hpt=10 and toast_duration=10 → -0.1*(0)²+1 = 1.0 (perfect)
        #      if hpt=10 and toast_duration=20 → -0.1*(10)²+1 = -9.0 (way off)
        toast_utility = -0.1*(toast_duration-hpt)**2+1
        wait_utility = -0.01*(wait_duration-hpw)**2+1

        # Combine both scores and scale by this toaster's quality multiplier
        overall_utility = (toast_utility + wait_utility) * toaster_utility

        # --- Adjust the score based on power level ---
        # The sin() wave means some power values are better than others — it's not linear!
        # The + power*0.2 adds a slight bonus for using more power.
        # This is why gradient ascent is useful: the power landscape has curves.
        power_factor = math.sin(10*power+math.pi/2 -10) + power*0.2
        overall_utility *= power_factor

        return overall_utility
