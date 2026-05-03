
import math


class Optimization_Problem():
    """
        This class represents the optimization problem.
    """

    @classmethod
    def utility(cls, toast_duration, wait_duration, power = 1.0, toaster = 1):
        """
            This is the function you are expected to optimize. 

            While you can look at this function out of interest, your optimization algorithms should assume you don't know this function.
            This means you shouldn't build the first derivation of this function directly.

            Parameters:
            - toast_duration: duration of toasting in seconds. An integer between 1 and 100
            - wait_duration: duration of waiting after toasting in seconds. An integer between 1 and 100
            - toaster: the number of the toaster you want to use. An integer, between 1 and 10.
            - power: how much power the toaster has (A floating point number between 0 and 2)
        """
        # handle input errors
        if (not type(toast_duration) is int) or not (1 <= toast_duration <= 100):
            raise ValueError("toast_duration is not an integer or is not in a valid range")
        if (not type(wait_duration) is int) or not (1 <= wait_duration <= 100):
            raise ValueError("wait_duration is not an integer or is not in a valid range")
        if (not type(toaster) is int) or not (1 <= toaster <= 10):
            raise ValueError("toaster is not an integer or is not in a valid range")
        if (not type(power) is float) or not (0.0 <= power <= 2.0):
            raise ValueError("power is not a float or not in the valid range")

        # get toaster specific configuration
        hpt = [10,8,15,7,9,2,9,19,92,32][toaster-1]
        hpw = [1,4,19,3,20,3,1,4,1,62][toaster-1]
        toaster_utility = [1,0.9,0.7,1.3,0.3,0.8,0.5,0.8,3,0.2][toaster-1]

        # calculate values
        toast_utility = -0.1*(toast_duration-hpt)**2+1
        wait_utility = -0.01*(wait_duration-hpw)**2+1
        overall_utility = (toast_utility + wait_utility) * toaster_utility

        # apply modifier based on electricity
        power_factor = math.sin(10*power+math.pi/2 -10) + power*0.2
        overall_utility *= power_factor

        return overall_utility
