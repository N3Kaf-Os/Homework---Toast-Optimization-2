
import time

from optimization.optimization_algorithms import Optimization_Algorithms

def test_one(optimizer_name, optimization_function):
    """
        Tests one specific optimizerprints out the test results.
    """
    print("\nTesting " + optimizer_name)

    start = time.time_ns()
    solution, utility = optimization_function()
    end = time.time_ns()
    time_ns = end - start

    if solution is None:
        print("- not implemented")
    else:
        print("- solution:", solution)
        print("- utility:", utility)
        print("- execution time:",time_ns, "ns")

        
# this is a test function. It tests your plan function 
def test_all():
    """
        tests all planning functions
    """
    test_one("Exhaustive Search", Optimization_Algorithms.exhaustive_search)
    test_one("Hill Climbing", Optimization_Algorithms.hill_climbing)
    test_one("Gradient Ascent", Optimization_Algorithms.gradient_ascent)



if __name__ == "__main__":
    test_all()
