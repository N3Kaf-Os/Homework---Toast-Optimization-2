<h1 align=center><strong>Toast Planning Homework</strong></h1>

This exercise lets you practice writing simple optimization algorithms. The optimization problem solved is a toy example for making a slice of toast.

You can find the implementation of the optimization problem in *optimization/optimization_problem.py*. While you can find the used functions there, you should not use knowledge of these functions to implement your optimization algorithms. 


# Setup
This project does not require additional dependencies beyond what's included in a standard Python installation.

# How to run?
You start the program by starting main.py

This script will compare all implemented optimization functions based on different start states. The comparison will be based on:
- The achieved utility
- How long the algoirthm ran



# Your Task

Your task is to implement the unimplemented functions in *optimization/optimization_algorithms.py*
This file should contain different implementations of optimization functions functions that will be compared. An exhaustive search algorthms is already implemented. Your task is to:
1) Implement hill-climbing for the parameters *toast_duration* and *wait_duration*
2) Implement gradient-ascent for the parameters *toast_duration*,  *wait_duration* and *power*

You can find some information on hill climbing and gradient ascent in the documentation folder.

Please note that 2) is designed to be challenging. This task has two discrete and one continuous parameter and will require some  kind of combination of the concepts from hill climbing and gradient ascent. How exactly that is achieved is up to you. There are multiple valid options. 

While implementing Gradient Ascent you can estimate the gradient by comparing the function values of points that are close by. You should avoid calculating the gradient of the function.


# Relation to Assessments

Solving this task can be used as basis for showing specific knowledge in Planning for an assessment in module *SE_14 Artificial Intelligence Basics*.