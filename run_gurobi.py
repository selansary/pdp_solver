#!/usr/bin/env python3
import random

from core import (
    LimitlessModelledTwoDimensionalProblem,
    TwoDimensionalCompartment,
    TwoDimensionalItem,
    Vehicle,
)


def run():
    possible_items = [
        TwoDimensionalItem(300, 400),
        TwoDimensionalItem(600, 300),
        TwoDimensionalItem(600, 400),
        TwoDimensionalItem(300, 200),
    ]
    nb_items = 15
    random.seed(0)
    items = random.choices(possible_items, k=nb_items)

    vehicle = Vehicle(
        compartments=[
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 600),
            TwoDimensionalCompartment(800, 600),
        ]
    )

    # Define the problem
    modelled_pdp = LimitlessModelledTwoDimensionalProblem(items, vehicle)
    modelled_pdp.create_model()
    modelled_pdp.apply_constraints()
    modelled_pdp.set_model_objective()
    modelled_pdp.solve()


def run_2_problems():
    possible_items = [
        TwoDimensionalItem(300, 400),
        TwoDimensionalItem(600, 300),
        TwoDimensionalItem(600, 400),
        TwoDimensionalItem(300, 200),
    ]

    vehicle = Vehicle(
        compartments=[
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 600),
            TwoDimensionalCompartment(800, 600),
        ]
    )

    # Define some problems

    # Problem 1
    nb_items_1 = 15
    V_1 = list(range(0, nb_items_1 * 2 + 1))
    random.seed(0)
    items_1 = random.choices(possible_items, k=nb_items_1)
    random.seed(0)
    distance_matrix_1 = [
        [random.randint(1, 100) if i != 0 and j!=0 and i != j else 0 for i in V_1]
        for j in V_1
    ]
    problem_1 = TwoDimensionalProblem(items_1, vehicle, distance_matrix_1, name="prob_1")

    # Problem 2
    nb_items_2 = 15
    V_2 = list(range(0, nb_items_2 * 2 + 1))
    random.seed(100)
    items_2 = random.choices(possible_items, k=nb_items_2)
    random.seed(100)
    distance_matrix_2 = [
        [random.randint(1, 100) if i != 0 and j!=0 and i != j else 0 for i in V_2]
        for j in V_2
    ]
    # problem_2 = TwoDimensionalProblem(items_2, vehicle, distance_matrix_2, name="prob_2")

    # Finding inital solution with gurobi
    # Initial solutions for the problem by running the gurobi solver
    initial_solution_time_limit = 2 # seconds
    # Problem 1
    modelled_problem_1 = LimitlessModelledTwoDimensionalProblem(items_1, vehicle, distance_matrix= problem_1.C)
    modelled_problem_1.create_model()
    modelled_problem_1.apply_constraints()
    modelled_problem_1.set_model_objective()
    modelled_problem_1.solve()
    initial_solution_1 = modelled_problem_1.extract_solution()

    # Problem 2
    modelled_problem_2 = LimitlessModelledTwoDimensionalProblem(items_2, vehicle,  distance_matrix=problem_2.C)
    modelled_problem_2.create_model()
    modelled_problem_2.apply_constraints()
    modelled_problem_2.set_model_objective()
    modelled_problem_2.solve()
    initial_solution_2 = modelled_problem_2.extract_solution()


if __name__ == "__main__":
    run()
