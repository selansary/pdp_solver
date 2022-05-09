#!/usr/bin/env python3
import random
import time
import math

from prettytable import PrettyTable

from core import (
    LNS,
    PLNS,
    Compartment,
    DestroyStrategy,
    DestructionDegreeCriterion,
    GreedySolver,
    Item,
    ModelledOneDimensionalProblem,
    ModelledTwoDimensionalProblem,
    Problem,
    RepairStrategy,
    StoppingCriterion,
    TwoDimensionalCompartment,
    TwoDimensionalItem,
    TwoDimensionalProblem,
    Vehicle,
)

POSSIBLE_ITEMS = [
    TwoDimensionalItem(300, 400),
    TwoDimensionalItem(600, 300),
    TwoDimensionalItem(600, 400),
    TwoDimensionalItem(300, 200),
]


def create_soto_vehicle() -> Vehicle:
    return Vehicle(
        compartments=[
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 300),
            TwoDimensionalCompartment(800, 600),
            TwoDimensionalCompartment(800, 600),
        ]
    )


def create_pdp(nb_requests: int = 30):
    # fix seed for items generation
    random.seed(0)
    items = random.choices(POSSIBLE_ITEMS, k=nb_requests)
    vehicle = create_soto_vehicle()
    return TwoDimensionalProblem(items, vehicle)


def run():
    """
    Run the same problem instance with different time limits, with the same
    initial solution but with different number of processes.
    """

    start = time.time()

    time_limits = [10, 30, 60]
    # degrees possible = 1, 2, 3,  4,  5..
    # factorials       = 1, 2, 6, 24, 120
    # nb_processes = [1, 2, 4, 8, 16, 32, 64]
    # 4 and 16 are not very informative bec we have 8 and 32
    nb_processes = [1, 2, 8, 32, 64]
    nb_items = [15, 20, 30, 35]

    results = []
    for nb_requests in nb_items:
        for time_limit in time_limits:
            # create the problem
            pdp = create_pdp(nb_requests=nb_requests)

            # deterministically find the initial greedy solution
            solver = GreedySolver(pdp)
            s_time = time.time()
            initial_solution = solver.solve(nb_solutions=1)[0]
            presolve_time = time.time() - s_time
            initial_objective = pdp.evaluate_solution(initial_solution)

            degree_percent = 0.15
            degree = max(1, int(degree_percent * nb_requests))
            max_processes = math.factorial(degree)

            # apply PLNS with different process numbers
            for nb_process in [nb for nb in nb_processes if nb / 2 <= max_processes]:

                solver = PLNS(
                    pdp,
                    initial_solution,
                    time_limit=time_limit,
                    nb_processes=nb_process,
                    min_destruction_degree=degree_percent,
                    max_destruction_degree=degree_percent,
                )
                # constant destruction degree to have constant time iterations
                solver.set_destruction_degree_criterion(
                    DestructionDegreeCriterion.CONSTANT
                )

                best_sol = solver.search()
                best_obj = solver.best_cached_solution.objective
                best_iteration = solver.best_cached_solution.iteration
                total_iterations = solver.iteration
                total_time = solver.stats["total_time"]

                result = dict(
                    nb_requests=nb_requests,
                    destruction_degree=degree,
                    plns_time_limit=time_limit,
                    nb_process=nb_process,
                    presolve_time="{:.4f}".format(presolve_time),
                    initial_objective=initial_objective,
                    plns_actual_time="{:.2f}".format(total_time),
                    total_time="{:.2f}".format(presolve_time + total_time),
                    plns_objective=best_obj,
                    plns_nb_iterations=total_iterations,
                    plns_best_iteration=best_iteration,
                )
                results.append(result)

    columns = list(results[0].keys())
    table = PrettyTable()

    table.field_names = columns
    for res in results:
        table.add_row([res.get(k) for k in columns])

    print(table)
    print(f"Total experiment time: {'{:.2f}'.format(time.time() - start)}")

if __name__ == "__main__":
    run()
