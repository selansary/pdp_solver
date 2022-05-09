#!/usr/bin/env python3
import random
import time

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

    time_limits = [5, 10]
    # nb_processes = [1, 2, 8, 16, 32, 64]
    nb_processes = [1, 2, 8]
    nb_items = [15]

    results = []
    for time_limit in time_limits:
        for nb_requests in nb_items:
            # create the problem
            pdp = create_pdp(nb_requests=nb_requests)

            # deterministically find the initial greedy solution
            solver = GreedySolver(pdp)
            s_time = time.time()
            initial_solution = solver.solve(nb_solutions=1)[0]
            presolve_time = time.time() - s_time
            initial_objective = pdp.evaluate_solution(initial_solution)

            # apply PLNS with different process numbers
            for nb_process in nb_processes:
                solver = PLNS(
                    pdp,
                    initial_solution,
                    time_limit=time_limit,
                    nb_processes=nb_process,
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
                    plns_time_limit=time_limit,
                    presolve_time="{:.4f}".format(presolve_time),
                    initial_objective=initial_objective,
                    plns_actual_time="{:.2f}".format(total_time),
                    total_time="{:.2f}".format(presolve_time + total_time),
                    nb_process=nb_process,
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


if __name__ == "__main__":
    run()
