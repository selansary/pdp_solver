#!/usr/bin/env python3
import random
import time
import timeit

from gurobipy import GRB
from prettytable import PrettyTable

from core import (
    LNS,
    PLNS,
    Compartment,
    DestroyStrategy,
    DestructionDegreeCriterion,
    Item,
    ModelledOneDimensionalProblem,
    ModelledTwoDimensionalProblem,
    Problem,
    RepairStrategy,
    TwoDimensionalCompartment,
    TwoDimensionalItem,
    TwoDimensionalProblem,
    Vehicle,
)


def print_solution(optimized_model):
    for var in optimized_model.getVars():
        if abs(var.x) > 1e-6 or var.vtype != GRB.BINARY:
            print("{0}: {1}".format(var.varName, var.x))
    print("Total cost: {0}".format(optimized_model.objVal))
    return None


def run_1d():
    possible_nb_items = [15]
    possible_nb_compartments = [5]

    total_limits = [3, 5]
    possible_time_limits = []
    for tl in total_limits:
        possible_time_limits.extend([(i, tl - i) for i in range(1, tl + 1)])

    possible_time_limits = [(1, 3)]

    results = []
    for nb_items in possible_nb_items:
        for nb_compartments in possible_nb_compartments:
            for presolve_time, lns_time in possible_time_limits:
                items = [Item(400) for _ in range(nb_items)]
                vehicle = Vehicle(
                    compartments=[Compartment(800) for _ in range(nb_compartments)]
                )

                # Define the problem
                modelled_pdp = ModelledOneDimensionalProblem(items, vehicle=vehicle)
                modelled_pdp.create_model()
                modelled_pdp.apply_constraints()
                modelled_pdp.set_model_objective()

                # Presolve
                s_time = timeit.default_timer()
                modelled_pdp.solve(time_limit=presolve_time)
                presolve_delta = timeit.default_timer() - s_time
                sol = modelled_pdp.extract_solution()
                presolve_obj = modelled_pdp.evaluate_solution(sol)
                print(sol)
                print(f"Presolve Solution cost: {presolve_obj}")

                # Apply LNS algorithm
                pdp = Problem(items, vehicle, modelled_pdp.C)
                solver = PLNS(pdp, sol, time_limit=lns_time)
                start_time = time.time()
                s_time = timeit.default_timer()
                best_sol = solver.search()
                lns_delta = timeit.default_timer() - s_time
                print(best_sol)
                delta = time.time() - start_time
                lns_obj = pdp.evaluate_solution(best_sol)
                print(f"LNS Solution cost: {lns_obj} in {delta}")

                result = dict(
                    nb_items=nb_items,
                    total_time=presolve_time + lns_time,
                    presolve_time=presolve_time,
                    presolve_delta="{:.2f}".format(presolve_delta),
                    lns_time=lns_time,
                    lns_delta="{:.2f}".format(lns_delta),
                    presolve_obj=presolve_obj,
                    lns_obj=lns_obj if lns_time else None,
                )
                results.append(result)

    columns = list(results[0].keys())
    table = PrettyTable()

    table.field_names = columns
    for res in results:
        table.add_row([res.get(k) for k in columns])

    print(table)


def run_2d():
    possible_items = [
        TwoDimensionalItem(300, 400),
        TwoDimensionalItem(600, 300),
        TwoDimensionalItem(600, 400),
        TwoDimensionalItem(300, 200),
    ]
    possible_nb_items = [15]
    possible_nb_compartments = [5]

    total_limits = [3, 5]
    possible_time_limits = []
    for tl in total_limits:
        possible_time_limits.extend([(i, tl - i) for i in range(1, tl + 1)])

    possible_time_limits = [(500, 0), (1, 500)]
    # possible_time_limits = [(1, 30), (15, 15), (30, 0)]
    # possible_time_limits = [(1, 5), (1, 10), (1, 30), (1, 50), (1, 100), (1, 200)]

    results = []
    for nb_items in possible_nb_items:
        for nb_compartments in possible_nb_compartments:
            for presolve_time, lns_time in possible_time_limits:
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
                modelled_pdp = ModelledTwoDimensionalProblem(items, vehicle)
                modelled_pdp.create_model()
                modelled_pdp.apply_constraints()
                modelled_pdp.set_model_objective()

                # Presolve
                s_time = timeit.default_timer()
                modelled_pdp.solve(time_limit=presolve_time)
                presolve_delta = timeit.default_timer() - s_time
                sol = modelled_pdp.extract_solution()
                presolve_obj = modelled_pdp.evaluate_solution(sol)
                print(f"Presolve Solution cost: {presolve_obj}")

                # Apply LNS without parallelism
                pdp = TwoDimensionalProblem(items, vehicle, modelled_pdp.C)
                solver = LNS(pdp, sol, time_limit=lns_time)
                solver.set_destruction_degree_criterion(
                    DestructionDegreeCriterion.CONSTANT
                )
                lns_sol = solver.search()
                lns_obj = pdp.evaluate_solution(lns_sol)
                lns_best_iteration = solver.stats["best_iteration"]
                lns_nb_iterations = solver.iteration

                # Apply LNS with parallel best effort repair
                solver = LNS(pdp, sol, time_limit=lns_time)
                solver.set_destruction_degree_criterion(
                    DestructionDegreeCriterion.CONSTANT
                )
                solver.set_repair_strategy(RepairStrategy.PARALLEL_OPTIMAL_LEAST_COST)
                best_sol = solver.search()
                parallel_processes_lns_obj = pdp.evaluate_solution(best_sol)
                pp_best_iteration = solver.stats["best_iteration"]
                pp_nb_iterations = solver.iteration

                # Apply Parallel LNS algorithm
                pdp = TwoDimensionalProblem(items, vehicle, modelled_pdp.C)
                solver = PLNS(pdp, sol, time_limit=lns_time)
                solver.set_destruction_degree_criterion(
                    DestructionDegreeCriterion.CONSTANT
                )
                best_sol = solver.search()
                pool_lns_obj = pdp.evaluate_solution(best_sol)
                pool_best_iteration = solver.stats["best_iteration"]
                pool_nb_iterations = solver.iteration

                result = dict(
                    nb_items=nb_items,
                    total_time=presolve_time + lns_time,
                    presolve_time=presolve_time,
                    presolve_obj=presolve_obj,
                    lns_time=lns_time,

                    lns_obj = lns_obj if lns_time else None,
                    lns_nb_iterations=lns_nb_iterations,
                    lns_best_iteration=lns_best_iteration,

                    processes_lns_obj=parallel_processes_lns_obj if lns_time else None,
                    processes_nb_iterations=pp_nb_iterations,
                    processes_best_iteration=pp_best_iteration,

                    pool_lns_obj=pool_lns_obj if lns_time else None,
                    pool_nb_iterations=pool_nb_iterations,
                    pool_best_iteration=pool_best_iteration,
                )
                results.append(result)

    columns = list(results[0].keys())
    table = PrettyTable()

    table.field_names = columns
    for res in results:
        table.add_row([res.get(k) for k in columns])

    print(table)


if __name__ == "__main__":
    # run_1d()
    run_2d()
