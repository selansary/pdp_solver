#!/usr/bin/env python3
import time
import timeit

from gurobipy import GRB
from prettytable import PrettyTable

from core import LNS, Compartment, Item, Problem, Vehicle, PLNS


def print_solution(optimized_model):
    for var in optimized_model.getVars():
        if abs(var.x) > 1e-6 or var.vtype != GRB.BINARY:
            print("{0}: {1}".format(var.varName, var.x))
    print("Total cost: {0}".format(optimized_model.objVal))
    return None


def main():

    # possible_nb_items = [7, 10, 12, 15, 18, 20]
    # possible_nb_compartments = [3, 5, 6]
    # total_limits = [3, 5, 7, 10]

    possible_nb_items = [15]
    possible_nb_compartments = [5]

    total_limits = [3, 5]
    possible_time_limits = []
    for tl in total_limits:
        possible_time_limits.extend([(i, tl - i) for i in range(1, tl + 1)])

    possible_time_limits = [(1, 10)]

    results = []
    for nb_items in possible_nb_items:
        for nb_compartments in possible_nb_compartments:
            for presolve_time, lns_time in possible_time_limits:
                items = [Item(400) for _ in range(nb_items)]
                vehicle = Vehicle(
                    compartments=[Compartment(800) for _ in range(nb_compartments)]
                )

                # Define the problem
                pdp = Problem(items, vehicle=vehicle)
                pdp.create_model()
                pdp.apply_constraints()
                pdp.set_model_objective()

                # Presolve
                s_time = timeit.default_timer()
                pdp.solve(time_limit=presolve_time)
                presolve_delta = timeit.default_timer() - s_time
                sol = pdp.extract_solution()
                presolve_obj = pdp.evaluate_solution(sol)
                # print(sol)
                print(f"Presolve Solution cost: {presolve_obj}")

                # Apply LNS algorithm
                solver = PLNS(pdp, max_iteration=1000, time_limit=lns_time)
                start_time = time.time()
                s_time = timeit.default_timer()
                best_sol = solver.search(sol)
                lns_delta = timeit.default_timer() - s_time
                # print(best_sol)
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


if __name__ == "__main__":
    main()
