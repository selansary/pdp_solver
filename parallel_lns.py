import multiprocessing as mp
import timeit
import time

from gurobipy import GRB
from prettytable import PrettyTable

from core import LNS, PLNS, Compartment, Item, Problem, Solution, Vehicle


# No parallelism, blocking LNS. Not useful.
def lns_in_gurobi_solution_callback():
    nb_items = 15
    nb_compartments = 5
    total_limit = 10
    best_solution = None
    best_obj = None

    def mycallback(model, where):
        nonlocal best_solution
        nonlocal best_obj

        if where == GRB.Callback.MIPSOL:
            u = model.cbGetSolution(pdp.data["u"])
            u_values = [int(u[i]) for i in pdp.V]
            order = [u_values.index(i) for i in range(len(pdp.V))]

            y_vars = pdp.data["y"]
            y = [[model.cbGetSolution(y_vars[i, k]) for k in pdp.M] for i in pdp.V]
            stack_assignment = [
                [i for i in pdp.P if abs(y[i][k]) > 1e-6] for k in pdp.M
            ]
            solution = Solution(pdp.items, order, stack_assignment)
            solution_obj = pdp.evaluate_solution(solution)
            solution_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if not best_solution or solution_obj < best_obj:
                best_solution = solution
                best_obj = solution_obj

                solution_delta = model.cbGet(GRB.Callback.RUNTIME)
                print(f"New gurobi solution found after {solution_delta} seconds")
                model.cbProceed()  # doesn't really work as expected, or at least
                # also doesn't prevent blocking by LNS

                # start a new LNS with the new solution
                lns_timelimit = total_limit - solution_delta
                lns_solver = LNS(pdp, max_iteration=1000, time_limit=lns_timelimit)
                lns_solution = lns_solver.search(solution)
                lns_obj = pdp.evaluate_solution(lns_solution)
                if lns_obj < best_obj:
                    print(f"Better LNS obj after {timeit.default_timer() - s_time}")
                    best_solution = lns_solution
                    best_obj = lns_obj

            # This doesn't work because it cripples the gurobi solver, the grobui solver would
            # not continue to search as long the callback hasn't terminated and since the
            # callback only terminates on the timeout,there is no time left for the gurobi search

            # idea: the callback should call another function with the solution and
            # the other function should spawn a new process that would only do the search
            # for the LNS

    items = [Item(400) for _ in range(nb_items)]
    vehicle = Vehicle(compartments=[Compartment(800) for _ in range(nb_compartments)])

    # Define the problem
    pdp = Problem(items, vehicle=vehicle)
    pdp.create_model()
    pdp.apply_constraints()
    pdp.set_model_objective()

    # Presolve
    s_time = timeit.default_timer()
    pdp.model._vars = pdp.model.getVars()
    pdp.model.Params.TimeLimit = total_limit
    pdp.model.optimize(mycallback)

    print(f"Final solution obj: {best_obj}")


# Parallelism, non-blocking LNS. Not useful.
def lns_process_spawned_in_gurobi_solution_callback():
    nb_items = 15
    nb_compartments = 5
    total_limit = 15
    best_solution = None
    best_obj = None
    processes = []

    def mycallback(model, where):
        nonlocal best_solution
        nonlocal best_obj
        nonlocal processes

        if where == GRB.Callback.MIPSOL:
            u = model.cbGetSolution(pdp.data["u"])
            u_values = [int(u[i]) for i in pdp.V]
            order = [u_values.index(i) for i in range(len(pdp.V))]

            y_vars = pdp.data["y"]
            y = [[model.cbGetSolution(y_vars[i, k]) for k in pdp.M] for i in pdp.V]
            stack_assignment = [
                [i for i in pdp.P if abs(y[i][k]) > 1e-6] for k in pdp.M
            ]
            solution = Solution(pdp.items, order, stack_assignment)
            solution_obj = pdp.evaluate_solution(solution)
            solution_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if not best_solution or solution_obj < best_obj:
                best_solution = solution
                best_obj = solution_obj

            solution_delta = model.cbGet(GRB.Callback.RUNTIME)

            # start a new LNS process with the new solution
            lns_timelimit = total_limit - solution_delta
            lns_solver = LNS(
                pdp,
                max_iteration=1000,
                time_limit=lns_timelimit,
                initial_solution=solution,
            )
            p = mp.Process(target=lns_solver.search_local_solution)
            processes.append(p)
            p.start()

    items = [Item(400) for _ in range(nb_items)]
    vehicle = Vehicle(compartments=[Compartment(800) for _ in range(nb_compartments)])

    # Define the problem
    pdp = Problem(items, vehicle=vehicle)
    pdp.create_model()
    pdp.apply_constraints()
    pdp.set_model_objective()

    # Presolve
    s_time = timeit.default_timer()
    pdp.model._vars = pdp.model.getVars()
    pdp.model.Params.TimeLimit = total_limit
    pdp.model.optimize(mycallback)

    print(f"Final gruobi solution obj: {best_obj}")

    # wait for spawned LNS processes to terminate
    for p in processes:
        p.join()


# Similar to previous, but by saving the gurobi solutions
# and then running a process per solution in parallel
# result not yet retrievable from process, but the objective is printed
def lns_processes_based_on_gurobi_solutions():
    nb_items = 15
    nb_compartments = 5
    total_limit = 15
    best_solution = None
    best_obj = None
    lns_solvers = []

    def mycallback(model, where):
        nonlocal best_solution
        nonlocal best_obj
        nonlocal lns_solvers

        if where == GRB.Callback.MIPSOL:
            u = model.cbGetSolution(pdp.data["u"])
            u_values = [int(u[i]) for i in pdp.V]
            order = [u_values.index(i) for i in range(len(pdp.V))]

            y_vars = pdp.data["y"]
            y = [[model.cbGetSolution(y_vars[i, k]) for k in pdp.M] for i in pdp.V]
            stack_assignment = [[i for i in pdp.P if abs(y[i][k]) > 1e-6] for k in pdp.M]
            solution = Solution(pdp.items, order, stack_assignment)
            solution_obj = pdp.evaluate_solution(solution)
            if not best_solution or solution_obj < best_obj:
                best_solution = solution
                best_obj = solution_obj

            solution_delta = model.cbGet(GRB.Callback.RUNTIME)

            # prepare a new LNS with the new solution
            lns_timelimit = total_limit - solution_delta
            solver = LNS(pdp, max_iteration=1000, time_limit=lns_timelimit, initial_solution=solution)
            lns_solvers.append(solver)


    best_solution = None
    best_obj = None


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
    pdp.model._vars = pdp.model.getVars()
    pdp.model.Params.TimeLimit = total_limit
    pdp.model.optimize(mycallback)

    print(f"Final presolve solution obj: {best_obj}")

    solving_processes = []
    for solver in lns_solvers:
        p = mp.Process(target=solver.search_local_solution)
        p.start()

    for process in solving_processes:
        process.join()


def parallelism_within_lns():
    # possible_nb_items = [7, 10, 12, 15, 18, 20]
    # possible_nb_compartments = [3, 5, 6]
    # total_limits = [3, 5, 7, 10]

    possible_nb_items = [15]
    possible_nb_compartments = [5]

    total_limits = [5]
    possible_time_limits = []
    for tl in total_limits:
        possible_time_limits.extend([(i, tl - i) for i in range(1, tl + 1)])

    # possible_time_limits = [(1, 30), (15, 15), (30, 0)]

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
                pdp.presolve(time_limit=presolve_time)
                presolve_delta = timeit.default_timer() - s_time
                sol = pdp.extract_solution()
                presolve_obj = pdp.evaluate_solution(sol)
                # print(sol)
                print(f"Presolve Solution cost: {presolve_obj}")

                # Apply Parallel LNS algorithm
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


if __name__ == '__main__':
    # lns_process_spawned_in_gurobi_solution_callback()
    # lns_processes_based_on_gurobi_solutions()
    parallelism_within_lns()
