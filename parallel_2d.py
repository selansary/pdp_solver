from copy import deepcopy
import multiprocessing as mp
import timeit
import time
import random

from gurobipy import GRB
from prettytable import PrettyTable

from core import LNS, PLNS, Compartment, Item, Problem, Solution, Vehicle
from core import TwoDCompartment, TwoDItem, TwoDLNS, TwoDProblem, TwoDPLNS
import plotly.express as px
import pandas as pd

POSSIBLE_ITEMS = [
        TwoDItem(300, 400),
        TwoDItem(600, 300),
        TwoDItem(600, 400),
        TwoDItem(300, 200),
    ]


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
            solution = Solution(pdp.items, order, stack_assignment, pdp.vehicle)
            solution_obj = pdp.evaluate_solution(solution)
            if not best_solution or solution_obj < best_obj:
                best_solution = solution
                best_obj = solution_obj

            solution_delta = model.cbGet(GRB.Callback.RUNTIME)

            # prepare a new LNS with the new solution
            lns_timelimit = total_limit - solution_delta
            solver = TwoDLNS(pdp, max_iteration=1000, time_limit=lns_timelimit, initial_solution=solution)
            lns_solvers.append(solver)


    best_solution = None
    best_obj = None


    random.seed(0)
    items = random.choices(POSSIBLE_ITEMS, k=nb_items)


    # Define the problem
    pdp = TwoDProblem(items)
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

    total_limits = [5, 10, 15]
    # total_limits = [3]

    possible_time_limits = []
    for tl in total_limits:
        possible_time_limits.extend([(i, tl - i) for i in range(1, tl + 1)])

    # possible_time_limits = [(1, 30), (15, 15), (30, 0)]
    possible_time_limits = []
    for t in total_limits:
        possible_time_limits.extend([(t, 0), (0.75 * t, 0.25 * t), (0.5 * t, 0.5 * t), (0.25 * t, 0.75 * t)])

    results = []
    for nb_items in possible_nb_items:
        for nb_compartments in possible_nb_compartments:
            for presolve_time, lns_time in possible_time_limits:
                random.seed(0)
                items = random.choices(POSSIBLE_ITEMS, k=nb_items)

                # Define the problem
                pdp = TwoDProblem(items)
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
                solver = TwoDPLNS(pdp, max_iteration=1000, time_limit=lns_time)
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
                    total_time="{:.2f}".format(presolve_time + lns_time) + "secs",
                    lns_time_percent="{:.2f}".format(lns_time/(presolve_time + lns_time)),
                    presolve_time=presolve_time,
                    presolve_delta="{:.2f}".format(presolve_delta),
                    lns_time=lns_time,
                    lns_delta="{:.2f}".format(lns_delta),
                    presolve_obj=presolve_obj,
                    lns_obj=lns_obj if lns_time else None,
                    objective=lns_obj if lns_obj else presolve_obj,
                )
                results.append(result)

    def split_dict(res_dict):
        lns_res = dict(res_dict)
        lns_res["solver"] = "LNS"
        lns_res["objective"] = res_dict["objective"] if lns_res["lns_time"] else None

        pre_res = dict(res_dict)
        pre_res["objective"] = res_dict["presolve_obj"]
        pre_res["solver"] = "Exact"
        return lns_res, pre_res


    df_list = []
    for res in results:
        lns_dict, pre_dict = split_dict(res)
        df_list.append(lns_dict)
        df_list.append(pre_dict)

    df = pd.DataFrame(df_list)
    # print(df)
    fig = px.bar(
        df,
        x="lns_time_percent",
        y="objective",
        color="solver",
        facet_col="nb_items",
        facet_row="total_time",
        barmode="group",
        title="Objective value with different total time allocation between Exact Solver and LNS",
        color_discrete_map={
        'Exact': '#005293',
        'LNS': '#64a0c8',
        },
        labels={
                     "objective": "Objective",
                     "nb_items": "Number of Items",
                     "total_time": "Time",
                     "lns_time_percent": "Percentage of Total Time Allocated to LNS",
                 },
        category_orders={
            # "lns_time_percent": {},

        }
    )
#     fig.for_each_trace(
#     lambda trace: trace.update(marker_color="lightsalmon") if trace.name == "LNS" else (),
# )

    # fig.update_traces(marker_color=["indianred", "lightsalmon"])
    # fig.update_layout(title_text="Objective value with different Time Allocation between Exact Solver and LNS")
    # fig.update_traces(width=0.05)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        font_size=10,
    )
    fig.show()

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # marks = [0, 0.25, 0.5, 0.75]
    # fig.add_trace(go.Bar(
    #     x=[0, 0.25, 0.5, 0.75],
    #     y=[20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17],
    #     name='Primary Product',
    #     marker_color='indianred'
    # ))
    # fig.add_trace(go.Bar(
    #     x=[0, 0.25, 0.5, 0.75],
    #     y=[19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16],
    #     name='Secondary Product',
    #     marker_color='lightsalmon'
    # ))

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
