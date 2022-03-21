import multiprocessing as mp
from copy import deepcopy
from typing import List, Optional

from gurobipy import GRB

from core import LNS, Compartment, Item, Problem, Solution, Vehicle


def generate_initial_solutions(
    problem: Problem,
    path: List[int],
    stacks: List[List[Optional[int]]],
    stack_assignment: List[List[Optional[int]]],
):
    N, V, P, D, C, M = problem.problem_vars()
    items = problem.items
    stack_capacities = [c.capacity for c in problem.vehicle.compartments]

    # print(path)

    if len(path) == len(V):
        yield list(path), deepcopy(stack_assignment)

    else:

        curr = path[-1] if path else 0

        stack_tops = {s[-1] for s in stacks if s}
        next_possible = {i for i in P if i not in path}
        next_possible |= {pick_vertex + N for pick_vertex in stack_tops}

        next_possible = list(next_possible)
        next_possible = sorted(next_possible, key=lambda v: C[curr][v])

        collected_demands = [sum(items[i].length for i in stack) for stack in stacks]
        for v in next_possible:
            is_pickup = v <= N

            if is_pickup:
                v_demand = items[v].length
            else:
                v_demand = -items[v - N].length

            for i in M:
                if is_pickup and collected_demands[i] + v_demand <= stack_capacities[i]:
                    path.append(v)
                    stacks[i].append(v)
                    stack_assignment[i].append(v)
                    yield from generate_initial_solutions(
                        problem, path, stacks, stack_assignment
                    )
                    # undo step
                    path.pop()
                    stacks[i].pop()
                    stack_assignment[i].pop()

                if not is_pickup and stacks[i]:
                    pick_v = v - N
                    if pick_v == stacks[i][-1]:
                        path.append(v)
                        stacks[i].pop()
                        yield from generate_initial_solutions(
                            problem, path, stacks, stack_assignment
                        )
                        # undo step
                        path.pop()
                        stacks[i].append(pick_v)


def test():
    nb_items = 15
    nb_compartments = 5
    items = [Item(400) for _ in range(nb_items)]
    vehicle = Vehicle(compartments=[Compartment(800) for _ in range(nb_compartments)])

    # Define the problem
    pdp = Problem(items, vehicle=vehicle)
    pdp.create_model()
    pdp.apply_constraints()
    pdp.set_model_objective()

    gen = generate_initial_solutions(
        pdp,
        [0],
        [[] for i in range(nb_compartments)],
        [[] for i in range(nb_compartments)],
    )

    initial_solutions = []
    for i in range(6):
        path, stack_assignment = next(gen)
        initial_solutions.append(Solution(pdp.items, path, stack_assignment))

    processes = []
    for initial_solution in initial_solutions:
        # start a new LNS process with the new solution
        lns_timelimit = 15
        lns_solver = LNS(
            pdp,
            max_iteration=1000,
            time_limit=lns_timelimit,
            initial_solution=initial_solution,
        )
        p = mp.Process(target=lns_solver.search_local_solution)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    test()
