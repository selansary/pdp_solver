import multiprocessing as mp
import time
from copy import deepcopy
from typing import List, Optional

from core import LNS, Compartment, Item, Problem, Solution, Vehicle


def generate_initial_solutions(
    problem: Problem,
    path: List[int],
    # current state of the stacks as a mapping of stored item vertices
    stacks: List[List[Optional[int]]],
    # assignment as a partial solution
    stack_assignment: List[List[Optional[int]]],
):
    N, V, P, D, C, M = problem.problem_data()
    items = problem.items
    compartments = problem.vehicle.compartments

    if len(path) == len(V):
        yield list(path), deepcopy(stack_assignment)

    else:

        curr = path[-1] if path else 0

        stack_tops = {s[-1] for s in stacks if s}
        next_possible = {i for i in P if i not in path}
        next_possible |= {pick_vertex + N for pick_vertex in stack_tops}

        next_possible = list(next_possible)
        next_possible = sorted(next_possible, key=lambda v: C[curr][v])

        collected_demands = [
            sum(compartments[idx].demand_for_item(items[i]) for i in stack)
            for idx, stack in enumerate(stacks)
        ]
        for v in next_possible:
            is_pickup = v <= N
            pick_v = v if is_pickup else v - N
            item = items[pick_v]

            compatible_stacks = [
                (i, comp)
                for i, comp in enumerate(compartments)
                if comp.is_item_compatible(item)
            ]
            for i, comp in compatible_stacks:
                v_demand = comp.demand_for_item(item)

                if not is_pickup:
                    # delivery demand
                    v_demand = -v_demand

                if is_pickup and collected_demands[i] + v_demand <= comp.capacity:
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
                    # delivery vertex, remove item mapping from top of stack
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
    # High level Parallelism (Multiple LNSs running in parallel, but no)
    nb_items = 15
    nb_compartments = 5
    items = [Item(400) for _ in range(nb_items)]
    vehicle = Vehicle(compartments=[Compartment(800) for _ in range(nb_compartments)])

    # Define the problem
    pdp = Problem(items, vehicle=vehicle)

    gen = generate_initial_solutions(
        pdp,
        [0],
        [[] for i in range(nb_compartments)],
        [[] for i in range(nb_compartments)],
    )

    initial_solutions = []
    for i in range(mp.cpu_count()):
        path, stack_assignment = next(gen)
        initial_solutions.append(Solution(pdp.items, path, stack_assignment, vehicle))
    initial_results = [pdp.evaluate_solution(sol) for sol in initial_solutions]

    def search(solver, qu):
        solution = solver.search()
        qu.put(pdp.evaluate_solution(solution))

    s_time = time.time()
    q = mp.Queue()
    processes = []
    for initial_solution in initial_solutions:
        # start a new LNS process with the new solution
        lns_timelimit = 15
        lns_solver = LNS(
            pdp,
            max_iterations=1000,
            time_limit=lns_timelimit,
            initial_solution=initial_solution,
        )
        p = mp.Process(target=search, args=(lns_solver, q))
        processes.append(p)
        p.start()

    results = [q.get() for p in processes]
    for p in processes:
        p.join()

    print(f"Inital solutions objective {initial_results}")
    print(f"Objectives of different LNSs {results} in {time.time() - s_time}")


if __name__ == "__main__":
    test()
