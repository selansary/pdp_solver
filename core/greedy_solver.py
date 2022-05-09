from copy import deepcopy
from typing import List, Optional

from core import Problem, Solution


def generate_greedy_solutions(
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
                    yield from generate_greedy_solutions(
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
                        yield from generate_greedy_solutions(
                            problem, path, stacks, stack_assignment
                        )
                        # undo step
                        path.pop()
                        stacks[i].append(pick_v)


class GreedySolver:
    """Greedy Heuristics for solving PDPMS problems."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def solve(self, nb_solutions: int = 1) -> List[Solution]:
        """
        Greedily solve the problem by backtracking. The number of solutions are a
        max of nb_solutions or maximum feasible solutions that could be found.

        The backtracking algorithm generates the solutions by permutation order, so
        the consecutive solutions are similar.
        """
        nb_compartments = len(self.problem.vehicle.compartments)
        solution_generator = generate_greedy_solutions(
            self.problem,
            path=[0],
            stacks=[[] for i in range(nb_compartments)],
            stack_assignment=[[] for i in range(nb_compartments)],
        )

        solutions: List[Solution] = []
        for i in range(nb_solutions):
            try:
                path, stack_assignment = next(solution_generator)
            except StopIteration:
                return solutions
            else:
                solutions.append(
                    Solution(
                        self.problem.items, path, stack_assignment, self.problem.vehicle
                    )
                )

        return solutions
