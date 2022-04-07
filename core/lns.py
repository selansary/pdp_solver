import itertools
import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Callable, List, Optional

import numpy as np

from . import (
    OptimizationObjective,
    Problem,
    Solution,
    TwoDimensionalCompartment,
    TwoDimensionalProblem,
    Vehicle,
)

CPU_COUNT = mp.cpu_count()


class AcceptanceCriterion(ABC):
    """Abstract base class for acceptance criteria to be used with LNS."""

    @abstractmethod
    def apply_acceptance_function(
        self,
        old_solution: Solution,
        new_solution: Solution,
        eval_fun: Callable[[Solution], int],
    ) -> bool:
        pass

    @abstractmethod
    def update_acceptance_params(self, *args, **kwargs):
        pass


class SimulatedAnnealing(AcceptanceCriterion):
    """Simulated Annealing as an accpetance criterion for LNS."""

    def __init__(
        self, alpha: float = 0.99975, starting_temperature: float = 1.05
    ) -> None:
        self.alpha = alpha
        self.temperature = starting_temperature

    def apply_acceptance_function(
        self,
        old_solution: Solution,
        new_solution: Solution,
        eval_fun: Callable[[Solution], int],
    ) -> bool:
        return self._apply_acceptance_function(old_solution, new_solution, eval_fun)

    def _apply_acceptance_function(
        self,
        old_solution: Solution,
        new_solution: Solution,
        eval_fun: Callable[[Solution], int],
        update_params: bool = True,
    ) -> bool:
        old_obj = eval_fun(old_solution)
        new_obj = eval_fun(new_solution)
        accept_prob = np.exp(-(new_obj - old_obj) / self.temperature)
        accept = random.choices([True, False], weights=[accept_prob, 1 - accept_prob])[
            0
        ]

        # Unless otherwise specified, prepare the acceptance for next iterartion
        # by reducing the temperature
        if update_params:
            self.update_acceptance_params()

        return accept

    def update_acceptance_params(self):
        self.temperature *= self.alpha


class StoppingCriterion(Flag):
    """Stopping criterion for the LNS."""

    TIMELIMIT = auto()
    MAX_ITERATIONS = auto()
    NO_IMPROVEMENT = auto()

    MAX_ITERATIONS_TIMELIMIT = MAX_ITERATIONS | TIMELIMIT


class DestructionDegreeCriterion(Enum):
    """Degree of Destruction cirterion to be used during LNS."""

    RANDOM = auto()
    """Randomly choose a suitable degree of distruction for each search iteration."""

    CONSTANT = auto()
    """Choose a constant degree of distruction throughout the search."""

    GRADUALLY_INCREASING = auto()
    """Increase the degree of distruction gradually as the search progresses."""

    GRADUALLY_DECREASING = auto()
    """Decrease the degree of distruction gradually as the search progresses."""


class DestroyStrategy(Enum):
    """Strategy to be used for the destroy operator during LNS."""

    RANDOM = auto()


class RepairStrategy(Enum):
    """Strategy to be used for the repair operator during LNS."""


@dataclass
class SearchSolution:
    iteration: int
    solution: Solution
    objective: int
    time_found: float = field(default_factory=time.time)


class LNS:
    """
    Large Neigborhood Search Metaheuristics class.
    """

    def __init__(
        self,
        problem: Problem,
        initial_solution: Solution,
        max_iterations: int = 1000,
        max_without_improvement: int = 10,
        time_limit: float = 1000.0,
        min_destruction_degree: float = 0.15,
        max_destruction_degree: float = 0.35,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        """
        `max_iterations` (number of iterations) is the parameter used for stopping
        criteria: `StoppingCriterion.MAX_ITERATIONS` and `MAX_ITERATIONS_TIMELIMIT`

        `time_limit` (seconds) is the parameter used for stopping criteria:
        `StoppingCriterion.TIMELIMIT` and `MAX_ITERATIONS_TIMELIMIT`

        `max_without_improvement` (number of iterations) is the parameter used for
        stopping criterion: `StoppingCriterion.MAX_ITERATIONS`

        The default stopping criterion of the search is based on the maximum number of
        iterations and time limit provided on initialization. Other stopping criteria
        can be set using `set_stopping_criterion`.

        `min_destruction_degree` (percentage of requests) is a parameter used for
        estimating the degree for destruction every search iteration.
        `max_destruction_degree` (percentage of requests) is a parameter used for
        estimating the degree for destruction every search iteration.

        The default degree of destruction is based on the destruction degreee criterion
        `DestructionDegree.CONSTANT` and is the average of the `min_destruction_degree`
        and the `max_destruction_degree`. Other degree of destruction criteria can be
        set using `set_destruction_degree_criterion`.

        `optimization_objective` is the minimization objective for the solution search
        """
        self.problem = problem
        self.initial_solution = initial_solution

        # Parameters for stopping criterion
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.max_without_improvement = max_without_improvement
        self.stop = self._stop_max_iterations_timelimit

        # Parameters for degree of destruction
        self.min_destruction_degree = min_destruction_degree
        self.max_destruction_degree = max_destruction_degree
        self.degree_of_destruction_strategy = self._get_constant_destruction_degree

        self.optimization_objective = optimization_objective

        # TODO possibly variable acceptance criterion
        self.acceptance_criterion = SimulatedAnnealing()

        # TODO set destroy strategy
        # TODO set insertion strategy

        initial_objective = self.evaluate_solution(initial_solution)
        self.solutions_cache: List[SearchSolution] = [
            SearchSolution(-1, initial_solution, initial_objective)
        ]

        self.start_time: Optional[float] = None
        self.iteration = 0

    def search(self) -> Solution:
        self.iteration = 0
        self.start_time = time.time()

        current_solution = self.initial_solution
        current_objective = self.evaluate_solution(current_solution)

        best_solution = current_solution
        best_objective = current_objective

        while not self.stop():
            # Get the degree of destruction based on the set DestructionDegreeCriterion
            nb_removed = self.get_destruction_degree()

            # Apply the destroy and repair operators to retrieve a new solution
            # in the neighborhood of the current solution
            solution = deepcopy(current_solution)
            solution = self.repair(self.destroy(solution, nb_removed))

            # Cache the found solution
            new_objective = self.evaluate_solution(solution)
            self.solutions_cache.append(
                SearchSolution(self.iteration, solution, new_objective)
            )

            # Check if the objective improved or if the acceptance criterion allows
            # the new unimproved solution (applicable for a minimization problem)
            if new_objective < current_objective or self.accept(
                current_solution, solution
            ):
                current_solution = solution
                current_objective = new_objective
                if new_objective < best_objective:
                    best_solution = solution
                    best_objective = new_objective

            self.iteration += 1

        return best_solution

    def evaluate_solution(self, solution: Solution):
        """
        Evaluate the solution for the problem based on the optimization objective
        specified on initialization.
        """
        return self.problem.evaluate_solution(solution, self.optimization_objective)

    def _stop_max_iterations(self) -> bool:
        return self.iteration > self.max_iterations

    def _stop_timelimit(self) -> bool:
        assert self.start_time
        return time.time() - self.start_time > self.time_limit

    def _stop_max_iterations_timelimit(self) -> bool:
        return self._stop_max_iterations() or self._stop_timelimit()

    def _stop_no_improvement(self) -> bool:
        max_no_impro = self.max_without_improvement
        if len(self.solutions_cache) < max_no_impro:
            return False

        impro_count = 0
        for i in range(1, len(self.solutions_cache) + 1):
            if (
                self.solutions_cache[-i].objective
                < self.solutions_cache[-i - 1].objective
            ):
                # the solution improved wrt previous solution
                impro_count = 1
            else:
                # the solution didn't improve wrt previous solution
                impro_count += 1

            if impro_count >= self.max_without_improvement:
                return False

        return True

    def set_stopping_criterion(self, stopping_criterion: StoppingCriterion):
        """
        Set the stopping criterion of the search. Possible stopping criteria are:
            - `StoppingCriterion.MAX_ITERATIONS`
            - `StoppingCriterion.TIMELIMIT`
            - `StoppingCriterion.MAX_ITERATIONS_TIMELIMIT`
            - `StoppingCriterion.NO_IMPROVEMENT`
        """
        if stopping_criterion == StoppingCriterion.MAX_ITERATIONS:
            self.stop = self._stop_max_iterations
        elif stopping_criterion == StoppingCriterion.TIMELIMIT:
            self.stop = self._stop_timelimit
        elif stopping_criterion == StoppingCriterion.MAX_ITERATIONS_TIMELIMIT:
            self.stop = self._stop_max_iterations_timelimit
        elif stopping_criterion == StoppingCriterion.NO_IMPROVEMENT:
            self.stop = self._stop_no_improvement

    def get_destruction_degree(self) -> int:
        """Get the destruction degree based on the desrtuction degree criterion set."""
        return self.degree_of_destruction_strategy()

    def _get_random_destruction_degree(self) -> int:
        nb_requests = len(self.problem.items)
        min_degree = max(1, int(self.min_destruction_degree * nb_requests))
        max_degree = max(1, int(self.max_destruction_degree * nb_requests))
        return random.randint(min_degree, max_degree)

    def _get_constant_destruction_degree(self) -> int:
        nb_requests = len(self.problem.items)
        min_degree = max(1, int(self.min_destruction_degree * nb_requests))
        max_degree = max(1, int(self.max_destruction_degree * nb_requests))
        return (max_degree + min_degree) // 2

    def _get_increasing_destruction_degree(self) -> int:
        nb_requests = len(self.problem.items)
        min_degree = max(1, int(self.min_destruction_degree * nb_requests))
        max_degree = max(1, int(self.max_destruction_degree * nb_requests))
        return min(max_degree, int(min_degree + ((1.01) ** self.iteration)))

    def _get_decreasing_destruction_degree(self) -> int:
        nb_requests = len(self.problem.items)
        min_degree = max(1, int(self.min_destruction_degree * nb_requests))
        max_degree = max(1, int(self.max_destruction_degree * nb_requests))
        return max(min_degree, int(max_degree * ((0.975) ** self.iteration)))

    def set_destruction_degree_criterion(
        self, destruction_degree_criterion: DestructionDegreeCriterion
    ):
        """
        Set the degree of destruction cirterion of the search. Possible destruction
        degree criteria are:
            - `DestructionDegreeCriterion.RANDOM`
            - `DestructionDegreeCriterion.CONSTANT`
            - `DestructionDegreeCriterion.GRADUALLY_INCREASING`
            - `DestructionDegreeCriterion.GRADUALLY_DECREASING`
        """
        if destruction_degree_criterion == DestructionDegreeCriterion.RANDOM:
            self.degree_of_destruction_strategy = self._get_random_destruction_degree
        elif destruction_degree_criterion == DestructionDegreeCriterion.CONSTANT:
            self.degree_of_destruction_strategy = self._get_constant_destruction_degree
        elif (
            destruction_degree_criterion
            == DestructionDegreeCriterion.GRADUALLY_INCREASING
        ):
            self.degree_of_destruction_strategy = (
                self._get_increasing_destruction_degree
            )
        elif (
            destruction_degree_criterion
            == DestructionDegreeCriterion.GRADUALLY_DECREASING
        ):
            self.degree_of_destruction_strategy = (
                self._get_decreasing_destruction_degree
            )

    def accept(self, old_solution: Solution, new_solution: Solution) -> bool:
        """
        Return whether to accept the new_solution based on the current
        acceptance_criterion of the LNS.
        """
        return self.acceptance_criterion.apply_acceptance_function(
            old_solution, new_solution, self.evaluate_solution
        )

    def destroy(self, solution: Solution, nb_requests_to_remove: int) -> Solution:
        """Return a destroyed copy of the solution."""
        return self.random_destroy(solution, nb_requests_to_remove)

    def repair(self, destroyed_solution: Solution) -> Solution:
        """Return a repaired copy of the destroyed solution."""
        return self.random_least_cost_repair(destroyed_solution)

    def random_destroy(
        self, solution: Solution, nb_requests_to_remove: int
    ) -> Solution:
        N = len(self.problem.items)
        pickup_vertices = self.problem.P
        requests_to_remove = np.random.choice(
            pickup_vertices, size=nb_requests_to_remove, replace=False
        )

        def removable_vertex(v: int) -> bool:
            pick_vertex = v if v <= N else v - N
            return pick_vertex in requests_to_remove

        partial_order = [v for v in solution.order if not removable_vertex(v)]
        partial_stack_assignment = [
            [v for v in assign if not removable_vertex(v)]
            for assign in solution.stack_assignment
        ]

        return Solution(
            self.problem.items,
            partial_order,
            partial_stack_assignment,
            self.problem.vehicle,
            is_partial=True,
        )

    def optimal_sequential_least_cost_repair(
        self, destroyed_solution: Solution
    ) -> Solution:
        """Brute force for an optimally-repaired solution. Single process repair.

        Note: Too slow, a single iteration may exceed time limit for the search.
        For optimal repair, use the multiple-process repair instead.
        """
        removed_pickups = [
            v for v in self.problem.P if v not in destroyed_solution.order
        ]

        # Try all possible insertion orders for the removed vertices
        # and choose the order that results in the least cost solution
        insertion_orders = permute(removed_pickups)
        best_solution = None
        best_total_cost = None
        for insertion_order in insertion_orders:
            repair_operator = SingleOrderLeastCostRepairOperator(
                self.problem,
                destroyed_solution,
                insertion_order,
                self.optimization_objective,
            )
            solution = repair_operator.repair()

            # Check if this insertion order contributes to the lowest cost total order
            total_cost = self.evaluate_solution(solution)
            if best_solution is None or best_total_cost > total_cost:
                best_total_cost = total_cost
                best_solution = solution

        return best_solution

    def random_least_cost_repair(self, destroyed_solution: Solution) -> Solution:
        """Least cost repair for a single randomly-chosen insertion order."""
        repair_operator = SingleOrderLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        return repair_operator.repair()


class DestroyOperator(ABC):
    """Abstract base class for a destroy operator."""

    def __init__(
        self, problem: Problem, solution: Solution, destruction_degree: int
    ) -> None:
        self.problem = problem
        self.solution = solution
        self.nb_request_to_remove = destruction_degree

    @abstractmethod
    def destroy(self) -> Solution:
        "Main method of the DestroyOperator. Returns a parital solution."


class RepairOperator(ABC):
    """Abstract base class for a repair operator."""

    def __init__(
        self,
        problem: Problem,
        destroyed_solution: Solution,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        self.problem = problem
        self.destroyed_solution = destroyed_solution
        self.stack_assignment = destroyed_solution.stack_assignment
        self.optimization_objective = optimization_objective

        P = self.problem.P
        self.removed_pickups = [v for v in P if v not in destroyed_solution.order]

        self.insertion_orders = permute(self.removed_pickups)

    @property
    def items(self):
        return self.problem.items

    @property
    def partial_order(self) -> List[int]:
        return [v for v in self.destroyed_solution.order]

    def pickup_vertex(self, v: int) -> int:
        N = self.problem.N
        return v if v <= N else v - N

    def evaluate_solution(self, solution: Solution) -> int:
        return self.problem.evaluate_solution(solution, self.optimization_objective)

    def single_order_repair(self, insertion_order: List[int]) -> Solution:
        """Performs least cost insertion for the `insertion_order`."""
        N, _, _, _, _, M = self.problem.problem_data()
        items = self.items
        compartments = self.problem.vehicle.compartments

        # stack assingment to be extended after every request inserted
        stack_assignment = deepcopy(self.stack_assignment)
        # order to be updated after every request inserted
        extended_partial_order = self.partial_order

        # Try to insert the pickup vertex anywhere on any
        # stack with capacity then try to insert the delivery vertex
        # in a feasible location (without violating LIFO constraints)
        for vertex in insertion_order:
            item = items[vertex]
            # best cost and order for the current vertex
            best_extended_cost = None
            best_extended_partial_order = None
            best_extended_stack_assignment = None

            # Generic support for 1D and 2D items
            possible_stack_indices = [
                i for i in M if compartments[i].is_item_compatible(item)
            ]
            # Try a stack assignment
            for stack_idx in possible_stack_indices:
                # Try a pos in the visiting order for the pickup vertex
                for pick_pos in range(1, len(extended_partial_order) + 1):
                    # extend the extended order with the pick
                    ep_order = slice_and_insert(
                        extended_partial_order, vertex, pick_pos
                    )

                    # Try a pos for the delivery vertex
                    del_pos = pick_pos + 1
                    while del_pos < len(ep_order) + 1:

                        feasible = False
                        if del_pos == pick_pos + 1:
                            # delivery right after pickup
                            feasible = True

                        else:
                            # current pos is after a vertex for another request
                            # if the vertex is related to a different stack, it's
                            # possible to deliver the request after this vertex
                            other_vertex = ep_order[del_pos - 1]
                            curr_stack_assignment = stack_assignment[stack_idx]
                            if (
                                self.pickup_vertex(other_vertex)
                                not in curr_stack_assignment
                            ):
                                feasible = True

                            # else: other request assigned to the same stack
                            elif self.pickup_vertex(other_vertex) == other_vertex:
                                # if the vertex before is a pickup vertex for
                                # another request, we cannot deliver the to-be-inserted
                                # request since its item is not on top of the stack!
                                #  --> jump till the delivery pos of the other request
                                del_pos = ep_order.index(other_vertex + N) + 1
                                continue
                            else:
                                # the other vertex is a delivery vertex
                                # since this after delivering the other request,
                                # it's only feasible if the other request has been
                                # picked after the to-be-inserted one
                                if ep_order.index(other_vertex) > pick_pos:
                                    feasible = True

                        if feasible:
                            # extend the extended pick order with the delivery
                            ed_order = slice_and_insert(ep_order, vertex + N, del_pos)

                            # sanity check for stack capacity
                            ep_stack_assign = deepcopy(stack_assignment)
                            ep_stack_assign[stack_idx].append(vertex)

                            related_vertices = ep_stack_assign[stack_idx]
                            corresponding_deliveres = [i + N for i in related_vertices]

                            stack = compartments[stack_idx]
                            collected_demand = 0
                            for v in ed_order[1:]:
                                # Generic support for 1D and 2D items
                                demand = stack.demand_for_item(item)
                                if v in related_vertices:
                                    collected_demand += demand
                                elif v in corresponding_deliveres:
                                    collected_demand -= demand

                                if (
                                    collected_demand < 0
                                    or collected_demand > stack.capacity
                                ):
                                    # not a valid delivery position
                                    feasible = False
                                    break

                        if feasible:
                            # feasible partial solution found!
                            # check detour and consider least cost detour
                            solution = Solution(
                                self.items,
                                ed_order,
                                ep_stack_assign,
                                self.problem.vehicle,
                                is_partial=True,
                            )
                            e_cost = self.evaluate_solution(solution)
                            if (
                                best_extended_cost is None
                                or best_extended_cost > e_cost
                            ):
                                best_extended_cost = e_cost
                                # least cost detour found, update extended order
                                # with new request
                                best_extended_partial_order = ed_order
                                best_extended_stack_assignment = ep_stack_assign

                        del_pos += 1

            assert best_extended_partial_order, "A solution must be repairable!"
            # Update the partial order and the stack assignment for the
            # newly-inserted request
            extended_partial_order = best_extended_partial_order
            stack_assignment = best_extended_stack_assignment

        return Solution(
            self.items, extended_partial_order, stack_assignment, self.problem.vehicle
        )

    @abstractmethod
    def repair(self) -> Solution:
        """Main method of a RepairOperator. Returns a full, feasible solution."""


class SingleOrderLeastCostRepairOperator(RepairOperator):
    """
    Least Cost repair operator based on a single insertion order. In case an insertion
    order is specified upon initialization, the order is used. Otherwise, an
    arbitrary insertion order is selected.
    """

    def __init__(
        self,
        problem: Problem,
        destroyed_solution: Solution,
        insertion_order: List[int] = [],
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        super().__init__(problem, destroyed_solution, optimization_objective)
        self.insertion_order = insertion_order or random.choice(self.insertion_orders)

    def repair(self) -> Solution:
        return self.single_order_repair(self.insertion_order)


class ParallelOptimalLeastCostRepairOperator(RepairOperator):
    """
    Optimal Least Cost repair operator based on multiple parallel insertion orders.
    Each possible insertion order is solved independently in a process and the best
    solution with the best insertion cost is selected.
    """

    def repair(self) -> Solution:
        q: mp.Queue = mp.Queue()
        best_solution = None
        best_objective = None

        processes = (
            []
        )  # using a pool could actually be better because spawinging is expensive
        for order in self.insertion_orders:
            process = mp.Process(target=self.single_order_repair, args=(order, q))
            processes.append(process)
            process.start()

            sol = q.get()
            obj = self.evaluate_solution(sol)
            if not best_solution or obj < best_objective:
                best_solution = sol
                best_objective = obj

        for process in processes:
            process.join()

        return best_solution

    def single_order_repair(self, insertion_order, q):
        solution = super().single_order_repair(insertion_order)
        q.put(solution)


class PLNS(LNS):
    """
    Parallel LNS. Offers parallelization of solution repair to achieve a best-effort
    exhaustive exploration for solution neighborhood before time_limit.
    If the repair space is small enough, the best-effort is an optimal repair.
    """

    def __init__(
        self,
        problem: Problem,
        initial_solution: Solution,
        max_iterations: int = 1000,
        max_without_improvement: int = 10,
        time_limit: float = 1000,
        min_destruction_degree: float = 0.15,
        max_destruction_degree: float = 0.35,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        super().__init__(
            problem,
            initial_solution,
            max_iterations,
            max_without_improvement,
            time_limit,
            min_destruction_degree,
            max_destruction_degree,
            optimization_objective,
        )
        self.processing_pool = self.init_processing_pool()

    def init_processing_pool(self) -> mp.pool.Pool:
        print(f"Parallel LNS initialized with {CPU_COUNT} processes")
        return mp.Pool(CPU_COUNT)

    def terminate_processing_pool(self):
        self.processing_pool.close()
        self.processing_pool.terminate()

    def search(self) -> Solution:
        solution = super().search()
        print(
            f"Parallel LNS terminating w/ objective {self.evaluate_solution(solution)}"
            f" in {self.iteration} iterations"
        )
        self.terminate_processing_pool()
        return solution

    def repair(self, destroyed_solution: Solution) -> Solution:
        """Parallel Best-effort repair for the destroyed solution."""

        arbitrary_repair_operator = SingleOrderLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        insertion_orders = arbitrary_repair_operator.insertion_orders

        def unwrap_problem(problem: Problem):
            # needed for pickling
            items = list(self.problem.items.values())
            vehicle = Vehicle(
                [
                    TwoDimensionalCompartment(comp.depth, comp.length)
                    for comp in problem.vehicle.compartments
                ]
            )
            return items, vehicle, deepcopy(problem.C)

        best_solution = None
        op_objective = self.optimization_objective
        async_results = [
            self.processing_pool.apply_async(
                process_repair,
                args=(
                    *unwrap_problem(self.problem),
                    destroyed_solution,
                    insertion_order,
                    op_objective,
                ),
            )
            for insertion_order in insertion_orders
        ]

        best_solution = async_results[0].get()
        best_objective = self.evaluate_solution(best_solution)
        remaining_time = self.time_limit - (time.time() - self.start_time)
        for res in async_results:
            try:
                sol = res.get(timeout=remaining_time)
            except mp.TimeoutError:
                print(
                    "Parallel LNS timed out, not all insertion orders could be tried."
                )
            else:
                obj = self.evaluate_solution(sol)
                if obj < best_objective:
                    best_solution = sol
                    best_objective = obj

        assert best_solution, "solution must be repairable!"
        return best_solution

    def repair_pool(self, destroyed_solution: Solution) -> Solution:
        """TODO Remove. Just keep for now for reference to compare using a
        processing pool in PLNS or delegate pool initialization and termination
        in the repair function. --> This could be also delegated to a repair operator.
        """
        arbitrary_repair_operator = SingleOrderLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        insertion_orders = arbitrary_repair_operator.insertion_orders

        def unwrap_problem(problem: Problem):
            items = list(self.problem.items.values())
            vehicle = Vehicle(
                [
                    TwoDimensionalCompartment(comp.depth, comp.length)
                    for comp in problem.vehicle.compartments
                ]
            )
            return items, vehicle, deepcopy(problem.C)

        best_solution = None
        op_objective = self.optimization_objective
        nb_processes = mp.cpu_count()
        print(
            f"It: {self.iteration} Initializing {nb_processes} processes for"
            f"{len(arbitrary_repair_operator.insertion_orders)}"
        )
        with mp.Pool(processes=nb_processes) as pool:
            async_results = [
                pool.apply_async(
                    process_repair,
                    args=(
                        *unwrap_problem(self.problem),
                        destroyed_solution,
                        order,
                        op_objective,
                    ),
                )
                for order in insertion_orders
            ]

            best_solution = async_results[0].get()
            best_objective = self.evaluate_solution(best_solution)
            for res in async_results:
                sol = res.get()
                obj = self.evaluate_solution(sol)
                if obj < best_objective:
                    best_solution = sol
                    best_objective = obj

        assert best_solution, "solution must be repairable!"
        return best_solution


def process_repair(
    items,
    vehicle,
    distance_matrix,
    destroyed_solution,
    insertion_order,
    optimization_objective,
) -> Solution:
    problem = TwoDimensionalProblem(items, vehicle, distance_matrix)
    operator = SingleOrderLeastCostRepairOperator(
        problem, destroyed_solution, insertion_order, optimization_objective
    )
    sol = operator.repair()
    return sol


def permute(input: List[Any]) -> List[List[Any]]:
    tupled = list(itertools.permutations(input))
    listed = [list(tup) for tup in tupled]
    return listed


def slice_and_insert(input_list: List[int], sth: int, idx: int) -> List[int]:
    return input_list[:idx] + [sth] + input_list[idx:]
