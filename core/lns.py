import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Callable, List, Optional

import numpy as np

from . import (
    OptimizationObjective,
    ParallelOptimalLeastCostRepairOperator,
    Problem,
    RandomDestroyOperator,
    SingleOrderLeastCostRepairOperator,
    Solution,
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
    """Randomly choose a suitable degree of destruction for each search iteration."""

    CONSTANT = auto()
    """Choose a constant degree of destruction throughout the search."""

    GRADUALLY_INCREASING = auto()
    """Increase the degree of destruction gradually as the search progresses."""

    GRADUALLY_DECREASING = auto()
    """Decrease the degree of destruction gradually as the search progresses."""


class DestroyStrategy(Enum):
    """Strategy to be used for the destroy operator during LNS."""

    RANDOM = auto()


class RepairStrategy(Enum):
    """Strategy to be used for the repair operator during LNS."""

    RANDOM_SINGLE_ORDER_LEAST_COST = auto()
    PARALLEL_OPTIMAL_LEAST_COST = auto()


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
        destroy_strategy: DestroyStrategy = DestroyStrategy.RANDOM,
        repair_strategy: RepairStrategy = (
            RepairStrategy.RANDOM_SINGLE_ORDER_LEAST_COST
        ),
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

        `destroy_strategy` is the strategy for the destroy operator with default
        strategy: `DestroyStrategy.RANDOM`. Other strategies can be set using:
        `set_destroy_strategy`.

        `repair_strategy` is the strategy for the repair operator with default strategy:
        `RepairStrategy.RANDOM_SINGLE_ORDER_LEAST_COST`. Other strategies can be set
        using: `set_repair_strategy`.

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

        # Destroy and Repair strategies
        self.destroy_strategy = destroy_strategy
        self.repair_strategy = repair_strategy

        self.optimization_objective = optimization_objective

        # TODO possibly variable acceptance criterion
        self.acceptance_criterion = SimulatedAnnealing()

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
        print(
            f"Terminating LNS in {self.iteration} iterations and "
            f"objs {[s.objective for s in self.solutions_cache]}"
        )
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
        return min(max_degree, int(min_degree + ((1.025) ** self.iteration)))

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

    def set_destroy_strategy(self, destroy_strategy: DestroyStrategy):
        """Set the destroy strategy for the search."""
        self.destroy_strategy = destroy_strategy

    def _random_destroy(
        self, solution: Solution, nb_requests_to_remove: int
    ) -> Solution:
        destroy_operator = RandomDestroyOperator(
            self.problem, solution, nb_requests_to_remove
        )
        return destroy_operator.destroy()

    def destroy(self, solution: Solution, nb_requests_to_remove: int) -> Solution:
        """Return a destroyed copy of the solution."""
        if self.destroy_strategy == DestroyStrategy.RANDOM:
            return self._random_destroy(solution, nb_requests_to_remove)
        else:
            raise NotImplementedError

    def set_repair_strategy(self, repair_strategy: RepairStrategy):
        """Set the repair strategy for the search."""
        self.repair_strategy = repair_strategy

    def _random_least_cost_repair(self, destroyed_solution: Solution) -> Solution:
        """Least cost repair for a single randomly-chosen insertion order."""
        repair_operator = SingleOrderLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        return repair_operator.repair()

    def _parallel_optimal_least_cost_repair(
        self, destroyed_solution: Solution
    ) -> Solution:
        """Exhaustive Least cost repair for all possible insertion orders."""
        repair_operator = ParallelOptimalLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        return repair_operator.repair()

    def repair(self, destroyed_solution: Solution) -> Solution:
        """Return a repaired copy of the destroyed solution."""
        if self.repair_strategy == RepairStrategy.RANDOM_SINGLE_ORDER_LEAST_COST:
            return self._random_least_cost_repair(destroyed_solution)
        elif self.repair_strategy == RepairStrategy.PARALLEL_OPTIMAL_LEAST_COST:
            return self._parallel_optimal_least_cost_repair(destroyed_solution)
        else:
            raise NotImplementedError


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
        destroy_strategy: DestroyStrategy = DestroyStrategy.RANDOM,
        repair_strategy: RepairStrategy = (
            RepairStrategy.RANDOM_SINGLE_ORDER_LEAST_COST
        ),
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
            destroy_strategy,
            repair_strategy,
            optimization_objective,
        )
        self.processing_pool = self.init_processing_pool()

    def init_processing_pool(self) -> "mp.pool.Pool":
        print(f"Parallel LNS initialized with {CPU_COUNT} processes")
        return mp.Pool(CPU_COUNT)

    def terminate_processing_pool(self):
        self.processing_pool.close()
        self.processing_pool.terminate()

    def search(self) -> Solution:
        solution = super().search()
        duration = time.time() - self.start_time
        print(
            f"Parallel LNS terminating w/ objective {self.evaluate_solution(solution)}"
            f" in {self.iteration} iterations in {duration} secs"
        )
        self.terminate_processing_pool()
        return solution

    def repair(self, destroyed_solution: Solution) -> Solution:
        """Parallel best-effort repair for the destroyed solution."""
        # Get all possible insertion orders for missing vertices
        arbitrary_repair_operator = SingleOrderLeastCostRepairOperator(
            self.problem,
            destroyed_solution,
            optimization_objective=self.optimization_objective,
        )
        insertion_orders = arbitrary_repair_operator.insertion_orders

        # Distribute insertion orders on parallel processes
        async_results = [
            self.processing_pool.apply_async(
                process_repair,
                args=(
                    self.problem,
                    destroyed_solution,
                    insertion_order,
                    self.optimization_objective,
                ),
            )
            for insertion_order in insertion_orders
        ]

        best_solution = async_results[0].get()
        best_objective = self.evaluate_solution(best_solution)
        for res in async_results:
            try:
                sol = res.get(timeout=self.time_limit - (time.time() - self.start_time))
            except mp.TimeoutError:
                print(
                    "Parallel LNS timed out, not all insertion orders could be tried."
                )
                break
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

        best_solution = None
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
                        self.problem,
                        destroyed_solution,
                        order,
                        self.optimization_objective,
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
    problem: Problem,
    destroyed_solution: Solution,
    insertion_order: List[int],
    optimization_objective: OptimizationObjective,
) -> Solution:
    operator = SingleOrderLeastCostRepairOperator(
        problem, destroyed_solution, insertion_order, optimization_objective
    )
    solution = operator.repair()
    return solution
