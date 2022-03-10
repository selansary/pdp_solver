#!/usr/bin/env python3
import itertools
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List

import numpy as np

from . import Problem, Solution


class AcceptanceCriterion(ABC):
    @abstractmethod
    def apply_acceptance_function(
        self, old_solution: Solution, new_solution: Solution, problem: Problem
    ) -> bool:
        pass

    @abstractmethod
    def update_acceptance_params(self, *args, **kwargs):
        pass


class SimulatedAnnealing(AcceptanceCriterion):
    def __init__(
        self, alpha: float = 0.99975, starting_temperature: float = 1.05
    ) -> None:
        self.alpha = alpha
        self.temperature = starting_temperature

    def apply_acceptance_function(
        self, old_solution: Solution, new_solution: Solution, problem: Problem
    ) -> bool:
        return self._apply_acceptance_function(old_solution, new_solution, problem)

    def _apply_acceptance_function(
        self,
        old_solution: Solution,
        new_solution: Solution,
        problem: Problem,
        update_params: bool = True,
    ) -> bool:
        old_obj = problem.evaluate_solution(old_solution)
        new_obj = problem.evaluate_solution(new_solution)
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


class LNS:
    def __init__(
        self, problem: Problem, max_iteration: int = 10, time_limit: float = 5.0
    ) -> None:
        self.problem = problem

        # TODO possibly variable stopping criterion
        self.time_limit = time_limit
        self.max_iteration = max_iteration

        # TODO possibly variable acceptance criterion
        self.acceptance_criterion = SimulatedAnnealing()

        # TODO set destroy strategy
        # TODO set insertion strategy

    def search(self, initial_solution: Solution) -> Solution:
        best_solution = initial_solution
        best_obj = self.problem.evaluate_solution(best_solution)

        current_solution = initial_solution
        nb_requests = len(initial_solution.items)

        it = 0
        start_time = time.time()
        while it < self.max_iteration and time.time() - start_time < self.time_limit:
            # Randomly select a nb of requests to be removed from the solution
            nb_removed = random.randint(1, nb_requests // 3)

            # Gradually decrement number of requets to remove: by 1 every 5 iterations
            # nb_removed = max(1, (nb_requests // 3) - (it // 20))
            # Gradually increment number of requets to remove: by 1 every 5 iterations
            # nb_removed = max(1, (nb_requests // 3) + (it // 5))

            # Apply the destroy and repair operators to retrieve a new solution
            # in the neighborhood of the current solution
            solution = deepcopy(current_solution)
            solution = self.repair(self.destroy(solution, nb_removed))

            curr_obj = self.problem.evaluate_solution(current_solution)
            new_obj = self.problem.evaluate_solution(solution)

            # Check if the objective improved or if the acceptance criterion allows
            # the new unimproved solution
            # (applicable in case the problem is a minimization problem)
            if new_obj < curr_obj or self.accept(current_solution, solution):
                current_solution = solution

                if new_obj < best_obj:
                    best_solution = solution
                    self.best_solution = best_solution
                    best_obj = new_obj

            it += 1

        return best_solution

    def accept(self, old_solution: Solution, new_solution: Solution) -> bool:
        return self.acceptance_criterion.apply_acceptance_function(
            old_solution, new_solution, self.problem
        )

    def destroy(self, solution: Solution, nb_requests_to_remove: int) -> Solution:
        return self.random_destroy(solution, nb_requests_to_remove)

    def repair(self, destroyed_solution: Solution):
        return self.least_cost_repair(destroyed_solution)

    def random_destroy(
        self, solution: Solution, nb_requests_to_remove: int
    ) -> Solution:
        N = len(self.problem.items)
        pickup_vertices = self.problem.P
        # This performs random with replacement!! May result in wrong / infeasible solutions
        # requests_to_remove = random.choices(pickup_vertices, k=nb_requests_to_remove)
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
            self.problem.items, partial_order, partial_stack_assignment, is_partial=True
        )

    def _least_cost_repair(self, destroyed_solution: Solution) -> Solution:
        N, V, P, D, C, M = self.problem.problem_vars()
        items = self.problem.items

        pickup_vertices = self.problem.P
        removed_pickups = [
            v for v in pickup_vertices if v not in destroyed_solution.order
        ]

        # removed_deliveries = [v + N for v in removed_pickups]
        # missing_vertices = removed_pickups + removed_deliveries

        def pickup_vertex(v: int) -> int:
            return v if v <= N else v - N

        def slice_and_insert(input_list: List[int], sth: int, ind: int) -> List[int]:
            return input_list[:ind] + [sth] + input_list[ind:]

        def evaluate_route_cost(route_order: List[int]) -> int:
            cost = 0
            prev = 0  # we always start at the depot
            for v in route_order[1:]:
                cost += C[prev][v]
                prev = v
            return cost

        # Try all possible insertion orders for the removed vertices
        # and choose the order that results in the least cost solution
        # (different from paper which specifies "insertion based on removal order")
        partial_order = destroyed_solution.order
        insertion_orders = permute(removed_pickups)
        best_total_cost = None
        best_total_order = None
        best_stack_assignment = None
        insertion_order = random.choices(insertion_orders, k=1)[0]

        # Try this insertion order and perform least cost insertion
        # for the vertices. Try to insert the pickup vertex anywhere on any
        # stack with capacity then try to insert the delivery vertex
        # in a feasible location (without violated LIFO constraints)
        stack_assignment = deepcopy(destroyed_solution.stack_assignment)
        extended_partial_order = list(partial_order)
        for vertex in insertion_order:
            # order to be updated for after every request inserted
            # in the current insert order
            best_extended_cost = None
            best_extended_partial_order = None
            best_extended_stack_assignment = None

            # Try a stack assignment
            for stack_idx in M:
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
                            if pickup_vertex(other_vertex) not in curr_stack_assignment:
                                feasible = True

                            # else: other request assigned to the same stack
                            elif pickup_vertex(other_vertex) == other_vertex:
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

                            stack = self.problem.vehicle.compartments[stack_idx]
                            collected_demand = 0
                            for v in ed_order[1:]:
                                demand = items[pickup_vertex(v)].length
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
                            # feasible solution found!
                            # check detour and consider least cost detour
                            e_cost = evaluate_route_cost(ep_order)
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

            # Update the partial order and the stack assignment for the
            # newly-inserted request
            extended_partial_order = best_extended_partial_order
            stack_assignment = best_extended_stack_assignment

        # Check if this insertion order contributes to the lowest cost total order
        total_cost = evaluate_route_cost(extended_partial_order)
        if best_total_cost is None or best_total_cost > total_cost:
            best_total_cost = total_cost
            best_total_order = extended_partial_order
            best_stack_assignment = best_extended_stack_assignment

        return Solution(items, best_total_order, best_stack_assignment)

    def least_cost_repair(self, destroyed_solution: Solution) -> Solution:
        N, V, P, D, C, M = self.problem.problem_vars()
        items = self.problem.items

        pickup_vertices = self.problem.P
        removed_pickups = [
            v for v in pickup_vertices if v not in destroyed_solution.order
        ]

        # removed_deliveries = [v + N for v in removed_pickups]
        # missing_vertices = removed_pickups + removed_deliveries

        def pickup_vertex(v: int) -> int:
            return v if v <= N else v - N

        def slice_and_insert(input_list: List[int], sth: int, ind: int) -> List[int]:
            return input_list[:ind] + [sth] + input_list[ind:]

        def evaluate_route_cost(route_order: List[int]) -> int:
            cost = 0
            prev = 0  # we always start at the depot
            for v in route_order[1:]:
                cost += C[prev][v]
                prev = v
            return cost

        # Try all possible insertion orders for the removed vertices
        # and choose the order that results in the least cost solution
        # (different from paper which specifies "insertion based on removal order")
        partial_order = destroyed_solution.order
        insertion_orders = permute(removed_pickups)
        insertion_order = random.choices(insertion_orders, k=1)[0]

        # Try this insertion order and perform least cost insertion
        # for the vertices. Try to insert the pickup vertex anywhere on any
        # stack with capacity then try to insert the delivery vertex
        # in a feasible location (without violated LIFO constraints)
        stack_assignment = deepcopy(destroyed_solution.stack_assignment)
        extended_partial_order = list(partial_order)
        for vertex in insertion_order:
            # order to be updated for after every request inserted
            # in the current insert order
            best_extended_cost = None
            best_extended_partial_order = None
            best_extended_stack_assignment = None

            # Try a stack assignment
            for stack_idx in M:
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
                            if pickup_vertex(other_vertex) not in curr_stack_assignment:
                                feasible = True

                            # else: other request assigned to the same stack
                            elif pickup_vertex(other_vertex) == other_vertex:
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

                            stack = self.problem.vehicle.compartments[stack_idx]
                            collected_demand = 0
                            for v in ed_order[1:]:
                                demand = items[pickup_vertex(v)].length
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
                            # feasible solution found!
                            # check detour and consider least cost detour
                            e_cost = evaluate_route_cost(ep_order)
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

            # Update the partial order and the stack assignment for the
            # newly-inserted request
            extended_partial_order = best_extended_partial_order
            stack_assignment = best_extended_stack_assignment

        return Solution(items, extended_partial_order, stack_assignment)


def permute(input: List[Any]) -> List[List[Any]]:
    tupled = list(itertools.permutations(input))
    listed = [list(tup) for tup in tupled]
    return listed
