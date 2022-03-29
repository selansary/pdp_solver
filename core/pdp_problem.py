import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from gurobipy import GRB, Model, Var, tupledict

from . import Compartment, Item, Solution, TwoDCompartment, TwoDItem, Vehicle


class ABSProblem(ABC):  # TODO
    def __init__(
        self,
        items: List[Item],
        distance_matrix: List[List[int]] = [],
        vehicle: Optional[Vehicle] = None,
        name: str = "PDP",
    ) -> None:
        super().__init__()

    @abstractmethod
    def create_model(self) -> Model:
        """"""


class Problem:
    def __init__(
        self,
        items: List[Item],
        distance_matrix: List[List[int]] = [],
        vehicle: Optional[Vehicle] = None,
        name: str = "PDP",
    ) -> None:

        self.items = {i + 1: item for i, item in enumerate(items)}
        self.N = len(items)
        self.V = [i for i in range(self.N * 2 + 1)]
        self.P = self.V[1 : self.N + 1]
        self.D = self.V[self.N + 1 :]
        self.PUD = self.P + self.D
        self.demands = (
            [0] + [item.length for item in items] + [-1 * item.length for item in items]
        )

        # Assume an arbitrary graph
        # Distance from / to depot is 0
        random.seed(0)
        self.C = distance_matrix or [
            [
                random.randint(1, 100) if i != 0 and j != 0 and i != j else 0
                for i in self.V
            ]
            for j in self.V
        ]
        nb_compartments = 3
        self.vehicle = vehicle or Vehicle(
            compartments=[Compartment(800) for _ in range(nb_compartments)]
        )
        self.M = list(range(len(self.vehicle.compartments)))

        self.name = name
        self.model: Optional[Model] = None
        self.data: Dict[str, tupledict[Tuple[Any, ...], Var]] = dict()

    def create_model(self) -> Model:
        model = Model("PDPMS")
        V = self.V
        M = self.M
        x = model.addVars(V, V, vtype=GRB.BINARY, name="x")
        y = model.addVars(V, M, vtype=GRB.BINARY, name="y")
        u = model.addVars(V, vtype=GRB.INTEGER, name="u")
        s = model.addVars(V, M, vtype=GRB.INTEGER, name="s")

        # Set the bounds of the integer vars
        for i in u:
            u[i].setAttr(GRB.Attr.LB, 0)
            u[i].setAttr(GRB.Attr.UB, len(V) - 1)

        for i, k in s:
            s[i, k].setAttr(GRB.Attr.LB, 0)
            s[i, k].setAttr(GRB.Attr.UB, self.vehicle.compartments[k].capacity)

        # Apply the bounds to the var domains
        model.update()

        self.data = dict(x=x, y=y, u=u, s=s)
        self.model = model
        return model

    def problem_vars(self):
        return self.N, self.V, self.P, self.D, self.C, self.M

    def model_vars(self):
        return self.data["x"], self.data["y"], self.data["u"], self.data["s"]

    def apply_constraints(self):
        N, V, P, D, _, M = self.problem_vars()
        x, y, u, s = self.model_vars()

        # 2. Every vertex is left once.
        self.model.addConstrs(
            (sum(x[i, j] for j in V) == 1 for i in V), name="out_vertex_constr"
        )
        # 3. Every vertex is entered once.
        self.model.addConstrs(
            (sum(x[j, i] for j in V) == 1 for i in V), name="in_vertex_constr"
        )
        # 4. Demand of every item must be fulfilled.
        self.model.addConstrs(
            ((sum(y[i, k] for k in M) == 1) for i in P), name="fulfill_demands_constr"
        )
        # 5. The route is constituted by the order of visiting vertices.
        self.model.addConstrs(
            (u[j] >= u[i] + 1 - (2 * N * (1 - x[i, j])) for i in V for j in P + D),
            name="visiting_order_constr",
        )
        # 6. Every item must be picked before it's delivered.
        self.model.addConstrs(
            (u[N + i] >= u[i] + 1 for i in P), name="pickup_before_delivery_constr"
        )

        ### stack constraints ###
        # 7
        self.model.addConstrs(
            (
                s[j, k]
                >= s[i, k]
                + self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_pickup_constr",
        )
        # 8 sub items[N + j] with -items[j]
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[i, k]
                - self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_delivery_constr",
        )
        # 9 sub items[N + j] with -items[j]
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[j, k]
                - self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )
        ### End of stack constraints ###

        # 10. We always start at the depot
        self.model.addConstr(u[0] == 0, name="start_at_depot_constr")
        # 11. Vertices are order along the route.
        # constraint implicit in the domain of u
        # 12. We always begin with empty stacks at the depot.
        self.model.addConstrs(
            (s[0, k] == 0 for k in M), name="empty_stacks_at_depot_constr"
        )
        # 13. The stack capacity is always maintained for all stacks.
        # constraint implicit in the domain of s

    def set_model_objective(self):
        N, V, P, D, C, M = self.problem_vars()
        x, y, u, s = self.model_vars()

        self.model.setObjective(
            (sum(C[i][j] * x[i, j] for i in V for j in V)), GRB.MINIMIZE
        )

    def presolve(self, time_limit: float = 3.0, write_model: bool = True):
        if not self.model:
            return

        if write_model:
            self.model.write("./lp_models/" + self.name + ".lp")

        self.model.Params.TimeLimit = time_limit
        self.model.optimize()

    def extract_solution(self) -> Solution:
        _, V, P, _, _, M = self.problem_vars()
        _, y, u, _ = self.model_vars()

        u_list = [int(u[i].x) for i in V]
        order = [u_list.index(i) for i in range(len(V))]
        stack_assignment = [[i for i in P if abs(y[i, k].x) > 1e-6] for k in M]

        return Solution(self.items, order, stack_assignment, self.vehicle)

    def evaluate_solution(self, solution: Solution) -> int:
        """Returns the objective value of the solution."""
        order = solution.order
        cost = 0
        prev = 0  # we always start at the depot
        for vertex in order[1:]:
            cost += self.C[prev][vertex]
            prev = vertex
        return cost


class TwoDProblem:
    def __init__(
        self,
        items: List[TwoDItem],
        distance_matrix: List[List[int]] = [],
        vehicle: Optional[Vehicle] = None,
        name: str = "2DPDP",
    ) -> None:

        self.items = {i + 1: item for i, item in enumerate(items)}
        self.N = len(items)
        self.V = [i for i in range(self.N * 2 + 1)]
        self.P = self.V[1 : self.N + 1]
        self.D = self.V[self.N + 1 :]
        self.PUD = self.P + self.D

        # Assume an arbitrary graph
        # Distance from / to depot is 0
        random.seed(0)
        self.C = distance_matrix or [
            [
                random.randint(1, 100) if i != 0 and j != 0 and i != j else 0
                for i in self.V
            ]
            for j in self.V
        ]

        # Default to SOTO Problem 2 600x800 and 4 300x800 compartments
        self.vehicle = vehicle or Vehicle(
            compartments=[TwoDCompartment(800, 600)] * 2
            + [TwoDCompartment(800, 300)] * 4
        )
        self.M = list(range(len(self.vehicle.compartments)))
        self.compartment_capabilities = self.init_compatibility_matrix()

        self.name = name
        self.model: Optional[Model] = None
        self.data: Dict[str, tupledict[Tuple[Any, ...], Var]] = dict()

    def init_compatibility_matrix(self) -> List[Dict[int, int]]:
        def item_fits_in_compartment(item: Item, compartment: Compartment) -> int:
            if item is None:
                return 0
            if item.length == compartment.length and item.width <= compartment.depth:
                return 1
            if item.width == compartment.length and item.length <= compartment.depth:
                return 1
            return 0

        return [
            {p: item_fits_in_compartment(item, comp) for p, item in self.items.items()}
            for comp in self.vehicle.compartments
        ]

    def create_model(self) -> Model:
        model = Model("2DPDPMS")
        V = self.V
        M = self.M
        P = self.P
        x = model.addVars(V, V, vtype=GRB.BINARY, name="x")
        y = model.addVars(V, M, vtype=GRB.BINARY, name="y")
        u = model.addVars(V, vtype=GRB.INTEGER, name="u")
        s = model.addVars(V, M, vtype=GRB.INTEGER, name="s")
        r = model.addVars(P, vtype=GRB.BINARY, name="r")
        z = model.addVars(P, M, vtype=GRB.BINARY, name="z")

        # Set the bounds of the integer vars
        for i in u:
            u[i].setAttr(GRB.Attr.LB, 0)
            u[i].setAttr(GRB.Attr.UB, len(V) - 1)

        for i, k in s:
            s[i, k].setAttr(GRB.Attr.LB, 0)
            s[i, k].setAttr(GRB.Attr.UB, self.vehicle.compartments[k].capacity)

        # Apply the bounds to the var domains
        model.update()

        self.data = dict(x=x, y=y, u=u, s=s, r=r, z=z)
        self.model = model
        return model

    def problem_vars(self):
        return self.N, self.V, self.P, self.D, self.C, self.M

    def model_vars(self):
        return self.data["x"], self.data["y"], self.data["u"], self.data["s"]

    def apply_constraints(self):
        N, V, P, D, _, M = self.problem_vars()
        x, y, u, s = self.model_vars()

        # 2. Every vertex is left once.
        self.model.addConstrs(
            (sum(x[i, j] for j in V) == 1 for i in V), name="out_vertex_constr"
        )
        # 3. Every vertex is entered once.
        self.model.addConstrs(
            (sum(x[j, i] for j in V) == 1 for i in V), name="in_vertex_constr"
        )
        # 4. Demand of every item must be fulfilled.
        self.model.addConstrs(
            ((sum(y[i, k] for k in M) == 1) for i in P), name="fulfill_demands_constr"
        )
        # 5. The route is constituted by the order of visiting vertices.
        self.model.addConstrs(
            (u[j] >= u[i] + 1 - (2 * N * (1 - x[i, j])) for i in V for j in P + D),
            name="visiting_order_constr",
        )
        # 6. Every item must be picked before it's delivered.
        self.model.addConstrs(
            (u[N + i] >= u[i] + 1 for i in P), name="pickup_before_delivery_constr"
        )

        ### stack constraints ###
        I = self.items
        z = self.data["z"]
        Capacities = [comp.capacity for comp in self.vehicle.compartments]

        # 7
        load_expr = (
            lambda j_, k_: I[j_].length * z[j_, k_]
            + I[j_].width * y[j_, k_]
            - I[j_].width * z[j_, k_]
        )
        self.model.addConstrs(
            (
                s[j, k] >= s[i, k] + load_expr(j, k) - Capacities[k] * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_pickup_constr",
        )
        #
        self.model.addConstrs(
            (
                s[j, k] <= s[i, k] + load_expr(j, k) + Capacities[k] * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_pickup_constr",
        )

        # 8
        unload_expr = lambda j_, k_: -1 * load_expr(j_, k_)
        # unload_expr = lambda j_, k_: - I[j_].l * z[j_, k_] - I[j_].w * y[j_, k_] \
        # + I[j_].w * z[j_, k_]
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[i, k] + unload_expr(j, k) - Capacities[k] * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_delivery_constr",
        )
        #
        self.model.addConstrs(
            (
                s[N + j, k]
                <= s[i, k] + unload_expr(j, k) + Capacities[k] * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_delivery_constr",
        )

        # 9
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[j, k] + unload_expr(j, k) - Capacities[k] * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )
        # Additional constraint
        self.model.addConstrs(
            (
                s[N + j, k]
                <= s[j, k] + unload_expr(j, k) + Capacities[k] * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )

        ### End of stack constraints ###

        # 10. We always start at the depot
        self.model.addConstr(u[0] == 0, name="start_at_depot_constr")
        # 11. Vertices are order along the route.
        # constraint implicit in the domain of u
        # 12. We always begin with empty stacks at the depot.
        self.model.addConstrs(
            (s[0, k] == 0 for k in M), name="empty_stacks_at_depot_constr"
        )
        # 13. The stack capacity is always maintained for all stacks.
        # constraint implicit in the domain of s

        # Rotation Constraints
        r = self.data["r"]
        # 14
        self.model.addConstrs(
            (z[i, k] <= r[i] for i in P for k in M), name="z_r_constr"
        )

        # 15
        self.model.addConstrs(
            (z[i, k] <= y[i, k] for i in P for k in M), name="z_y_constr"
        )

        # 16
        self.model.addConstrs(
            (z[i, k] >= r[i] + y[i, k] - 1 for i in P for k in M), name="z_r_y_constr"
        )

        # Item-Compartment compatibility constraint
        Capabilities = self.compartment_capabilities
        # 17
        self.model.addConstrs(
            (y[i, k] <= Capabilities[k][i] for i in P for k in M),
            name="item_comp_constr",
        )

        # 18 rest of rotation constraints
        same_length = (
            lambda i_, k_: I[i_].length == self.vehicle.compartments[k_].length
        )
        self.model.addConstrs(
            (y[i, k] <= 1 - r[i] for i in P for k in M if same_length(i, k)),
            name="item_comp_not_rot_constr",
        )

        # 19
        self.model.addConstrs(
            (y[i, k] <= r[i] for i in P for k in M if not same_length(i, k)),
            name="item_comp_rot_constr",
        )
        self.model.update()

    def set_model_objective(self):
        N, V, P, D, C, M = self.problem_vars()
        x, y, u, s = self.model_vars()

        # Objective that also considers rotation
        # r = self.data["r"]
        # sum_expr = sum(C[i][j] * x[i, j] for i in V for j in V)
        # sum_expr += sum(r[i] for i in P)
        # self.model.setObjective(sum_expr, GRB.MINIMIZE)

        self.model.setObjective(
            (sum(C[i][j] * x[i, j] for i in V for j in V)), GRB.MINIMIZE
        )

    def presolve(self, time_limit: float = 3.0, write_model: bool = True):
        if not self.model:
            return

        if write_model:
            self.model.write("./lp_models/" + self.name + ".lp")

        self.model.Params.TimeLimit = time_limit
        self.model.optimize()

    def extract_solution(self) -> Solution:
        _, V, P, _, _, M = self.problem_vars()
        _, y, u, _ = self.model_vars()

        u_list = [int(u[i].x) for i in V]
        order = [u_list.index(i) for i in range(len(V))]
        stack_assignment = [[i for i in P if abs(y[i, k].x) > 1e-6] for k in M]
        # TODO include Rotaiton in solution?

        return Solution(self.items, order, stack_assignment, self.vehicle)

    def evaluate_solution(self, solution: Solution) -> int:
        """Returns the objective value of the solution."""
        order = solution.order
        cost = 0
        prev = 0  # we always start at the depot
        for vertex in order[1:]:
            cost += self.C[prev][vertex]
            prev = vertex
        return cost
