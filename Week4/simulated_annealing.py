import math
import random
import matplotlib.pyplot as plt
import animated_visualizer


class SimAnneal:
    def __init__(self, coordinates, place, stopping_iter, N=-1, nodes=-1, temp=-1, stopping_temperature=-1):
        self.coords = coordinates
        self.place = place
        self.N = len(coordinates)
        self.stopping_temperature = 1e-8
        self.temp = 1000
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.nodes = list(range(self.N))
        self.best_path = None
        self.best_cost = float("inf")
        self.cost_list = []
        self.path_history = []

    def path_cost(self, solution):
        return sum(self.dist(solution[i % self.N], solution[(i + 1) % self.N]) for i in range(self.N))

    def dist(self, node0, node1):
        coord0, coord1 = self.coords[node0], self.coords[node1]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord0, coord1)))

    def accept(self, candidate):
        candidate_cost = self.path_cost(candidate)

        if candidate_cost < self.best_cost:
            self.best_cost = candidate_cost
            self.best_path = candidate
        else:
            delta_energy = candidate_cost - self.current_cost
            acceptance_probability = math.exp(-delta_energy / self.temp)
            if random.random() < acceptance_probability:
                self.current_cost = candidate_cost
                self.current_path = candidate

    def initial_solution(self):
        path = [random.choice(self.nodes)]
        remaining_nodes = set(self.nodes) - set(path)

        while remaining_nodes:
            current_node = path[-1]
            next_node = min(remaining_nodes, key=lambda x: self.dist(current_node, x))
            path.append(next_node)
            remaining_nodes.remove(next_node)

        initial_cost = self.path_cost(path)
        if initial_cost < self.best_cost:
            self.best_cost = initial_cost
            self.best_path = path

        self.cost_list.append(initial_cost)
        self.path_history.append(path)
        return path, initial_cost

    def simulated_annealing(self):
        self.current_path, self.current_cost = self.initial_solution()
        cooling_rate = 0.9995

        while self.temp >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = self.current_path[:]
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - 1)
            candidate[i:(i+l)] = reversed(candidate[i:(i+l)])
            self.accept(candidate)

            self.temp *= cooling_rate
            self.iteration += 1
            self.cost_list.append(self.current_cost)
            self.path_history.append(self.current_path)

        print(f"Best cost obtained: {self.best_cost}")

    def display_optimal_path(self):
        tour = ' -> '.join(self.place[i] for i in self.best_path + [self.best_path[0]])
        print(f"Optimal Path: {tour}")

    def animateSolutions(self):
        animated_visualizer.animateTSP(self.path_history, self.coords)

    def plot_learning(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_list)), self.cost_list)
        plt.axhline(y=self.cost_list[0], color='r', linestyle='--', label='Initial Cost')
        plt.axhline(y=self.best_cost, color='g', linestyle='--', label='Optimized Cost')
        plt.title("Learning Progress")
        plt.ylabel("Cost")
        plt.xlabel("Iteration")
        plt.legend()
        plt.show()
