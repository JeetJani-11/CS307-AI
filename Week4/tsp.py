import random
from simulated_annealing import SimAnneal
import numpy as np
import time


def parse_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    dimension = None
    coordinates = []
    locations = []
    node_coord_section = False

    for line in lines:
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
            print(f'Dimension: {dimension}')
        elif line.startswith('NODE_COORD_SECTION'):
            node_coord_section = True
        elif node_coord_section:
            parts = line.strip().split()
            if len(parts) == 3:
                location, x, y = parts
                coordinates.append([float(x), float(y)])
                locations.append(location)

    print(f"File Name: {file_path.split('/')[-1]}")
    return coordinates, locations


def main():
    file_path = "Data/rajasthan.tsp"
    coordinates, locations = parse_tsp_file(file_path)
    coord_array = np.array(coordinates)
    
    start_time = time.perf_counter_ns()
    
    sa_solver = SimAnneal(coord_array, locations, stopping_iter=len(coord_array) * 10000000)
    
    end_time = time.perf_counter_ns()
    print(f'Initialization Time: {(end_time - start_time) / 1e9:.6f} seconds')
    
    sa_solver.simulated_annealing()
    sa_solver.display_optimal_path()
    sa_solver.animateSolutions()
    sa_solver.plot_learning()


if __name__ == "__main__":
    main()
