import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import time
import math
import os


START = 'S'  # Start -> nó inicial (canto superior esquerdo)
END = 'E'  # End -> nó final (canto inferior direito)
TERRA = 'T'  # Terra -> movimento para este nó tem custo 1
AGUA = 'A'  # Agua -> movimento para este nó tem custo 3
BARREIRA = 'B'  # Barreira -> can't move here
FRONTEIRA = 'F'  # Fronteira -> can't move here

CSV_ENV_FILE = 'input/example-environment.csv'
MAP_SIZE = 25  # 99

INIT = (0, 0)  # initial state
GOAL = (MAP_SIZE, MAP_SIZE)  # goal state


class Node:
    def __init__(self, name, pos, cost_so_far, heuristic, path):
        self.name = name
        self.pos = pos
        self.cost_so_far = cost_so_far
        self.heuristic = heuristic
        self.path = path
        self.visited = False
        self.done = False

    def __str__(self):
        return f'{self.name} | {self.pos} | {self.cost_so_far} | {self.heuristic} | {self.combined_cost}\n{self.path}'

    def __eq__(self, obj):
        return obj.name == self.name and obj.pos == self.pos

    @property
    def combined_cost(self):
        return self.cost_so_far + self.heuristic


class Node_Structures:
    def __init__(self, init, grid):
        self.counter = 0
        self.done = []
        self.q = [init]
        self.node_map = {init: Node(grid[0][0], init, 0, 0, [init])}
        for (y, line) in enumerate(grid):
            for (x, _) in enumerate(line):
                if (y, x) != init:
                    pos = (y, x)
                    heuristic = calculate_heuristic_distance(pos, GOAL)
                    self.node_map[pos] = Node(
                        grid[y][x], pos, 99999, heuristic, [])

    def get_node_from_queue(self, pos):
        return self.node_map[pos]

    def get_graph_map_list(self, path=[]):
        map = [[''] * (MAP_SIZE + 1) for i in range(MAP_SIZE + 1)]
        for pos in self.node_map:
            if pos in path:
                map[pos[0]][pos[1]] = 'P'
            elif self.node_map[pos].done:
                map[pos[0]][pos[1]] = 'D'
            elif self.node_map[pos].visited:
                map[pos[0]][pos[1]] = 'V'
            else:
                map[pos[0]][pos[1]] = self.node_map[pos].name
        map[MAP_SIZE][MAP_SIZE] = 'E'
        return map


def read_environment_file(file_name):
    with open(file_name) as fp:
        return fp.readlines()


def process_environment_file_info(lines):
    lines_info = []

    for line in lines[1:]:  # ignore 1st line: positions
        cleared_line = line[:-1]  # remove \n
        line_info = cleared_line.split(',')[1:]
        lines_info.append(line_info)

    # print(lines_info)
    return lines_info

# positions: (x, y)


def calculate_heuristic_distance(initial_pos, final_pos):
    horizontal_distance = abs(final_pos[0] - initial_pos[0])
    vertical_distance = abs(final_pos[1] - initial_pos[1])
    return max(horizontal_distance, vertical_distance)


def get_possible_move_positions(h):
    return [(h[0] - 1, h[1] - 1), (h[0] - 1, h[1]), (h[0] - 1, h[1] + 1),
            (h[0], h[1] - 1),                       (h[0], h[1] + 1),
            (h[0] + 1, h[1] - 1), (h[0] + 1, h[1]), (h[0] + 1, h[1] + 1), ]


def visualize_env(map, file_name):
    map_envir = pd.DataFrame(map)
    map_env_num = map_envir.replace(['T', 'F', 'S', 'E', 'A', 'B', 'V', 'D', 'P'], [
                                    0, 1, 2, 2, 2.5, 1, 3, 4, 5])
    f, ax = plt.subplots(figsize=(16, 10))
    ax = sns.heatmap(map_env_num)
    f.savefig(f'output/{file_name}.png')


def a_star_algo(node_structures, goal):
    node_map = node_structures.node_map
    done = node_structures.done
    q = node_structures.q

    if node_structures.counter % 10 == 0:
        visualize_env(node_structures.get_graph_map_list(),
                      f'map_at_{int(node_structures.counter / 10)}')

    h = q[0]
    #print("\nHEAD -> ", node_map[h])
    #print('q = [', end='')
    # for (i, n) in enumerate(q):
    #print(f'{node_map[n].pos} {node_map[n].combined_cost}', end=(', ' if i != len(q) - 1 else ']\n'))
    #print('done = ', done)
    done.append(h)
    node_map[h].done = True
    q.pop(0)

    if h == goal:
        print("GG LETS GOOOOOOOOOOOOOOOOOOOOOOOO")
        print(node_map[h].path)
        visualize_env(node_structures.get_graph_map_list(
            node_map[h].path), 'final_map')
        return

    #print(f'head {h} -> {node_map[h].cost_so_far}')
    possible_move_positions = get_possible_move_positions(h)
    #print(f'possible_move_positions -> {possible_move_positions}')
    for pos in possible_move_positions:

        # Outside map - should check over 100 as well
        if pos[0] < 0 or pos[1] < 0:
            continue

        node = node_map[pos]
        #print(f'{node.pos} actual move -> {node.name not in [FRONTEIRA, BARREIRA] and node.pos not in done}')
        if node.name not in [FRONTEIRA, BARREIRA] and node.pos not in done:
            node.visited = True
            #print(f'{node.name} {node.pos}: {node.cost_so_far} + {node.heuristic} = {node.combined_cost} -> ', end='')
            node.cost_so_far = min(
                node.cost_so_far, node_map[h].cost_so_far + (3 if node.name == AGUA else 1))
            #print(f'{node.cost_so_far} + {node.heuristic} = {node.combined_cost} ')
            node.path = node_map[h].path + [node.pos]
            if node.pos not in q:
                q.append(node.pos)

    node_structures.q = list(set(q))
    node_structures.q.sort(key=lambda x: node_map[x].combined_cost)
    node_structures.counter += 1
    a_star_algo(node_structures, goal)


lines = read_environment_file(CSV_ENV_FILE)
lines_info = process_environment_file_info(lines)
grid = np.array(lines_info, dtype=object)

node_structures = Node_Structures(INIT, grid)
shutil.rmtree('output')
os.makedirs('output')
a_star_algo(node_structures, GOAL)
