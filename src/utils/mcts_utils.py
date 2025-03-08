import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

def mcts_search(graph, start_nodes, target_nodes, max_iterations=1000, exploration_weight=1.0):
    paths = []
    for start in start_nodes:
        for target in target_nodes:
            root = MCTSNode(start)
            for _ in range(max_iterations):
                node = select_node(root, exploration_weight)
                if not node.children and node.state != target:
                    expand_node(node, graph)
                result = simulate(node.state, target, graph)
                backpropagate(node, result)
            best_path = get_best_path(root, target, graph)
            if best_path:
                paths.append(best_path)
    return paths

def select_node(node, exploration_weight):
    while node.children:
        if not all(child.visits > 0 for child in node.children):
            return next(child for child in node.children if child.visits == 0)
        node = max(node.children, key=lambda n: ucb_score(n, exploration_weight))
    return node

def ucb_score(node, exploration_weight):
    return node.value/node.visits + exploration_weight * math.sqrt(math.log(node.parent.visits)/node.visits)

def expand_node(node, graph):
    try:
        if node.state not in graph:
            return  # 如果当前节点不在图中，直接返回
        neighbors = list(graph.neighbors(node.state))
        for neighbor in neighbors:
            if neighbor in graph:  # 确保邻居节点在图中
                rel = graph[node.state][neighbor]["relation"]
                child = MCTSNode(neighbor, parent=node, action=(node.state, rel, neighbor))
                node.children.append(child)
    except:
        pass

def simulate(state, target, graph, max_steps=50):
    current = state
    steps = 0
    while current != target and steps < max_steps:
        if current not in graph:
            return 0.0  # 如果当前节点不在图中，返回0分
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            return 0.0
        current = random.choice(neighbors)
        steps += 1
    return 1.0 if current == target else 0.0

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.value += result
        node = node.parent

def get_best_path(root, target, graph):
    path = []
    node = root
    while node.children and node.state != target:
        node = max(node.children, key=lambda n: n.visits)
        if node.action:
            path.append(node.action)
    return path if path and node.state == target else None