# Vis
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Graph:
    def __init__(self):
        self.graph = {}
        self.weight = {}
        self.heuristic = {}

    def addEdge(self, o, d, w = 1):
        if o not in self.graph:
            self.graph[o] = []
            self.weight[o] = []
            self.heuristic[o] = 100
        if d not in self.graph:
            self.graph[d] = []
            self.weight[d] = []
            self.heuristic[d] = 100
        self.graph[o].append(d)
        self.weight[o].append(w)
        combined = sorted(zip(self.graph[o], self.weight[o]), key=lambda x: x[0])
        self.graph[o], self.weight[o] = map(list, zip(*combined))
        self.graph[d].append(o)
        self.weight[d].append(w)
        combined = sorted(zip(self.graph[d], self.weight[d]), key=lambda x: x[0])
        self.graph[d], self.weight[d] = map(list, zip(*combined))

    def addHeuristics(self, o, h):
        self.heuristic[o] = h

    def __str__(self):
        return f"{self.graph}\n{self.weight}\n{self.heuristic}"

class GraphVisualization:
    def visualize_traversal(self, g, o, d, traversal_algorithm, bw = 1):
        G = nx.Graph()
        for node, neighbors in g.graph.items():
            for neighbor, weight in zip(neighbors, g.weight[node]):
                G.add_edge(node, neighbor, weight=weight)
        if traversal_algorithm.__name__ == "BS":
            paths = traversal_algorithm(g, o, d, bw)
        else:
            paths = traversal_algorithm(g, o, d)
        pos = nx.planar_layout(G)
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            node_labels = {node: f"{node}\nH:{g.heuristic[node]}" for node in G.nodes()}
            nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, node_color='lightblue', font_color='black', font_weight='bold',labels = node_labels, ax=ax)
            edge_labels = {(node, neighbor): G[node][neighbor]['weight'] for node, neighbor in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8, ax=ax)
            if frame < len(paths):
                path = paths[frame]
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)
        ani = FuncAnimation(fig, update, frames=len(paths) + 1, repeat=False, interval=1000)
        plt.show()
    

BMS:
Algorithm:
1. Initialize an empty list called paths to store all the paths found during the search.
2. Initialize a stack data structure with the starting node and a list containing only the starting node.
3. While the stack is not empty, do the following:
    a. Pop the top element (node, path) from the stack.
    b. Append the current path to the paths list.
    c. For each neighbor of the current node in the graph g:
        - If the neighbor is not already in the current path, add it to the stack with an updated path.
4. After the search is complete, print and return the list of all paths found during the search.

Program:
class Algorithm:
    def BMS(self, g, o, d):
        paths = []
        stack = [(o, [o])]
        while stack:
            node, path = stack.pop()
            paths.append(path)
            for neighbor in g.graph[node]:
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
                    print(stack)
        print(paths)
        return paths

g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.BMS)

Algorithm:
1.	Initialize an empty set called visited to keep track of visited nodes.
2.	Initialize a stack data structure with the starting node and a list containing only the starting node.
3.	Initialize an empty list called total_path to store all the paths explored during the search.
4.	While the stack is not empty, do the following:
a. Pop the top element (node, path) from the stack.
b. Append the current path to the total_path list.
c. If the current node is the destination node (d), print the path and return the total_path.
d. If the current node has not been visited yet, add it to the visited set.
e. For each neighbor of the current node in the graph g (sorted in reverse order):
           - If the neighbor has not been visited, add it to the stack with an updated path.
5. If the destination node is not found after the search, return None.

Program:
class Algorithm:
    def DFS(self, g, o, d):
        visited = set()
        stack = [(o, [o])]
        total_path = []
        while stack:
            node, path = stack.pop()
            total_path.append(path)
            if node == d:
                print(path)
                return total_path
            if node not in visited:
                visited.add(node)
                for neighbor in sorted(g.graph[node], reverse=True):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
                        print(stack)
        return None

g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.DFS)

Algorithm:
1. Initialize an empty set called visited to keep track of visited nodes.
2. Initialize a queue data structure with the starting node and a list containing only the starting node.
3. Initialize an empty list called total_path to store all the paths explored during the search.
4. While the queue is not empty, do the following:
    a. Remove the first element (node, path) from the front of the queue.
    b. Append the current path to the total_path list.
    c. If the current node is the destination node (d), print the path and return the total_path.
    d. If the current node has not been visited yet, add it to the visited set.
    e. For each neighbor of the current node in the graph g:
        - If the neighbor has not been visited, add it to the queue with an updated path.
5. If the destination node is not found after the search, return None.

Program:
class Algorithm:
    def BFS(self, g, o, d):
        visited = set()
        queue = [(o, [o])]
        total_path = []
        while queue:
            node, path = queue.pop(0)
            total_path.append(path)
            if node == d:
                print(path)
                return total_path
            if node not in visited:
                visited.add(node)
                for neighbor in g.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
                        print(queue)
        return None
    
g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.BFS)

Algorithm:
1. Initialize a beam with a single element containing the heuristic value of the origin node and a tuple with the origin node and a list containing only the origin node.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. While the beam is not empty, do the following:
    a. Sort the beam based on the heuristic values and select the top paths up to the beam width (bw).
    b. Clear the beam for the next iteration.
    c. For each path in the selected best paths:
        - Append the current path to the total_path list.
        - If the current node is the destination node (d), print the path and return the total_path.
        - For each neighbor of the current node in the graph g, If the neighbor is not already in the current path, calculate its heuristic score and update the beam with the new path.
4. If the destination node is not found after the search, return None.

Program:
class Algorithm:
    def BS(self, g, o, d, bw=2):
        beam = [(g.heuristic[o], (o, [o]))]
        total_path = []
        while beam:
            beam.sort(key=lambda x: x[0])
            best_paths = beam[:bw]
            beam = []
            for misc, (node, path) in best_paths:
                total_path.append(path)
                if node == d:
                    print(path)
                    return total_path
                for neighbor in g.graph[node]:
                    if neighbor not in path:
                        heuristic_score = g.heuristic[neighbor]
                        new_path = path + [neighbor]
                        beam.append((heuristic_score, (neighbor, new_path)))
		    print(total_path)
        return None

g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.BS)

Algorithm:
1. Initialize an empty list called path to store the current path and an empty list called total_path to store all the paths explored during the search.
2. Initialize an empty set called visited to keep track of visited nodes.
3. Set the current node to the origin node (o).
4. While the current node is not the destination node (d), do the following:
    a. Append the current node to the path list.
    b. Add the current node to the visited set.
    c. Retrieve the neighbors of the current node from the graph.
    d. Calculate the heuristic values for each neighbor and select the neighbor with the minimum heuristic value.
    e. If the best neighbor is already visited, return the total_path.
    f. Update the current node to the best neighbor and append the current path to the total_path list.
5. If the destination node is reached, append it to the path and the total_path lists.
6. Print the total_path and path lists and return the total_path.

Program:
class Algorithm:
    def HC(self, g, o, d):
        path = []
        total_path = []
        visited = set()
        node = o
        while node != d:
            path.append(node)
            visited.add(node)
            neighbors = g.graph[node]
            neighbor_heuristics = [g.heuristic[neighbor] for neighbor in neighbors]
            best_neighbor = neighbors[neighbor_heuristics.index(min(neighbor_heuristics))]
            if best_neighbor in visited: return total_path
            node = best_neighbor
            total_path.append(list(path[:]))
        path.append(d)
        total_path.append(list(path[:]))
        print(total_path)
        print(path)
        return total_path
    
g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.HC)

Algorithm:
1. Initialize an empty list called all_paths to store all the paths from the origin to the destination, along with their respective costs.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. Initialize a stack data structure with a tuple containing the origin node, an empty path, and a cost of 0.
4. While the stack is not empty, do the following:
    a. Pop the top element (current, path, cost) from the stack.
    b. Append the current path (including the current node) to the total_path list.
    c. If the current node is the destination node (d), append the current path and its cost to the all_paths list.
    d. Otherwise, for each neighbor of the current node, calculate the cumulative cost and add the neighbor, updated path, and cost to the stack.
5. Sort the all_paths list based on the costs of the paths.
6. Print the total_path and all_paths lists and return the total_path.

Program:
class Algorithm:
    def Oracle(self, g, o, d):
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            total_path.append(path+[current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        all_paths=sorted(all_paths)
  print(total_path)   
  print(all_paths)
        return total_path

g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.Oracle)

Algorithm:
1. Initialize an empty list called all_paths to store all the paths from the origin to the destination, along with their respective costs.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. Initialize a stack data structure with a tuple containing the origin node, an empty path, and a cost of 0.
4. While the stack is not empty, do the following:
   a. Pop the top element (current, path, cost) from the stack.
   b. Append the current path (including the current node) to the total_path list.
   c. If the current node is the destination node (d), append the current path and its cost to the all_paths list.
   d. Otherwise, for each neighbor of the current node, calculate the cumulative cost, including the heuristic value, and add the neighbor, updated path, and cost to the stack.
5. Print the total_path and all_paths lists and return the total_path.

Program:
class Algorithm:
    def OHC(self, g, o, d):
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)

        while stack:
            current, path, cost = stack.pop()
            total_path.append(path+[current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight + g.heuristic[neighbor]))
        print(total_path)
  print(all_paths)
        return total_path

g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.OHC)

Algorithm:
1. Initialize variables best_path and best_cost to track the best path and its corresponding cost, setting the initial cost to infinity.
2. Initialize a priority queue containing the cost, the current node, and an empty path.
3. Initialize an empty list called total_path to store all the paths explored during the search.
4. While the priority queue is not empty, do the following:
    a. Find the index of the element with the minimum cost in the priority queue and pop it.
    b. Append the current path (including the current node) to the total_path list.
    c. If the current node is the destination node (d), update the best_path and best_cost if the current cost is lower than the previous best cost.
    d. Otherwise, for each neighbor of the current node, check if adding the neighbor to the path would not exceed the best_cost. If it does not, add the neighbor, updated cost, and path to the priority queue.
5. Print the total_path, best_path, and best_cost and return the total_path.

Program:
class Algorithm:
    def BB(self, g, o, d):
        best_path = None
        best_cost = float('inf')
        priority_queue = [(0, o, [])]
        total_path = []
        while priority_queue: 
min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        if cost+weight<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))
  print(total_path)
  print(best_path, best_cost)
        return total_path
g = Graph()
algo = Algorithm()
g.addEdge('S','A',3)
g.addEdge('S','B',5)
g.addEdge('A','B',4)
g.addEdge('A','D',3)
g.addEdge('B','C',4)
g.addEdge('C','E',6)
g.addEdge('D','G',5)
g.addHeuristics('A',7.3)
g.addHeuristics('B',6)
g.addHeuristics('C',7.5)
g.addHeuristics('D',5)
g.addHeuristics('G',0)
GraphVisualization().visualize_traversal(g, 'S', 'G', algo.BB)

Algorithm:
1. Initialize variables best_path and best_cost to track the best path and its corresponding cost, setting the initial cost to infinity. Initialize a priority queue (implemented as a list) with a tuple containing the cost, the current node, and an empty path.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. While the priority queue is not empty, do the following:
    a. Find the index of the element with the minimum combined cost and heuristic value in the priority queue and pop it.
    c. Append the current path (including the current node) to the total_path list.
    d. If the current node is the destination node (d), update the best_path and best_cost if the current cost is lower than the previous best cost.
    e. Otherwise, for each neighbor of the current node, check if adding the neighbor to the path along with its heuristic value would not exceed the best_cost. If it does not, add the neighbor, updated cost, and path to the priority queue.
4. Print the total_path, best_path, and best_cost and return the total_path.

Program:
class Algorithm:
    def EH(self, g, o, d):
        best_path = None
        best_cost = float('inf')
        priority_queue = [(0, o, [])]
        total_path = []
        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]: min_index = i
            cost, current, path = priority_queue.pop(min_index)
            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        if cost+weight+g.heuristic[current]<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))
        print(total_path)
  print(best_path, best_cost)
        return total_path

Algorithm:
1. Initialize variables best_path and best_cost to track the best path and its corresponding cost, setting the initial cost to infinity. Initialize a priority queue (implemented as a list) with a tuple containing the cost, the current node, and an empty path.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. Initialize an extended_list dictionary to keep track of visited nodes, setting all nodes initially to False.
4. While the priority queue is not empty, do the following:
    a. Find the index of the element with the minimum cost in the priority queue and pop it.
    b. Append the current path (including the current node) to the total_path list.
    c. If the current node is the destination node (d), update the best_path and best_cost if the current cost is lower than the previous best cost.
    d. Otherwise, for each neighbor of the current node, check if neither the current node nor the neighbor is in the extended list. If the condition is met and the cost is within the best_cost, add the neighbor, updated cost, and path to the priority queue.
5. Mark the current node as visited in the extended_list.
6. Print the total_path, best_path, best_cost, and the keys of the extended_list. And return the total_path.

Program:
class Algorithm:
    def EL(self, g, o, d):
        best_path = None
        best_cost = float('inf')
        priority_queue = [(0, o, [])]
        total_path = []
        extended_list = {node: False for node in g.graph}
        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor]:
                        if cost+weight<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))
            extended_list[current] = True
        print(total_path)
        print(best_path, best_cost)
        print(extended_list.keys())
        return total_path

1. Initialize variables best_path and best_cost to track the best path and its corresponding cost, setting the initial cost to infinity. Initialize a priority queue (implemented as a list) with a tuple containing the cost, the origin node, and an empty path.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. Initialize an extended_list dictionary to keep track of visited nodes, setting all nodes initially to False.
4. While the priority queue is not empty, do the following:
    a. Find the index of the element with the minimum combined cost and heuristic value in the priority queue and pop it.
    b. Create a set of visited nodes from the current path.
    c. Append the current path (including the current node) to the total_path list.
    d. If the current node is the destination node (d), update the best_path and best_cost if the current cost is lower than the previous best cost.
    e. Otherwise, for each neighbor of the current node, check if the current node and the neighbor are not in the extended list and the neighbor is not in the visited set. If the condition is met and the cost is within the best_cost, add the neighbor, updated cost, and path to the priority queue.
5. Mark the current node as visited in the extended_list.
6. Print the total_path, best_path, best_cost, extended_list and return the total_path.

Program:
class Algorithm:
    def Astar(self, g, o, d):
        best_path = None
        best_cost = float('inf')
        priority_queue = [(0, o, [])]
        total_path = []
        extended_list = {node: False for node in g.graph}
        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            visited = set(path)
            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor] and neighbor not in visited:
                        if cost+weight+g.heuristic[current]<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))
                            #print(priority_queue)
            extended_list[current] = True
        print(total_path)
        print(best_path, best_cost)
        print(extended_list.keys())
        return total_path
    
Algorithm:
1. Initialize an open list, a closed list, and a total_path list to manage the nodes, visited nodes, and paths explored during the search, respectively.
2. While the open list is not empty, do the following:
    a. Sort the open list based on the heuristic values and Pop the element with the lowest heuristic value from the open list.
    b. Append the current path (including the current node) to the total_path list.
    c. If the current node is the destination node (d), print the optimal path and return the total_path.
    d. For each neighbor of the current node, calculate the g, h, and f values and update the open list with the new values and path if the neighbor is not in the path or closed list.
3. Add the current node to the closed list to mark it as visited.
4. If no path is found, print "No path found" and return None.

Program:
class Algorithm:
    def AOstar(self, g, o, d):
        open_list = [(g.heuristic[o], o, [])]
        closed_list = []
        total_path = []
        while open_list:
            open_list.sort(key=lambda x: x[0])
            h, current, path = open_list.pop(0)
            total_path.append(path+[current])
            print(total_path)
            if current == d:
                print("Optimal path:", path + [current])
                return total_path
            for neighbor, weight in zip(g.graph[current], g.weight[current]):
                if neighbor not in path and neighbor not in closed_list:
                    g_value = len(path) + weight
                    h_value = g.heuristic[neighbor]
                    f_value = g_value + h_value
                    new_path = path + [current]
                    open_list.append((f_value, neighbor, new_path))
            closed_list.append(current)
        print("No path found")
        return None

Algorithm:
1. Initialize a variable best_path to keep track of the best path found during the search and a priority queue (implemented as a list) with a tuple containing the heuristic value, the origin node, and an empty path.
2. Initialize an empty list called total_path to store all the paths explored during the search.
3. While the priority queue is not empty, do the following:
    a. Find the index of the element with the minimum heuristic value in the priority queue.
    b. Pop the element with the minimum heuristic value from the priority queue.
    c. Append the current path (including the current node) to the total_path list.
    d. If the current node is the destination node (d), update the best_path with the current path, print it, and return the total_path.
    e. Otherwise, for each neighbor of the current node, check if the neighbor is not in the path. If it is not, add the neighbor, updated heuristic value, and path to the priority queue.
4. If the destination node is not found, print the best_path and return the total_path.

Program:
class Algorithm:
    def BestFirstSearch(self, g, o, d):
        best_path = None
        priority_queue = [(g.heuristic[o], o, [])]
        total_path = []
        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            heuristic, current, path = priority_queue.pop(min_index)
            total_path.append(path+[current])
            if current == d:
                best_path = path + [current]
                print(best_path)
                return total_path
            else:
                for neighbor in g.graph[current]:
                    if neighbor not in path:
                        priority_queue.append((g.heuristic[neighbor], neighbor, path + [current]))
                        print(priority_queue)
        print(best_path)
        return total_path
    
