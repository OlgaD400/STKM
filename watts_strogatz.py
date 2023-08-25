""" Create Watts Strogatz Dynamic Graph """
import random
import networkx as nx
import matplotlib.pyplot as plt

class WattsStrogatz:
    """ Watts Strogatz Dynamic Graph """
    def __init__(self, num_nodes: int ,num_neighbors: int, probability: float):
        """ Initialize the Watts Strogatz dynamic graph.
        
        Args:
            num_nodes: number of nodes in graph
            num_neighbors: Each node is connected to num_neighbors//2 neighbors
                both clockwise and counterclockwise
            probability: probability of an edge being moved at each time step
        """
        self.graph = nx.Graph()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.probability = probability

        for i in range(self.num_nodes):
            for j in range(1, int(self.num_neighbors/2 + 1)):
                self.graph.add_edge(i, (i+j)%self.num_nodes)
                self.graph.add_edge(i, (i-j)%self.num_nodes)
        self.pos = nx.spring_layout(self.graph)


    def visualize(self):
        """ Visualize one time step of the dyanmic graph."""
        plt.cla()
        nx.draw(self.graph, pos = self.pos, with_labels = True)

    def update(self):
        """ Update graph by randomly moving edges. """
        edgs = self.graph.edges()

        for node_1, node_2 in edgs:
            nds = list(self.graph.nodes())

            if random.random() <= self.probability:
                self.graph.remove_edge(node_1, node_2)
                #ensure no self-connection
                nds.remove(node_1)
                #ensure you don't choose an existing connection
                for j in self.graph.neighbors(node_1):
                    try:
                        nds.remove(j)
                    except:
                        continue

                #choose a new node
                new_node = random.choice(nds)
                self.graph.add_edge(node_1, new_node)

        return nx.adjacency_matrix(self.graph, nodelist = range(self.num_nodes))

# WS = WattsStrogatz(n = 8, q = 2, probability = .20)

# for i in range(10):
#     connectivity = WS.update()
#     WS.visualize()
#     plt.show()
    # print(np.sum(connectivity, axis = 1))
