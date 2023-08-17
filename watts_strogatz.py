import networkx as nx 
import random 
import matplotlib.pyplot as plt
import numpy as np


class WattsStrogatz:
    def __init__(self, n,q, probability):
        self.g = nx.Graph()
        self.n = n
        self.q = q
        self.probability = probability

        for i in range(self.n):
            for j in range(1, int(self.q/2 + 1)):
                self.g.add_edge(i, (i+j)%self.n)
                self.g.add_edge(i, (i-j)%self.n)
        
        self.pos = nx.spring_layout(self.g)


    def visualize(self):
        plt.cla()
        nx.draw(self.g, pos = self.pos, with_labels = True)

    def update(self):
        edgs = self.g.edges()

        for u,v in edgs:
            nds = list(self.g.nodes())

            if random.random() <= self.probability:
                self.g.remove_edge(u, v)
                
                #ensure no self-connection
                nds.remove(u)
                #ensure you don't choose an existing connection
                for j in self.g.neighbors(u):
                    try:
                        nds.remove(j)
                    except:
                        continue

                #choose a new node 
                new_node = random.choice(nds)
                self.g.add_edge(u, new_node)

        return nx.adjacency_matrix(self.g, nodelist = range(8))

# WS = WattsStrogatz(n = 8, q = 2, probability = .20)

# for i in range(10):
#     connectivity = WS.update()
#     WS.visualize()
#     plt.show()
    # print(np.sum(connectivity, axis = 1))


