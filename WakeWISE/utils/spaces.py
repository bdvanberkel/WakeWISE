import numpy as np
from gymnasium.spaces import Box, Dict, MultiDiscrete


class CustomGraph(Dict):

    def __init__(self, num_nodes, num_edges, node_space, edge_space):

        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_space = node_space
        self.edge_space = edge_space

        self.spaces['node_features'] = Box(low=np.vstack([[self.node_space.low] for _ in range(num_nodes)]), high=np.vstack([[self.node_space.high] for _ in range(num_nodes)]), shape=(num_nodes, node_space.shape[0]), dtype=np.float32)
        self.spaces['edge_features'] = Box(low=np.vstack([[self.edge_space.low] for _ in range(num_edges)]), high=np.vstack([[self.edge_space.high] for _ in range(num_edges)]), shape=(num_edges, edge_space.shape[0]), dtype=np.float32)
        self.spaces['edge_links'] = MultiDiscrete(np.ones((2, num_edges), dtype=np.int32) * num_nodes)

if __name__ == '__main__':
    node_space = Box(low=-1., high=+1., shape=(3,), dtype=np.float32)
    edge_space = Box(low=-1., high=+1., shape=(2,), dtype=np.float32)
    graph_space = CustomGraph(3, 2, node_space, edge_space)
    print(type(graph_space) == CustomGraph)