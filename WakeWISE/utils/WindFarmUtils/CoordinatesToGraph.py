import math

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import (
    Cartesian,
    Delaunay,
    FaceToEdge,
    KNNGraph,
    LocalCartesian,
    Polar,
    RadiusGraph,
)
from torch_geometric.utils import dense_to_sparse


def to_graph(points: np.array, connectivity: str, min_dist=None, constant=30, add_edge='polar'):
    '''
    Converts np.array to torch_geometric.data.data.Data

    ...

    Arguments:
    ----------
    points : np.array
        a [num_turbines, 2] array containing the coordinates of the turbines.
    connectivity : str
        string with the accepted values \'delaunay\', \'knn\' and \'radial\'.
    min_dist : (float, optional)
        float number stating the minimal distance between turbines (default None). Required for connectivity = \'radial\'.
    constant : (float, optional)
        float constant to be multiplied by the minimal distance defining a radius (default 30). Required for connectivity = \'radial\'.
    add_edge : (str, optional)
        str specifing if \'polar\', \'cartesian\' or \'local cartesian\' coordinates of the nodes should be added to the edges (default \'polar\').

    Returns
    -------
    torch_geometric.data.data.Data
        a torch_geometric.data.data.Data graph.

    Raises
    ------
    ValueError
        If connectivity string isn't \'delaunay\', \'knn\' or \'radial\'.
        If minimal distance isn't defined for radial connectivity.
    '''

    assert (points.shape[1] == 2)
    t = torch.Tensor(points)
    x = Data(pos=t)
    if connectivity.casefold() == 'delaunay':
        d = Delaunay()
        e = FaceToEdge()
        g = e(d(x))
    elif connectivity.casefold() == 'knn':
        kv = math.ceil(np.sqrt(len(points)))
        knn = KNNGraph(k=kv)
        g = knn(x)
    elif connectivity.casefold() == 'radial':
        if (min_dist is not None):
            radius = min_dist * constant
            r = RadiusGraph(r=radius)
            g = r(x)
        else:
            raise ValueError('Minimal distance between turbines is required.')
    elif connectivity.casefold() == 'fully_connected':
        adj = torch.ones(t.shape[0], t.shape[0])
        g = Data(pos=t, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
    else:
        raise ValueError('Please define the connectivity scheme (available types: : \'delaunay\', \'knn\', \'radial\', , \'fully_connected\')')

    if add_edge == 'polar'.casefold():
        p = Polar(norm=False)
        g = p(g)
    elif add_edge == 'cartesian'.casefold():
        c = Cartesian(norm=False)
        g = c(g)
    elif add_edge == 'local cartesian'.casefold():
        lc = LocalCartesian(norm=False)
        g = lc(g)
    else:
        raise ValueError(
            'Please select a coordinate system that is supported (available types: : \'polar\', \'cartesian\' or \'local cartesian\')')
    return g