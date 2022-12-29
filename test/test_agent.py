import numpy as np
import networkx as nx
import federatedbandit.agent as fba

def test_fast_gossip():
    g = nx.from_dict_of_lists({
        0: [1,],
        1: [0, 2,],
        2: [1, 3,],
        3: [2, ]
    })
    comm_net = fba.CommNet(g)
    P, gap = comm_net.fast_gossip('SDP', spectral_gap=True)
    assert gap > 0
    i_row = np.random.randint(0, P.shape[0]-1)
    res = np.sum(P, axis=1)[i_row]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    j_col = np.random.randint(0, P.shape[1]-1)
    res = np.sum(P, axis=0)[j_col]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)

def test_fast_gossip_large_graph():
    n = 50
    r = np.sqrt(np.log(n) ** 1.1 / n)
    graph = nx.random_geometric_graph(
        n, r
    )
    comm_net = fba.CommNet(graph)
    assert max([d for i, d in graph.degree()]) <= 2*np.sqrt(np.log(n) ** 1.1 * n)
    P, gap= comm_net.fast_gossip('SDP', spectral_gap=True)
    assert gap > 0
    i_row = np.random.randint(0, P.shape[0]-1)
    res = np.sum(P, axis=1)[i_row]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    j_col = np.random.randint(0, P.shape[1]-1)
    res = np.sum(P, axis=0)[j_col]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)