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

def test_max_degree_large_graph():
    n = 1000
    r = np.sqrt(np.log(n) ** 1.1 / n)
    graph = nx.random_geometric_graph(
        n, r
    )
    comm_net = fba.CommNet(graph)
    assert max([d for i, d in graph.degree()]) <= 2*np.sqrt(np.log(n) ** 1.1 * n)
    P, gap= comm_net.max_deg_gossip(spectral_gap=True)
    assert gap > 0
    i_row = np.random.randint(0, P.shape[0]-1)
    res = np.sum(P, axis=1)[i_row]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    j_col = np.random.randint(0, P.shape[1]-1)
    res = np.sum(P, axis=0)[j_col]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)

def test_p2p_gossip():
    n = 10
    graph = nx.grid_graph(
        [n, n]
    )
    comm_net = fba.CommNet(graph)
    P, gap= comm_net.p2p_gossip(spectral_gap=True)
    assert gap > 0
    i_row = np.random.randint(0, P.shape[0]-1)
    res = np.sum(P, axis=1)[i_row]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    j_col = np.random.randint(0, P.shape[1]-1)
    res = np.sum(P, axis=0)[j_col]
    assert np.isclose(res, 1, rtol=1e-05, atol=1e-08, equal_nan=False)

def test_fedexp3_ub_exact():
    n_epochs = 10
    n_agents = 10
    n_arms = 3
    spectral_gap = 1/.4
    lr_array = [.1] * n_epochs
    gamma_array = [.01 * 1/(t+1)**(1/3)  for t in range(n_epochs)]
    cumreg_generator = fba.fedexp3_ub_exact(
        n_epochs,
        n_agents,
        n_arms,
        spectral_gap,
        lr_array,
        gamma_array
    )
    curr = 0
    for _ in range(n_epochs):
        prev = curr
        curr = next(cumreg_generator)
        assert curr >= prev