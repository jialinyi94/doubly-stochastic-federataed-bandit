import torch
import numpy as np
import networkx as nx
col_softmax = torch.nn.Softmax(dim=1)

class FedExp3:
    def __init__(self, n_agents, n_arms, W, lr, expr_scheduler, device) -> None:
        self.Z = torch.zeros([n_agents, n_arms], device=device)
        self.W = W
        self.lr = lr
        self.expr_scheduler = expr_scheduler
        self.X0 = torch.ones([n_agents, n_arms], device=device) / n_arms
        self.device = device

    def action(self, rng):
        K = self.Z.shape[-1]
        X = col_softmax(-self.lr*self.Z)
        gamma = next(self.expr_scheduler)
        P = (1 - gamma) * X + gamma * self.X0
        A = torch.multinomial(P, num_samples=1, generator=rng)
        A_one_hot = torch.nn.functional.one_hot(A, num_classes=K).squeeze(1)
        return A_one_hot, P

    def update(self, loss_matrix, actions, probs):
        L_t = loss_matrix.to(self.device)
        G = L_t * actions / probs
        self.Z = torch.mm(self.W.float(), self.Z.float()) + G

class CommNet:
    def __init__(self, nx_graph) -> None:
        self.comm_net = nx_graph
    
    def max_deg_gossip(self):
        degrees = [val for (node, val) in self.comm_net.degree()]
        max_deg = max(degrees)
        D = np.diag(degrees)
        A = nx.to_numpy_matrix(self.comm_net)
        P = np.eye(len(degrees)) - (D - A) / (max_deg+1)
        return P


def cube_root_scheduler(gamma=0.01):
    '''Generates a series of exploration ratios'''
    step = 1
    while True:
        yield gamma / step ** (1/3) 
        step += 1


if __name__ == "__main__":
    # no communication
    N = 4
    adj = np.zeros([N, N])
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())

    # complete graph
    adj = np.ones([N, N])
    for i in range(N):
        adj[i][i] = 0
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())
    
    # sqrt(N)-by-sqrt(N) grid
    g = CommNet(nx.grid_graph([
        int(np.sqrt(N)),
        int(np.sqrt(N))
    ]))
    print(g.max_deg_gossip())
    
    # random geometric graphs with connectivity radius
    # r = 2 sqrt(log(n)^2 / n)
    r = np.sqrt(np.log(N) ** 1.1 / N)
    g = CommNet(nx.random_geometric_graph(
        N, r
    ))
    print(g.max_deg_gossip())

    