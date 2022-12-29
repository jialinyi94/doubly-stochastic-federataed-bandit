import numpy as np
import federatedbandit.env as fbe

def test_stochastic_activation_bandit():
    n, k = 15, 10
    train_loader = fbe.StoActBandit(
        n_epochs=120,
        n_agents=30,
        n_arms=k,
        activate_size=n,
        rng=np.random.default_rng(1)
    )
    for loss_matrix in train_loader:
        assert np.sum(loss_matrix) <= n*k

def test_fix_activation_bandit():
    n, k = 15, 10
    train_loader = fbe.FixActBandit(
        n_epochs=120,
        n_agents=30,
        n_arms=k,
        activate_size=n,
        rng=np.random.default_rng(1)
    )
    for loss_matrix in train_loader:
        for i in range(n, 30):
            for j in range(k):
                assert np.sum(loss_matrix[i,j]) == 0


def test_least_cum_loss(n_epochs=2, n_agents=3, n_arms=4, seed=0):
    train_data = fbe.HomoBandit(
        n_epochs,
        n_agents,
        n_arms,
        np.random.default_rng(seed)
    )
    res = train_data.least_cumloss()
    assert np.isclose(res, 0, rtol=1e-05, atol=1e-08, equal_nan=False)