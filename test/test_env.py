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