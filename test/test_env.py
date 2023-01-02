import os, pytest
import numpy as np
import pandas as pd
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


def test_least_cum_loss(n_epochs=1000, n_agents=16, n_arms=50, seed=0):
    train_data = fbe.HomoBandit(
        n_epochs,
        n_agents,
        n_arms,
        np.random.default_rng(seed)
    )
    cum_losses, _ = train_data.cumloss_of_best_arm()
    res = cum_losses[-1]
    assert np.isclose(res, 0, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_movie_lens():
    movie_lens = fbe.MovieLens()
    assert len(movie_lens) == 12800
    assert movie_lens.__getitem__(0).shape == (58**2, 20)
    epoch = 4
    i = 0
    k = movie_lens.get_armId('Action')
    res = movie_lens.data[epoch, i, k] 
    ans = movie_lens.rate2loss(4.5)
    assert np.isclose(res, ans, rtol=1e-05, atol=1e-08, equal_nan=False)

def test_movie_lens_best_arm():
    movie_lens = fbe.MovieLens()
    cum_losses, arm_id = movie_lens.cumloss_of_best_arm()
    assert movie_lens.get_genre(arm_id) == 'Film-Noir'
