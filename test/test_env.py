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
    assert len(movie_lens) == 1309
    assert movie_lens.__getitem__(0).shape == (610, 20)
    epoch = 4
    i = 1 - 1
    k = movie_lens.get_armId('Action')
    res = movie_lens.data[epoch, i, k] 
    ans = movie_lens.rating_to_loss(4.0)
    assert np.isclose(res, ans, rtol=1e-05, atol=1e-08, equal_nan=False)
    fitted_loss = movie_lens.rating_to_loss(2.75)
    assert len(np.where(movie_lens.data != fitted_loss)[0]) == 274480


def test_movie_lens_random_access(excution_number = 10):
    movie_lens = fbe.MovieLens()
    path_to_MovieLens = os.path.join(
        os.path.dirname(__file__),
        '../MovieLens/'
    )
    movies_df = pd.read_csv(
        os.path.join(
            path_to_MovieLens,
            'movies.csv'
        )
    )
    ratings_df = pd.read_csv(
        os.path.join(
            path_to_MovieLens,
            'ratings.csv'
        )
    )
    ratings_df = pd.merge(
        ratings_df,
        movies_df,
        on = 'movieId'
    )
    ratings_df['genres'] = ratings_df['genres'].apply(
        lambda row: row.split('|')
    )
    ratings_df = ratings_df.explode('genres').sort_values(['timestamp', 'userId', 'genres'])
    ratings_gb = ratings_df.groupby(['userId', 'genres'])['rating'].aggregate(list).reset_index()
    ratings_gb['rating'] = ratings_gb['rating'].apply(
        lambda row: [(v, i) for i, v in enumerate(row)]
    )
    ratings_gb = ratings_gb.explode('rating')
    ratings_gb['epoch'] = ratings_gb['rating'].apply(
        lambda row: row[-1]
    )
    ratings_gb['rating'] = ratings_gb['rating'].apply(
        lambda row: row[0]
    )
    ratings_gb.reset_index(inplace=True)
    for _ in range(excution_number):
        idx = np.random.randint(0, ratings_gb.shape[0])
        epoch = ratings_gb['epoch'][idx]
        i = ratings_gb['userId'][idx] - 1
        k = movie_lens.get_armId(ratings_gb['genres'][idx])
        res = movie_lens.data[epoch, i, k]
        ans = movie_lens.rating_to_loss(ratings_gb['rating'][idx])
        assert np.isclose(res, ans, rtol=1e-05, atol=1e-08, equal_nan=False)

def test_movie_lens_best_arm():
    movie_lens = fbe.MovieLens()
    cum_losses, arm_id = movie_lens.cumloss_of_best_arm()
    assert movie_lens.get_genre(arm_id) == 'Drama'
