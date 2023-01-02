import os, sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DoubleAdvBandit(Dataset):
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

    def cumloss_of_best_arm(self):
        true_loss = np.mean(self.data, axis=1)
        cum_losses = np.cumsum(true_loss, axis=0)
        best_arm = np.argmin(cum_losses[-1,])
        return cum_losses[:,best_arm], best_arm

class HomoBandit(DoubleAdvBandit):
    def __init__(self, n_epochs, n_agents, n_arms, rng) -> None:
        super().__init__()
        global_means = np.linspace(0, 1, n_arms)
        L = np.array(
            [rng.binomial(1, p, size=(n_epochs, n_agents)) for p in global_means], 
            dtype=np.float32
        )
        self.data = np.transpose(L, (1, 2, 0))

        
class StoActBandit(HomoBandit):
    def __init__(self, n_epochs, n_agents, n_arms, activate_size, rng) -> None:
        super().__init__(n_epochs, n_agents, n_arms, rng)
        for t in range(n_epochs):
            non_selected_idx = rng.choice(
                n_agents,
                size=n_agents - activate_size,
                replace=False
            )
            self.data[t,non_selected_idx,:] = 0


class FixActBandit(HomoBandit):
    def __init__(self, n_epochs, n_agents, n_arms, activate_size, rng) -> None:
        super().__init__(n_epochs, n_agents, n_arms, rng)
        self.data[:,activate_size:,:] = 0
            

class RealData(DoubleAdvBandit):
    def __init__(self, file, rate_min, rate_max, genres) -> None:
        super().__init__()
        try:
            with open(file, 'rb') as f:
                self.data = np.load(f)
                self.n_epochs, self.n_agents, self.n_arms = self.data.shape
                self.genres = genres
                self.r_min, self.r_max = rate_min, rate_max
        except FileNotFoundError:
            print('Please follow the colab notebook to download the dataset.')

    def get_armId(self, genre):
        return self.genres.index(genre)

    def get_genre(self, arm_id):
        return self.genres[arm_id]

    def rate2loss(self, r):
        return (self.r_max + self.r_min - r) / self.r_max


class MovieLens(RealData):
    def __init__(self) -> None:
        super().__init__(
            file=os.path.join(
                os.path.dirname(__file__),
                '../MovieLens/movielens.npy'
            ),
            rate_min=.5, 
            rate_max=5, 
            genres = ['(no genres listed)',
                'Action',
                'Adventure',
                'Animation',
                'Children',
                'Comedy',
                'Crime',
                'Documentary',
                'Drama',
                'Fantasy',
                'Film-Noir',
                'Horror',
                'IMAX',
                'Musical',
                'Mystery',
                'Romance',
                'Sci-Fi',
                'Thriller',
                'War',
                'Western'
            ]
        )

# class AmazonReview(RealData):
#     def __init__(self) -> None:
#         super().__init__(
#             file=os.path.join(
#                 os.path.dirname(__file__),
#                 '../Amazon/amazonreviews.npy'
#             ),
#             rate_min=.5, 
#             rate_max=5, 
#             genres = []
#         )


if __name__ == '__main__':
    train = MovieLens()