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
            

class MovieLens(DoubleAdvBandit):
    def __init__(self) -> None:
        super().__init__()
        path_to_MovieLens = os.path.join(
            os.path.dirname(__file__),
            '../MovieLens/'
        )
        path_to_npy_data = os.path.join(
                path_to_MovieLens,
                'MovieLens_loss.npy'
        )
        try:
            with open(path_to_npy_data, 'rb') as f:
                self.data = np.load(f)
                self.n_epochs, self.n_agents, self.n_arms = self.data.shape
                self.genres = ['Action',
                    'Adventure',
                    'Animation',
                    'Children',
                    'Comedy',
                    'Crime',
                    'Drama',
                    'Fantasy',
                    'Film-Noir',
                    'Horror',
                    'Musical',
                    'Mystery',
                    'Romance',
                    'Sci-Fi',
                    'Thriller',
                    'War',
                    'Western',
                    'Documentary',
                    'IMAX',
                    '(no genres listed)'
                ]
        except FileNotFoundError:
            if query_yes_no("Do you want to generate the loss data for MovieLens?"):
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
                all_df = pd.DataFrame(
                    [{
                        'userId': list(range(1, ratings_gb['userId'].max()+1)),
                        'genres': list(ratings_gb['genres'].unique()),
                        'epoch': list(range(ratings_gb['epoch'].max()+1))
                    }]
                )
                all_df = all_df.explode(['epoch'])
                all_df = all_df.explode(['userId'])
                all_df = all_df.explode(['genres'])
                all_df = pd.merge(
                    all_df,
                    ratings_gb,
                    how='left',
                    on = ['userId', 'genres', 'epoch']
                )
                all_df.fillna(
                    value = (ratings_df['rating'].max() + ratings_df['rating'].min()) / 2,
                    inplace = True
                )
                ratings = all_df['rating'].to_numpy(dtype=np.float32)
                losses = (ratings.max() + ratings.min() - ratings) / ratings.max()
                losses = losses.reshape(
                    ratings_gb['epoch'].max()+1,
                    ratings_gb['userId'].max(),
                    len(ratings_gb['genres'].unique()),
                    order='C'
                )
                self.data = losses
                self.n_epochs, self.n_agents, self.n_arms = losses.shape
                self.genres = all_df.head(20)['genres'].values.tolist()
                if query_yes_no("Do you want to save MovieLens_loss.npy?"):
                    with open(path_to_npy_data, 'wb') as f:
                        np.save(f, self.data)
                    print('Data saved: ' + path_to_npy_data)

    def get_armId(self, genre):
        return self.genres.index(genre)

    def get_genre(self, arm_id):
        return self.genres[arm_id]

    def rating_to_loss(self, rating, r_max=5, r_min=.5):
        return (r_max + r_min - rating) / r_max


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


if __name__ == '__main__':
    train = MovieLens()