import os, sys
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class DoubleAdvBandit(Dataset):
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

    def cumloss_of_best_arm(self):
        true_loss = np.mean(self.data, axis=1)
        cum_losses = np.cumsum(true_loss, axis=0)
        best_arm = np.argmin(cum_losses[-1,])
        return cum_losses[:,best_arm]

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
        self.genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
            'Documentary', '(no genres listed)'
        ]
        start = '1996-03-29'
        end = '2018-09-24'
        self.start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
        self.end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
        delta = self.end_date - self.start_date
        self.n_epochs = delta.days + 1
        self.n_agents = 610
        self.n_arms = len(self.genres)
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
        except FileNotFoundError:
            if query_yes_no("Do you want to generate the loss data for MovieLens?"):
                pkldata = pd.read_pickle(
                    os.path.join(
                        os.path.dirname(__file__),
                        '../MovieLens/MovieLens_loss.pkl'
                    )
                )
                self.data = np.zeros(
                    (
                        self.n_epochs,
                        self.n_agents,
                        self.n_arms
                    ),
                    dtype=np.float32
                )
                for index in tqdm(range(self.n_epochs)):
                    today = self.start_date + datetime.timedelta(days=index)
                    for i in range(self.n_agents):
                        for j, genre in enumerate(self.genres):
                            key = (str(today), i+1, genre)
                            if key in pkldata.index:
                                self.data[index, i, j] = pkldata['loss'][key]
                if query_yes_no("Do you want to save MovieLens_loss.npy?"):
                    with open(path_to_npy_data, 'wb') as f:
                        np.save(f, self.data)
                    print('Data saved: ' + path_to_npy_data)

    def get_loss_by_key(self, key):
        date_str, u_id, genre = key
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        delta = date - self.start_date
        return self.data[delta.days, u_id-1, self.get_armId(genre)]

    def get_armId(self, genre):
        return self.genres.index(genre)

                


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