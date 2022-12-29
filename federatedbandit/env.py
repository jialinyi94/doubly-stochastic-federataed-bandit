import numpy as np
from torch.utils.data import Dataset

class HomoBandit(Dataset):
    def __init__(self, n_epochs, n_agents, n_arms, rng) -> None:
        super().__init__()
        global_means = np.linspace(0, 1, n_arms)
        L = np.array(
            [rng.binomial(1, p, size=(n_epochs, n_agents)) for p in global_means], 
            dtype=np.float32
        )
        self.data = np.transpose(L, (1, 2, 0))
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

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
            