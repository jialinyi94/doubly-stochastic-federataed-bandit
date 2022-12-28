import wandb
import torch
import numpy as np
import federatedbandit.agent as fba
import federatedbandit.env as fbe
from PIL import Image
from tqdm import tqdm
from matplotlib import cm
from torch.utils.data import DataLoader


def main(config):
    use_cuda = torch.cuda.is_available()
    config['device'] = torch.device("cuda" if use_cuda else "cpu")
    rng = torch.Generator(device=config['device'])
    rng.manual_seed(config['seed'])

    # Create dataset
    env = config['env'].split('-')[0]
    if env == "HomoBandit":
        train_loader = DataLoader(
            fbe.HomoBandit(
                config['horizon'], 
                config['n_agents'], 
                config['n_arms'],
                np.random.default_rng(
                    int(config['env'].split('-')[1]) # seed of the loss tensor
                )
            ),
            batch_size=1, shuffle=False
        )
    else:
        raise NotImplementedError("The "+env+" environment has not been implemented.")

    # Create FedExp3
    if config['gossip'] == "COMPLETE":
        agent = fba.FedExp3(
            config['n_agents'],
            config['n_arms'],
            torch.ones( # complete random communication
                [config['n_agents'], config['n_agents']], device=config['device']
            ) / config['n_agents'], 
            config['lr'],
            expr_scheduler=fba.cube_root_scheduler(config['gamma']),
            device=config['device']
            )
    elif config['gossip'] == "NONE":
        agent = fba.FedExp3(
            config['n_agents'],
            config['n_arms'],
            torch.eye( # no communication
                config['n_agents'], device=config['device']
            ), 
            config['lr'],
            expr_scheduler=fba.cube_root_scheduler(config['gamma']),
            device=config['device']
            )
    else:
        raise NotImplementedError("The "+env+" mechanism has not been implemented.")

    
    # Initialize WANDB
    if config['WANDB']:
        wandb.init(
            project=config['proj'], reinit=True, config=config
        )
        prob_imgs = []

    cumu_loss = 0
    rounds = len(train_loader)
    for i, loss_matrix in tqdm(enumerate(train_loader), total=rounds):
        L_t = torch.squeeze(loss_matrix, 0).to(config['device'])
        # make actions
        actions, probs = agent.action(rng)
        # compute cumulative losses
        cumu_loss += torch.matmul(
            torch.mean(L_t, dim=0),
            torch.transpose(actions.float(), 1, 0)
        )
        # update
        agent.update(L_t, actions, probs)

        # logging
        if config['WANDB']:
            wandb.log({
                'mean': torch.mean(cumu_loss).item(),
                'max': torch.max(cumu_loss).item(),
            })
            if i % (config['horizon'] // 10) == 0:
                prob_imgs.append(
                    wandb.Image(
                        Image.fromarray(
                            np.uint8(cm.viridis(probs.tolist())*255)
                        )
                    )
                )

    if config['WANDB']:
        wandb.log({"visual_probs": prob_imgs})
        wandb.finish()

if __name__ == "__main__":
    config = dict(
        proj = 'FedExp3',
        env = 'HomoBandit-0',
        gossip = 'COMPLETE',
        n_agents = 10,
        n_arms = 50,                 
        horizon = 4000,                  
        lr = .1,
        gamma = 0.01,
        seed = 0,
        WANDB = False
    )

    main(config)