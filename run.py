import wandb
import torch
import numpy as np
import networkx as nx
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

    # Specify communcation network
    if config['network'] == 'COMPLETE':
        graph = nx.complete_graph(config['n_agents'])
    elif config['network'] == 'NONE':
        graph = nx.from_numpy_array(
            np.zeros([
                config['n_agents'], config['n_agents']
            ])
        )
    else:
        raise NotImplementedError("The "+config['network']+" network has not been implemented.")
    comm_net = fba.CommNet(graph)

    # Specify the gossip
    if config['gossip'] == 'MaxDegree':
        gossip_numpy = comm_net.max_deg_gossip()
    else:
        raise NotImplementedError("The "+config['gossip']+" mechanism has not been implemented.")
    gossip = torch.tensor(gossip_numpy, device=config['device'])

    # Create FedExp3
    agent = fba.FedExp3(
        config['n_agents'],
        config['n_arms'],
        gossip, 
        config['lr'],
        expr_scheduler=fba.cube_root_scheduler(config['gamma']),
        device=config['device']
    )

    
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
        network = 'NONE',
        gossip = 'MaxDegree',
        n_agents = 10,
        n_arms = 50,                 
        horizon = 4000,                  
        lr = .1,
        gamma = 0.01,
        seed = 1,
        WANDB = True
    )

    main(config)