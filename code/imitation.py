import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import time

from utilities.dataset import ChessDataset

import gym
import chess
import chess.engine
import gym_chess

import numpy as np

from network import ActorCritic

class Imitaition:
    
    def __init__(self, conf) -> None:
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        
        self.env = gym.make('ChessAlphaZero-v0')
        
        # Define the input shape, number of residual blocks, and number of actions
        input_shape = (8, 8, 119)
        num_res_blocks = conf['num_res_blocks']
        num_actions = self.env.action_space.n

        # Create the Actor-Critic model
        
        self.model = ActorCritic(input_shape, num_res_blocks, num_actions).to(device)
        
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=conf['actor_lr'])
        self.critic_optimizer = optim.Adam(self.model.parameters(), lr=conf['critic_lr'])
        
        self.pgn_dir = conf['pgn_dir']
        
        self.game_buffer_size = conf['game_buffer_size'] 
        
        self.data_loader = ChessDataset(self.pgn_dir, self.game_buffer_size)
        
        self.actor_loss_history = 0
        self.critic_loss_history = 0
        
        self.time_limit = conf['time_limit']
        self.batch_size = conf['batch_size']
        self.K_epochs = conf['epochs']
        
        self.save_pth = conf['save_pth']
        
        self.landa = conf['landa']
        
        self.MseLoss = nn.MSELoss()
        self.CeLoss = nn.CrossEntropyLoss()
        
        self.loss_report_rate = conf['loss_report_rate']
        
    def train(self):
        end_time = (time.time() / 3600) + self.time_limit
        
        totoal_epoch = 0
        
        while (time.time() / 3600) < end_time:
            self.data_loader.read_data()
            print('loading games...')
            states, actions, results = self.data_loader.full_sample() 
            
            # Assuming states, actions, and results are already tensors
            states = np.transpose(states, axes=(0, 3, 1, 2))
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            results = torch.from_numpy(results).float().to(self.device)

            dataset = TensorDataset(states, actions, results)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Example iteration through the DataLoader
            
            self.model.train()
            for epoch in range(self.K_epochs):
                for batch in dataloader:
                    batch_states, batch_actions, batch_results = batch

                    win = batch_results > -0.5
                    
                    batch_actions = batch_actions[win]

                    if len (batch_actions) > 0:
                        actor_out, _ = self.model(batch_states[win])

                        ac_loss = self.CeLoss(actor_out, batch_actions)
                        self.actor_loss_history = ac_loss.item() + self.landa * self.actor_loss_history

                        self.actor_optimizer.zero_grad() # Zero the gradients
                        ac_loss.backward(retain_graph=True)
                        self.actor_optimizer.step()

                    _ , critic_out = self.model(batch_states)

                    cr_loss = self.MseLoss(critic_out.squeeze(), batch_results)
                    self.critic_loss_history = cr_loss.item() + self.landa * self.critic_loss_history

                    self.critic_optimizer.zero_grad() # Zero the gradients
                    cr_loss.backward()
                    self.critic_optimizer.step()
                    
                totoal_epoch += 1    
                avrage_cr_loss = (1 - self.landa) * self.critic_loss_history
                avrage_ac_loss = (1 - self.landa) * self.actor_loss_history
                
                if totoal_epoch % self.loss_report_rate == 0: 
                    print(f'Epoch {totoal_epoch+1}, Actor Loss: {avrage_ac_loss:.4f}, Critic Loss: {avrage_cr_loss:.4f}')

                # Save the model
            torch.save(self.model.state_dict(), self.save_pth)


            
        
        
        
