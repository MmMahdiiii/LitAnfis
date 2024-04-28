import torch 
import gymnasium as gym
from dqn.DQN import DQN_Agent
import matplotlib.pyplot as plt
from utilities.config import *
import pygame 
from minigrid.wrappers import RGBImgObsWrapper, PositionBonus
from utilities.config import *


class Model_TrainTest:
    def __init__(self, hyperparams):
        
        # Define RL Hyperparameters
        self.train_mode             = hyperparams["train_mode"]
        self.RL_load_path           = hyperparams["RL_load_path"]
        self.save_path              = hyperparams["save_path"]
        self.save_interval          = hyperparams["save_interval"]
        
        self.clip_grad_norm         = hyperparams["clip_grad_norm"]
        self.learning_rate          = hyperparams["learning_rate"]
        self.discount_factor        = hyperparams["discount_factor"]
        self.batch_size             = hyperparams["batch_size"]
        self.update_frequency       = hyperparams["update_frequency"]
        self.max_episodes           = hyperparams["max_episodes"]
        self.max_steps              = hyperparams["max_steps"]
        self.render                 = hyperparams["render"]
        
        self.epsilon_max            = hyperparams["epsilon_max"]
        self.epsilon_min            = hyperparams["epsilon_min"]
        self.epsilon_decay          = hyperparams["epsilon_decay"]
        
        self.memory_capacity        = hyperparams["memory_capacity"]
        
        self.num_states             = hyperparams["num_states"]
        self.map_size               = hyperparams["map_size"]
        self.render_fps             = hyperparams["render_fps"]
                        
        # Define Env
        self.env = gym.make("MiniGrid-UnlockPickup-v0", max_episode_steps=20000, max_steps=5000)
        self.env = PositionBonus(RGBImgObsWrapper(self.env))
        
        # Define the agent class
        self.agent = DQN_Agent(env                = self.env, 
                                epsilon_max       = self.epsilon_max, 
                                epsilon_min       = self.epsilon_min, 
                                epsilon_decay     = self.epsilon_decay,
                                clip_grad_norm    = self.clip_grad_norm,
                                learning_rate     = self.learning_rate,
                                discount          = self.discount_factor,
                                memory_capacity   = self.memory_capacity)
                
        
    def state_preprocess(self, state):
        """
        Convert an state to a tensor and basically it encodes the state into 
        an onehot vector. For example, the return can be something like tensor([0,0,1,0,0]) 
        which could mean agent is at state 2 from total of 5 states.

        """
        image = state['image']
        image = np.transpose(image, axes=(2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        return image
    
    
    def train(self): 
        """                
        Reinforcement learning training loop.
        """
        
        total_steps = 0
        self.reward_history = []
        
        # Training loop over episodes
        for episode in range(1, self.max_episodes+1):
            state, _ = self.env.reset(seed=seed)
            state = self.state_preprocess(state)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            selected_actions = [0,0,0]
            while not done and not truncation:
                action = self.agent.select_action(state)
                selected_actions[action] += 1
                done0, done1, done2 = False, False, False 
                truncation0, truncation1, truncation2 = False, False, False 
                reward0, reward1, reward2= 0, 0, 0
                
                next_state, reward0, done0, truncation0, _ = self.env.step(3)
                if (not done0) and (not truncation0):
                    next_state, reward1, done1, truncation1, _ = self.env.step(5)
                if (not done0) and (not truncation0) and (not done1) and (not truncation1):
                    next_state, reward2, done2, truncation2, _ = self.env.step(action)
                        
                done = done0 or done1 or done2 
                truncation = truncation0 or truncation1 or truncation2    
                reward = reward0 + reward1 + reward2
                
                
                
                next_state = self.state_preprocess(next_state)
                
                self.agent.replay_memory.store(state, action, next_state, reward, done) 
                
                
                state = next_state
                episode_reward += reward
                step_size +=1
                            
            if len(self.agent.replay_memory) > self.batch_size and sum(self.reward_history) > 0:
                self.agent.learn(self.batch_size, (done or truncation))
            
                # Update target-network weights
                if total_steps % self.update_frequency == 0:
                    self.agent.hard_update()
                    
            # Appends for tracking history
            self.reward_history.append(episode_reward) # episode reward                        
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            #-- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            print(selected_actions)

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}, {done=}, {truncation=}")
            print(result)
        self.plot_training(episode)
        self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
        print('\n~~~~~~Interval Save: Model saved.\n')
                                                                    

    def test(self, max_episodes):  
        """                
        Reinforcement learning policy evaluation.
        """
           
        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()
        
        # Testing loop over episodes
        for episode in range(1, max_episodes+1):         
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
                                                           
            while not done and not truncation:
                state = self.state_preprocess(state)
                action = self.agent.select_action(state)
                
                done0, done1, done2 = False, False, False 
                truncation0, truncation1, truncation2 = False, False, False 
                reward0, reward1, reward2= 0, 0, 0
                
                next_state, reward0, done0, truncation0, _ = self.env.step(3)
                if (not done0) and (not truncation0):
                    next_state, reward1, done1, truncation1, _ = self.env.step(5)
                if (not done0) and (not truncation0) and (not done1) and (not truncation1):
                    next_state, reward2, done2, truncation2, _ = self.env.step(action)
                        
                done = done0 or done1 or done2 
                truncation = truncation0 or truncation1 or truncation2    
                reward = reward0 + reward1 + reward2
                   
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print log            
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, {done=}, {truncation=}")
            print(result)
            
        pygame.quit() # close the rendering window
        
    
    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./artifacts/reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        # plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./artifacts/Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        # plt.show()        
        