import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SharedResNet(nn.Module):
    def __init__(self, input_shape, num_res_blocks):
        super(SharedResNet, self).__init__()
        self.conv = nn.Conv2d(input_shape[2], 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResNetBlock(256, 256) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_res_blocks, num_actions):
        super(ActorCritic, self).__init__()
        self.shared_resnet = SharedResNet(input_shape, num_res_blocks)

        # Actor network
        self.actor_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.actor_bn = nn.BatchNorm2d(256)
        self.actor_fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.actor_fc2 = nn.Linear(1024, num_actions)

        # Critic network
        self.critic_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.critic_bn = nn.BatchNorm2d(256)
        self.critic_fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.critic_fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.shared_resnet(x)

        # Actor forward pass
        actor_x = F.relu(self.actor_bn(self.actor_conv(x)))
        actor_x = actor_x.view(actor_x.size(0), -1)
        actor_x = F.relu(self.actor_fc1(actor_x))
        actor_out = F.softmax(self.actor_fc2(actor_x), dim=-1)

        # Critic forward pass
        critic_x = F.relu(self.critic_bn(self.critic_conv(x)))
        critic_x = critic_x.view(critic_x.size(0), -1)
        critic_x = F.relu(self.critic_fc1(critic_x))
        critic_out = F.tanh(self.critic_fc2(critic_x))

        return actor_out, critic_out


