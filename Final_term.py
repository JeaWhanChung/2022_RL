import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import wandb

wandb.init(project="RL_1", entity="jeawhan")

# env = gym.make('SpaceInvaders-v0').unwrapped
env = gym.make('Assault-v0').unwrapped
# env = gym.make('Qbert-v0').unwrapped
# env = gym.make('Pong-v0').unwrapped
# env = gym.make('Breakout-v0').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        self.conv1(x)
        self.bn1(self.conv1(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((110,84), interpolation=Image.CUBIC),
                    T.CenterCrop(84),
                    T.ToTensor()])

def get_screen():
    # env.render(mode='human')
    screen = env.render(mode='rgb_array')#.transpose((2, 0, 1))
    screen = screen[25:200,10:150]
    image_data = cv2.cvtColor(cv2.resize(screen, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data,(84, 84, 1))
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor.unsqueeze(0)



env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

# BATCH_SIZE = config.batch_size
BATCH_SIZE = 19
GAMMA = 0.999
EPS_START = 1
# EPS_END = config.eps_end
EPS_END = 0.3495585029185744
# EPS_DECAY = 0.0001
EPS_DECAY = 10000
TARGET_UPDATE = 10
# EPS_EXPLORE = config.eps_explore
EPS_EXPLORE = 3656191

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(),lr=0.7032015923608359)
memory = ReplayMemory(95559)


steps_done = 0
eps_threshold = EPS_START

def select_action(state,i_episode):
    global steps_done
    global eps_threshold
    sample = random.random()

    if eps_threshold > EPS_END:
            eps_threshold -= (EPS_START - EPS_END) / EPS_EXPLORE
    steps_done += 1
    random_action = random.random() <= eps_threshold

    if random_action:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        return policy_net(state).max(1)[1].view(1, 1)


episode_score_average = 0

def optimize_model(i_episode):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    wandb.log({"loss": loss})

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # wandb.watch(policy_net)

num_episodes = 200
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen().to(device).squeeze(0)
    state = torch.stack([last_screen,last_screen,last_screen,last_screen],dim=1)
    next_state = torch.zeros_like(state)
    get_score = 0
    for t in count():
        action = select_action(state,i_episode)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        get_score = get_score + reward.item()

        current_screen = get_screen().to(device)
        if not done:
            next_state[:,0:2,:,:] = state[:,1:3,:,:].clone()
            next_state[:,3,:,:] = current_screen
            next_state = next_state

        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model(t)
        if done:
            wandb.log({"get_score": get_score})
            wandb.log({"Iteration": t})

            # episode_score.append(get_score)
            break
    # wandb.watch(target_net)
    wandb.log({"episode": i_episode})
    episode_score_average = (episode_score_average * (i_episode)+get_score)/(i_episode+1)
    wandb.log({"Score_Average": episode_score_average})

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode > 99:
        if i_episode % 100 == 0:
            torch.save(policy_net, "./trained_model_JJH/current_model_" + str(i_episode) + ".pth")