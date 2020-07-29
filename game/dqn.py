import torch
import torch.nn.optimizer as optim
import torch.nn.functional as F
from neural_net.pytorch_ann import NeuralNetwork


class DQN:
    def __init__(self, input_size, nb_actions, gamma):
        self.model = NeuralNetwork(input_size, nb_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # TODO check this
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        temperature = 75
        probs = F.softmax(self.model(state) * temperature)
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state)
            .gather(1, batch_action.unsqueeze(1))
            .squeeze(1)
        )
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    # TODO
    def update(self, reward, new_signal):
        raise NotImplementedError

    # TODO
    def score(self):
        raise NotImplementedError

    # TODO
    def save(self):
        raise NotImplementedError

    # TODO
    def load(self):
        raise NotImplementedError

