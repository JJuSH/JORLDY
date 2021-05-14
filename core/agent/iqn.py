import torch
import torch.nn.functional as F
import numpy as np
import copy

from core.network import Network
from core.optimizer import Optimizer
from .dqn import DQNAgent

class IQNAgent(DQNAgent):
    def __init__(self,
                state_size,
                action_size,
                network='iqn',
                optimizer='adam',
                learning_rate=3e-4,
                opt_eps=1e-8,
                num_sample=64,
                embedding_dim=64,
                sample_min=0.0,
                sample_max=1.0,
                **kwargs,
                ):
        super(IQNAgent, self).__init__(state_size, action_size, network=network, **kwargs)
        
        self.network = Network(network, state_size, action_size, embedding_dim, num_sample).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        
        self.action_size = action_size
        self.num_support = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max
        
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
        sample_min = 0 if training else self.sample_min
        sample_max = 1 if training else self.sample_max
        
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            logits, _ = self.network(torch.FloatTensor(state).to(self.device), sample_min, sample_max)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).data.cpu().numpy()
        return action

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        # Get Theta Pred, Tau
        logit, tau = self.network(state)
        logits, q_action = self.logits2Q(logit)
        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action.long()]

        theta_pred = action_onehot @ logits
        tau = torch.transpose(tau, 1, 2)
        
        with torch.no_grad():
            # Get Theta Target 
            logit_next, _ = self.network(next_state)
            _, q_next = self.logits2Q(logit_next)

            logit_target, _ = self.target_network(next_state)
            logits_target, _ = self.logits2Q(logit_target)
            
            max_a = torch.argmax(q_next, axis=-1, keepdim=True)
            max_a_onehot = action_eye[max_a.long()]

            theta_target = reward + (1-done) * self.gamma * torch.squeeze(max_a_onehot @ logits_target, 1)
            theta_target = torch.unsqueeze(theta_target, 2)
        
        error_loss = theta_target - theta_pred 
        huber_loss = F.smooth_l1_loss(theta_target, theta_pred, reduction='none')
        
        # Get Loss
        loss = torch.where(error_loss < 0.0, 1-tau, tau) * huber_loss
        loss = torch.mean(torch.sum(loss, axis = 2))
        
        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
        }
        return result
    
    def logits2Q(self, logits):
        _logits = torch.transpose(logits, 1, 2)

        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action