import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import copy
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network architecture for paddle game AI.
    """
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Small network architecture optimized for CPU
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAI:
    """
    Deep Q-Learning AI agent for paddle game with Double DQN and target network.
    """
    def __init__(self, state_size=6, action_size=3, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Smaller memory for CPU efficiency
        self.gamma = 0.99    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cpu")  # Force CPU usage
        
        # Training tracking
        self.steps = 0
        self.decay_rate = 0.001  # For logarithmic epsilon decay
        
        # For tracking training progress
        self.training_step = 0
        self.target_update_frequency = 100  # Update target network every 100 steps
        
        # Create main and target networks
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()
        
        print(f"DQN AI initialized with state_size={state_size}, action_size={action_size}")
        
    def update_target_model(self):
        """Update target network with main network weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Target network updated at step {self.training_step}")
    
    def get_state(self, ball_y, paddle_y, ball_direction_x, ball_direction_y, ball_velocity_x, paddle_height):
        """
        Get normalized state vector with enhanced features.
        
        Args:
            ball_y: Ball Y position
            paddle_y: Paddle Y position  
            ball_direction_x: Ball X direction
            ball_direction_y: Ball Y direction
            ball_velocity_x: Ball X velocity
            paddle_height: Paddle height
            
        Returns:
            torch.Tensor: Normalized state vector
        """
        # Normalize inputs by screen dimensions
        ball_y = ball_y / 600.0  # WINDOW_HEIGHT
        paddle_y = paddle_y / 600.0
        ball_direction_x = ball_direction_x / 3.0  # BALL_SPEED_X
        ball_direction_y = ball_direction_y / 3.0  # BALL_SPEED_Y
        ball_velocity_x = ball_velocity_x / 5.0  # Normalize velocity
        paddle_height = paddle_height / 90.0  # PADDLE_HEIGHT
        
        state = torch.FloatTensor([ball_y, paddle_y, ball_direction_x, ball_direction_y, ball_velocity_x, paddle_height]).to(self.device)
        
        # Debug assertion
        assert not torch.isnan(state).any(), f"NaN detected in state: {state}"
        assert not torch.isinf(state).any(), f"Inf detected in state: {state}"
        
        return state
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy with logarithmic decay.
        
        Args:
            state: Current game state
            
        Returns:
            int: Action (-1: up, 0: stay, 1: down)
        """
        # Logarithmic epsilon decay
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-self.decay_rate * self.steps)
        
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size) - 1  # -1, 0, or 1
            print(f"Random action: {action} (epsilon: {self.epsilon:.3f})")
            return action
        
        with torch.no_grad():
            state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.model(state)
            action = torch.argmax(q_values).item() - 1
            print(f"Greedy action: {action} (Q-values: {q_values.squeeze().tolist()})")
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Debug assertions
        assert isinstance(reward, (int, float)), f"Reward must be numeric, got {type(reward)}"
        assert isinstance(done, bool), f"Done must be boolean, got {type(done)}"
        
        self.memory.append((state, action + 1, reward, next_state, done))
        self.steps += 1
    
    def replay(self, batch_size=32):
        """
        Train the model using Double DQN with experience replay.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.stack([x[0] for x in minibatch])
        actions = torch.tensor([x[1] for x in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float).to(self.device)
        next_states = torch.stack([x[3] for x in minibatch])
        dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select action, target network to evaluate
        with torch.no_grad():
            # Select best action using main network
            best_actions = torch.argmax(self.model(next_states), dim=1)
            # Get Q values from target network for selected actions
            next_q_values = self.target_model(next_states).gather(1, best_actions.unsqueeze(1))
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()
        
        # Compute loss with more robust target handling
        target_tensor = current_q_values.clone().detach()
        target_tensor = target_tensor.squeeze()
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Debug assertions
        assert not torch.isnan(loss), f"NaN loss detected: {loss}"
        assert not torch.isinf(loss), f"Inf loss detected: {loss}"
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_model()
        
        # Print training info occasionally
        if self.training_step % 1000 == 0:
            print(f"Training step {self.training_step}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.3f}")
    
    def save_model(self, filename):
        """
        Save model weights and training state.
        
        Args:
            filename: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_step': self.training_step
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load model weights and training state.
        
        Args:
            filename: Path to load the model from
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {filename}") 