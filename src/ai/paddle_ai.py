import numpy as np
import random
import math

class PaddleAI:
    """
    Simple Q-Learning AI agent for paddle game with enhanced features.
    """
    def __init__(self, learning_rate=0.2, discount_factor=0.99, exploration_rate=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        
        # Enhanced state space: (ball_y, paddle_y, ball_velocity_x, paddle_height)
        # Action space: (up, stay, down)
        self.q_table = {}
        self.steps = 0
        self.decay_rate = 0.001  # For logarithmic epsilon decay
        
        print(f"Simple PaddleAI initialized with learning_rate={learning_rate}")
        
    def get_state_key(self, ball_y, paddle_y, ball_velocity_x, paddle_height):
        """
        Get discretized state key with enhanced features.
        
        Args:
            ball_y: Ball Y position
            paddle_y: Paddle Y position
            ball_velocity_x: Ball X velocity
            paddle_height: Paddle height
            
        Returns:
            tuple: Discretized state key
        """
        # Normalize and discretize state space
        ball_y = round(ball_y / 10) * 10
        paddle_y = round(paddle_y / 10) * 10
        ball_velocity_x = round(ball_velocity_x / 2) * 2  # Discretize velocity
        paddle_height = round(paddle_height / 10) * 10  # Discretize height
        
        return (ball_y, paddle_y, ball_velocity_x, paddle_height)
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy with logarithmic decay.
        
        Args:
            state: Current game state
            
        Returns:
            int: Action (-1: up, 0: stay, 1: down)
        """
        # Logarithmic epsilon decay
        self.exploration_rate = self.min_exploration_rate + (1.0 - self.min_exploration_rate) * math.exp(-self.decay_rate * self.steps)
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            action = random.choice([-1, 0, 1])  # -1: up, 0: stay, 1: down
            print(f"Random action: {action} (epsilon: {self.exploration_rate:.3f})")
            return action
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # Initialize Q-values for the state
            
        action = np.argmax(self.q_table[state]) - 1  # Convert index to action
        print(f"Greedy action: {action} (Q-values: {self.q_table[state]})")
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning algorithm.
        
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
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]
            
        # Q-learning update
        action_index = action + 1  # Convert action to index
        old_value = self.q_table[state][action_index]
        next_max = max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max * (1 - done))
        self.q_table[state][action_index] = new_value
        
        self.steps += 1
        
        # Print update info occasionally
        if self.steps % 1000 == 0:
            print(f"Step {self.steps}, State: {state}, Action: {action}, Reward: {reward}, New Q-value: {new_value:.3f}")
    
    def save_model(self, filename):
        """
        Save Q-table to file.
        
        Args:
            filename: Path to save the model
        """
        np.save(filename, self.q_table)
        print(f"Simple AI model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load Q-table from file.
        
        Args:
            filename: Path to load the model from
        """
        self.q_table = np.load(filename, allow_pickle=True).item()
        print(f"Simple AI model loaded from {filename}") 