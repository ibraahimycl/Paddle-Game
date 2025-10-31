# 🏓 Paddle Game with Enhanced AI

## 🎥 Demo Videos

The repository includes two demonstration videos showing the AI's performance before and after training:

### Before Training (Random Movements)
[![Watch Before Training](https://img.youtube.com/vi/1-ApeaywOaY8l-0MRj3qgG3jiSiO4YyM0/0.jpg)](https://drive.google.com/file/d/1-ApeaywOaY8l-0MRj3qgG3jiSiO4YyM0/view?usp=drive_link)

> **Click the image above to watch the before training demo on Google Drive.**

### After Training (Intelligent Gameplay)
[![Watch After Training](https://img.youtube.com/vi/1QPb9R3c9aFxh0yCi8EzIQkjx22HBDh9K/0.jpg)](https://drive.google.com/file/d/1QPb9R3c9aFxh0yCi8EzIQkjx22HBDh9K/view?usp=drive_link)

> **Click the image above to watch the after training demo on Google Drive.**

These videos demonstrate the significant improvement in AI performance through the training process.

## Features

### AI Models
- **DQN AI** (`src/ai/dqn_ai.py`): Advanced AI using Deep Q-Network with Double DQN
- **Simple AI** (`src/ai/paddle_ai.py`): Basic Q-Learning based AI

### Advanced AI Features

#### 1. **Enhanced State Vector**
- **Previous**: 4-dimensional state (ball_y, paddle_y, ball_direction_x, ball_direction_y)
- **New**: 6-dimensional state (ball_y, paddle_y, ball_direction_x, ball_direction_y, **ball_velocity_x**, **paddle_height**)
- **Normalization**: All values are normalized by screen dimensions

#### 2. **Double DQN Implementation**
```python
# Select best action using main network
best_actions = torch.argmax(self.model(next_states), dim=1)
# Calculate Q value using target network
next_q_values = self.target_model(next_states).gather(1, best_actions.unsqueeze(1))
```

#### 3. **Target Network Updates**
- Target network updates every 100 steps
- Ensures more stable training

#### 4. **Logarithmic Epsilon Decay**
```python
self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-decay_rate * self.steps)
```

#### 5. **Robust Loss Calculation**
```python
target_tensor = current_q_values.clone().detach()
loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
```

## 📁 Project Structure

```
PaddleGame/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # Main entry point
├── src/                   # Source code
│   ├── __init__.py
│   ├── paddle_game.py     # Main game logic
│   └── ai/                # AI implementations
│       ├── __init__.py
│       ├── dqn_ai.py      # Advanced DQN AI
│       └── paddle_ai.py   # Simple Q-Learning AI
├── tests/                 # Test files
│   ├── __init__.py
│   └── ai_test_untrained.py   # Untrained model test
├── models/                # Trained model checkpoints
│   ├── paddle_dqn_model.pth
│   └── paddle_dqn_model_checkpoint_*.pth
└── media/                 # Demo videos
    ├── before.MOV         # Video showing untrained AI
    └── after.MOV          # Video showing trained AI
```

## Usage

### Installation
```bash
cd PaddleGame
pip install -r requirements.txt
```

### Training the AI
```bash
python main.py train
```

### Resume Training from Checkpoint
```bash
python main.py resume models/paddle_dqn_model_checkpoint_100000.pth
```

### Playing the Game
```bash
python main.py
```

### Testing Untrained Model
```bash
python main.py test
# or directly run
python -m tests.ai_test_untrained
```

## 🔧 Technical Details

### Reward System
- **Hitting the ball**: +2 points
- **Hitting near center**: +1 extra point
- **Missing the ball**: -2 points
- **Being close to ball**: +0.2 points

## Test Scenarios

### 1. Untrained Model Test
Use the test command to observe untrained model behavior:
- Random movements
- High epsilon values
- Frame count and epsilon display

### 2. Before/After Comparison
1. Run `python main.py test` (untrained)
2. Train with `python main.py train`
3. Test trained model with `python main.py`


## Performance Optimizations

1. **CPU Optimization**: Uses CPU instead of GPU for better compatibility
2. **Compact Network**: 64-32-3 architecture
3. **Efficient Memory**: 10,000 experience limit
4. **Frequent Updates**: Target network updates every 100 steps

## 🛠️ Dependencies

- PyTorch
- Pygame
- NumPy
- Matplotlib (for visualization)

## 📝 License

This project is developed for educational purposes.
