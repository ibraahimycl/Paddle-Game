# 🏓 Paddle Game with Enhanced AI

This project features an advanced AI system for a paddle game using Deep Q-Learning (DQN) with sophisticated improvements and optimizations.

## 🚀 Features

### 🤖 AI Models
- **DQN AI** (`dqn_ai.py`): Advanced AI using Deep Q-Network with Double DQN
- **Simple AI** (`paddle_ai.py`): Basic Q-Learning based AI

### 🧠 Advanced AI Features

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

#### 6. **Debugging and Monitoring**
- Comprehensive docstrings for all functions
- Print/log outputs for action selections
- Assert statements for NaN/Inf checking
- Training progress monitoring

## 📁 Project Structure

```
PaddleGame/
├── dqn_ai.py              # Advanced DQN AI
├── paddle_ai.py           # Simple Q-Learning AI
├── paddle_game.py         # Main game file
├── ai_test_untrained.py   # Untrained model test
├── requirements.txt       # Python dependencies
├── models/                # Trained model checkpoints
│   ├── paddle_dqn_model.pth
│   └── paddle_dqn_model_checkpoint_*.pth
├── before.MOV             # Video showing untrained AI performance
├── after.MOV              # Video showing trained AI performance
└── README.md              # This file
```

## 🎮 Usage

### Installation
```bash
cd PaddleGame
pip install -r requirements.txt
```

### Training the AI
```bash
cd PaddleGame
python paddle_game.py train
```

### Playing the Game
```bash
cd PaddleGame
python paddle_game.py
```

### Testing Untrained Model
```bash
cd PaddleGame
python ai_test_untrained.py
```

## 🔧 Technical Details

### State Vector (6-dimensional)
1. `ball_y / WINDOW_HEIGHT` - Ball Y position (normalized)
2. `paddle_y / WINDOW_HEIGHT` - Paddle Y position (normalized)
3. `ball_direction_x / BALL_SPEED_X` - Ball X direction (normalized)
4. `ball_direction_y / BALL_SPEED_Y` - Ball Y direction (normalized)
5. `ball_velocity_x / 5.0` - Ball X velocity (normalized) ⭐ **NEW**
6. `paddle_height / PADDLE_HEIGHT` - Paddle height (normalized) ⭐ **NEW**

### Reward System
- **Hitting the ball**: +2 points
- **Hitting near center**: +1 extra point
- **Missing the ball**: -2 points
- **Being close to ball**: +0.2 points

### Training Parameters
- **Learning Rate**: 0.01
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Decay**: Logarithmic
- **Target Network Update**: Every 100 steps
- **Batch Size**: 32
- **Memory Size**: 10,000

## 🧪 Test Scenarios

### 1. Untrained Model Test
Use `ai_test_untrained.py` to observe untrained model behavior:
- Random movements
- High epsilon values
- Frame count and epsilon display

### 2. Before/After Comparison
1. Run `ai_test_untrained.py` (untrained)
2. Train with `paddle_game.py train`
3. Test trained model with `paddle_game.py`

## 📊 Monitoring and Debugging

### Debug Assertions
- NaN/Inf checking in state values
- Reward type validation
- Boolean done value checking

## 🎯 Performance Optimizations

1. **CPU Optimization**: Uses CPU instead of GPU for better compatibility
2. **Compact Network**: 64-32-3 architecture
3. **Efficient Memory**: 10,000 experience limit
4. **Frequent Updates**: Target network updates every 100 steps

## 🎥 Demo Videos

The repository includes two demonstration videos showing the AI's performance before and after training:

### Before Training (Random Movements)
<video width="640" height="480" controls>
  <source src="before.MOV" type="video/quicktime">
  Your browser does not support the video tag.
</video>

### After Training (Intelligent Gameplay)
<video width="640" height="480" controls>
  <source src="after.MOV" type="video/quicktime">
  Your browser does not support the video tag.
</video>

These videos demonstrate the significant improvement in AI performance through the training process.

## 🔮 Future Enhancements

- [ ] Prioritized Experience Replay
- [ ] Dueling DQN
- [ ] Multi-agent training
- [ ] Visual state representation
- [ ] Curriculum learning
- [ ] Performance metrics dashboard

## 🛠️ Dependencies

- PyTorch
- Pygame
- NumPy
- Matplotlib (for visualization)

## 📝 License

This project is developed for educational purposes.

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing the AI algorithms

## 📞 Contact

For questions or suggestions, please open an issue on GitHub. 
