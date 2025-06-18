import pygame # oyun kütüphanesi
import sys # sistem kütüphanesi
import random # rastgele sayı üretmek için
from dqn_ai import DQNAI # derin Q-öğrenme AI modeli

# Pygame'i başlat
pygame.init()

# Sabitler
WINDOW_WIDTH = 800  
WINDOW_HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
BALL_SIZE = 15
PADDLE_SPEED = 5
BALL_SPEED_X = 5
BALL_SPEED_Y = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Oyun penceresini oluştur
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Derin Q-Öğrenme AI ile Paddle Oyunu")

# Oyun nesnelerini oluştur
class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.score = 0
        self.speed = PADDLE_SPEED

    def move(self, up=True):
        if up and self.rect.top > 0:
            self.rect.y -= self.speed
        if not up and self.rect.bottom < WINDOW_HEIGHT:
            self.rect.y += self.speed

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(WINDOW_WIDTH//2 - BALL_SIZE//2,
                              WINDOW_HEIGHT//2 - BALL_SIZE//2,
                              BALL_SIZE, BALL_SIZE)
        self.speed_x = BALL_SPEED_X * random.choice((1, -1))
        self.speed_y = BALL_SPEED_Y * random.choice((1, -1))
        
        # Add slight random variation to initial speed
        speed_variation = random.uniform(0.8, 1.2)
        self.speed_x *= speed_variation
        self.speed_y *= speed_variation

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= WINDOW_HEIGHT:
            self.speed_y *= -1

    def draw(self):
        pygame.draw.circle(screen, WHITE, self.rect.center, self.rect.width // 2)
    
    @property
    def velocity(self):
        """Return ball velocity as a list [vx, vy] for AI state."""
        return [self.speed_x, self.speed_y]

def resume_training_from_checkpoint(checkpoint_file, total_timesteps=250000):
    """
    Resume training from a checkpoint file.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        total_timesteps: Total timesteps to train for
    """
    ai = DQNAI()
    ai.load_model(checkpoint_file)
    
    # Extract timesteps from checkpoint filename
    import re
    match = re.search(r'checkpoint_(\d+)', checkpoint_file)
    if match:
        start_timesteps = int(match.group(1))
    else:
        start_timesteps = 0
    
    remaining_timesteps = total_timesteps - start_timesteps
    
    print(f"\nResuming training from checkpoint: {checkpoint_file}")
    print(f"Starting from {start_timesteps} timesteps")
    print(f"Remaining timesteps: {remaining_timesteps}")
    
    return train_ai(remaining_timesteps, ai, start_timesteps)

def train_ai(total_timesteps=250000, ai=None, start_timesteps=0):
    """
    Train the AI with optional checkpoint resumption.
    
    Args:
        total_timesteps: Total timesteps to train for
        ai: Pre-loaded AI model (for resumption)
        start_timesteps: Starting timestep count (for resumption)
    """
    if ai is None:
        ai = DQNAI()
    
    player1 = Paddle(50, WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2)
    player2 = Paddle(WINDOW_WIDTH - 50 - PADDLE_WIDTH, WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2)
    ball = Ball()
    
    print("\nStarting Deep Q-Learning AI Training...")
    print(f"Total timesteps: {total_timesteps}")
    print("Training parameters:")
    print(f"Learning rate: 0.001")
    print(f"Gamma: 0.99")
    print(f"Epsilon decay: Logarithmic")
    print(f"Target network update: Every 100 steps")
    print(f"State size: 6 (enhanced with ball velocity and paddle height)")
    print(f"Model saving: Every 10,000 timesteps")
    print("\nTraining in progress...")
    
    start_time = pygame.time.get_ticks()
    timesteps = start_timesteps
    episode = 0
    last_save_timesteps = start_timesteps
    save_interval = 10000  # Save every 10,000 timesteps
    
    while timesteps < total_timesteps:
        # Reset game state
        player1.rect.y = WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2
        player2.rect.y = WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2
        ball.reset()
        
        episode_reward = 0
        steps = 0
        
        while True:
            # Get current state with enhanced features
            state = ai.get_state(ball.rect.y, player1.rect.y, 
                               ball.speed_x, ball.speed_y,
                               ball.velocity[0], player1.rect.height)
            
            # Get AI action
            action = ai.get_action(state)
            
            # Move AI paddle
            if action == -1:  # Up
                player1.move(up=True)
            elif action == 1:  # Down
                player1.move(up=False)
            
            # Move ball
            ball.move()
            
            # Calculate reward with clear definition
            reward = 0
            done = False
            
            if ball.rect.colliderect(player1.rect):
                # Reward for hitting the ball
                reward = 2
                # Additional reward for hitting the ball in the center of the paddle
                paddle_center = player1.rect.centery
                ball_center = ball.rect.centery
                distance = abs(paddle_center - ball_center)
                if distance < PADDLE_HEIGHT/4:  # If hit near center
                    reward += 1
            elif ball.rect.left <= 0:
                # Penalty for missing the ball
                reward = -2
                done = True
            else:
                # Small reward for being close to the ball's y position
                paddle_center = player1.rect.centery
                ball_center = ball.rect.centery
                distance = abs(paddle_center - ball_center)
                if distance < 50:  # If paddle is close to ball's y position
                    reward = 0.2
            
            episode_reward += reward
            steps += 1
            timesteps += 1
            
            # Get next state with enhanced features
            next_state = ai.get_state(ball.rect.y, player1.rect.y,
                                    ball.speed_x, ball.speed_y,
                                    ball.velocity[0], player1.rect.height)
            
            # Remember the experience
            ai.remember(state, action, reward, next_state, done)
            
            # Train the AI
            ai.replay(32)
            
            # Save model at regular intervals
            if timesteps - last_save_timesteps >= save_interval:
                ai.save_model(f"paddle_dqn_model_checkpoint_{timesteps}.pth")
                last_save_timesteps = timesteps
                print(f"Checkpoint saved at {timesteps} timesteps")
            
            # Check if episode should end
            if ball.rect.left <= 0 or ball.rect.right >= WINDOW_WIDTH:
                break
        
        # Progress update every 100 episodes
        if episode % 100 == 0:
            current_time = pygame.time.get_ticks()
            elapsed_time = (current_time - start_time) / 1000  # Convert to seconds
            progress = (timesteps / total_timesteps) * 100
            print(f"\nProgress: {progress:.1f}%")
            print(f"Timesteps: {timesteps}/{total_timesteps}")
            print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"Epsilon: {ai.epsilon:.2f}")
            print(f"Average reward: {episode_reward/steps:.2f}")
            print("------------------------")
        
        episode += 1
    
    print("\nTraining complete! Saving final model...")
    ai.save_model("paddle_dqn_model.pth")
    print("Final model saved as 'paddle_dqn_model.pth'")
    print("\nStarting game in 3 seconds...")
    pygame.time.wait(3000)  # Wait 3 seconds before starting the game
    return ai

def play_against_ai(ai):
    player1 = Paddle(50, WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2)
    player2 = Paddle(WINDOW_WIDTH - 50 - PADDLE_WIDTH, WINDOW_HEIGHT//2 - PADDLE_HEIGHT//2)
    ball = Ball()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle player input (player2)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            player2.move(up=True)
        if keys[pygame.K_DOWN]:
            player2.move(up=False)
        
        # AI move (player1) with enhanced state
        state = ai.get_state(ball.rect.y, player1.rect.y,
                           ball.speed_x, ball.speed_y,
                           ball.velocity[0], player1.rect.height)
        action = ai.get_action(state)
        if action == -1:
            player1.move(up=True)
        elif action == 1:
            player1.move(up=False)
        
        # Move ball
        ball.move()
        
        # Ball collision with paddles
        if ball.rect.colliderect(player1.rect) or ball.rect.colliderect(player2.rect):
            ball.speed_x *= -1
        
        # Score points
        if ball.rect.left <= 0:
            player2.score += 1
            ball.reset()
        if ball.rect.right >= WINDOW_WIDTH:
            player1.score += 1
            ball.reset()
        
        # Draw everything
        screen.fill(BLACK)
        pygame.draw.line(screen, WHITE, (WINDOW_WIDTH//2, 0), (WINDOW_WIDTH//2, WINDOW_HEIGHT), 2)
        
        # Draw scores
        font = pygame.font.Font(None, 74)
        score1 = font.render(str(player1.score), True, WHITE)
        score2 = font.render(str(player2.score), True, WHITE)
        screen.blit(score1, (WINDOW_WIDTH//4, 20))
        screen.blit(score2, (3*WINDOW_WIDTH//4, 20))
        
        player1.draw()
        player2.draw()
        ball.draw()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training Deep Q-Learning AI...")
            ai = train_ai()
            print("Training complete! Model saved as 'paddle_dqn_model.pth'")
        elif sys.argv[1] == "resume" and len(sys.argv) > 2:
            checkpoint_file = sys.argv[2]
            print(f"Resuming training from checkpoint: {checkpoint_file}")
            ai = resume_training_from_checkpoint(checkpoint_file)
            print("Training complete! Model saved as 'paddle_dqn_model.pth'")
        elif sys.argv[1] == "test":
            print("Testing untrained AI...")
            import ai_test_untrained
        else:
            print("Usage:")
            print("  python paddle_game.py train          # Train new model")
            print("  python paddle_game.py resume <file>  # Resume from checkpoint")
            print("  python paddle_game.py test           # Test untrained model")
            print("  python paddle_game.py                # Play against trained model")
    else:
        try:
            ai = DQNAI()
            ai.load_model("paddle_dqn_model.pth")
            print("Loaded trained model. Starting game...")
        except:
            print("No trained model found. Starting training...")
            ai = train_ai()
            print("Training complete! Starting game...")
        
        play_against_ai(ai) 