import pygame
import sys
from paddle_game import Ball, Paddle
from dqn_ai import DQNAI

# Pygame başlangıç
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Test - Untrained Model")

clock = pygame.time.Clock()
FPS = 60

# Oyun nesneleri
ball = Ball()
left_paddle = Paddle(x=50, y=HEIGHT//2 - 60)
right_paddle = Paddle(x=WIDTH - 60, y=HEIGHT//2 - 60)
ai = DQNAI()  # Untrained model

print("Testing untrained AI model behavior...")
print("This will show random actions before training.")
print("Press any key to continue...")
input()

running = True
frame_count = 0
while running:
    screen.fill((0, 0, 0))
    pygame.draw.line(screen, (200, 200, 200), (WIDTH//2, 0), (WIDTH//2, HEIGHT))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # AI paddle hareketi - untrained model will make random decisions
    state = ai.get_state(ball.rect.y, left_paddle.rect.y, 
                        ball.speed_x, ball.speed_y,
                        ball.velocity[0], left_paddle.rect.height)
    action = ai.get_action(state)
    
    if action == -1:  # Up
        left_paddle.move(up=True)
    elif action == 1:  # Down
        left_paddle.move(up=False)
    # action == 0 means stay still

    # Sağ paddle elle kontrol (istersen burayı otomatik yapabilirsin)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        right_paddle.move(up=True)
    if keys[pygame.K_DOWN]:
        right_paddle.move(up=False)

    ball.move()
    
    # Ball collision with paddles
    if ball.rect.colliderect(left_paddle.rect) or ball.rect.colliderect(right_paddle.rect):
        ball.speed_x *= -1

    # Score points
    if ball.rect.left <= 0:
        right_paddle.score += 1
        ball.reset()
    if ball.rect.right >= WIDTH:
        left_paddle.score += 1
        ball.reset()

    # Draw scores
    font = pygame.font.Font(None, 74)
    score1 = font.render(str(left_paddle.score), True, (255, 255, 255))
    score2 = font.render(str(right_paddle.score), True, (255, 255, 255))
    screen.blit(score1, (WIDTH//4, 20))
    screen.blit(score2, (3*WIDTH//4, 20))

    left_paddle.draw()
    right_paddle.draw()
    ball.draw()

    # Show frame count and epsilon
    info_font = pygame.font.Font(None, 36)
    frame_text = info_font.render(f"Frame: {frame_count}", True, (255, 255, 255))
    epsilon_text = info_font.render(f"Epsilon: {ai.epsilon:.3f}", True, (255, 255, 255))
    screen.blit(frame_text, (10, HEIGHT - 60))
    screen.blit(epsilon_text, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(FPS)
    frame_count += 1

pygame.quit()
print(f"Test completed. Total frames: {frame_count}")
print("This shows how the untrained model behaves with random actions.") 