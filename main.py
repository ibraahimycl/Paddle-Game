#!/usr/bin/env python3
"""
Main entry point for PaddleGame
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.paddle_game import train_ai, resume_training_from_checkpoint, play_against_ai
from src.ai.dqn_ai import DQNAI

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training Deep Q-Learning AI...")
            ai = train_ai()
            print("Training complete! Model saved as 'models/paddle_dqn_model.pth'")
        elif sys.argv[1] == "resume" and len(sys.argv) > 2:
            checkpoint_file = sys.argv[2]
            print(f"Resuming training from checkpoint: {checkpoint_file}")
            ai = resume_training_from_checkpoint(checkpoint_file)
            print("Training complete! Model saved as 'models/paddle_dqn_model.pth'")
        elif sys.argv[1] == "test":
            print("Testing untrained AI...")
            import tests.ai_test_untrained
        else:
            print("Usage:")
            print("  python main.py train          # Train new model")
            print("  python main.py resume <file>  # Resume from checkpoint")
            print("  python main.py test           # Test untrained model")
            print("  python main.py                # Play against trained model")
    else:
        try:
            ai = DQNAI()
            ai.load_model("models/paddle_dqn_model.pth")
            print("Loaded trained model. Starting game...")
        except:
            print("No trained model found. Starting training...")
            ai = train_ai()
            print("Training complete! Starting game...")
        
        play_against_ai(ai)

