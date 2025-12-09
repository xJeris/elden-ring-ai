"""
Main script to train and test Elden Ring AI
"""

from game_interface import GameInterface
from ai_agent import EldenRingEnv, AIAgent, FrameStackWrapper
import time
import os
import shutil
from datetime import datetime


def archive_old_checkpoints():
    """Archive old checkpoints to a subfolder when starting fresh training"""
    checkpoint_dir = "models/checkpoints"
    archive_dir = os.path.join(checkpoint_dir, "archived")
    
    # Check if there are any checkpoints to archive
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.zip') or f.endswith('_metadata.txt')]
    
    if not checkpoint_files:
        return  # No checkpoints to archive
    
    # Create archived folder if it doesn't exist
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    
    # Create a timestamp subfolder for this archive batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(archive_dir, f"archived_{timestamp}")
    os.makedirs(batch_dir)
    
    # Move all checkpoint files to the archive
    for filename in checkpoint_files:
        source = os.path.join(checkpoint_dir, filename)
        destination = os.path.join(batch_dir, filename)
        try:
            shutil.move(source, destination)
        except Exception as e:
            print(f"Warning: Could not archive {filename}: {e}")
    
    print(f"\n✓ Old checkpoints archived to: {batch_dir}")
    print(f"  ({len(checkpoint_files)} files moved)\n")


def countdown_to_game(seconds=10):
    """Countdown timer to give user time to focus on game window"""
    print(f"\n⚠️  Switch focus to Elden Ring window!")
    print(f"Starting training in {seconds} seconds...\n")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end='\r', flush=True)
        time.sleep(1)
    print("  STARTING!        ")
    print()


def main():
    print("=" * 60)
    print("ELDEN RING AI - TRAINING SYSTEM")
    print("=" * 60)
    
    # Step 1: Initialize game interface
    print("\n[1/4] Initializing game interface...")
    print("Make sure Elden Ring is running and in focus!")
    
    game_interface = GameInterface(window_rect=(0, 0, 1920, 1080))
    
    # Step 2: Test game capture
    print("[2/4] Testing game capture...")
    state = game_interface.get_game_state()
    print(f"✓ Screen captured: {state['screen'].shape}")
    print(f"✓ Screenshot shape: {state['raw_screen'].shape}")
    
    # Step 3: Create environment
    print("\n[3/4] Creating Gymnasium environment...")
    env = EldenRingEnv(game_interface=game_interface, render_mode='human')
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")
    
    # Wrap environment with frame stacking for CNN processing
    # Converts raw RGB images to stacked grayscale frames (84x84x4)
    # CNN extracts spatial features, LSTM learns temporal patterns
    env = FrameStackWrapper(env, num_stack=4)
    print(f"✓ Environment wrapped with frame stacking (4 frames for motion detection)")
    
    # Step 4: Create and train AI agent
    print("\n[4/4] Creating AI agent...")
    agent = AIAgent(env)
    print("✓ Agent initialized with PPO algorithm")
    
    # Training options
    print("\n" + "=" * 60)
    print("What would you like to do?")
    print("1. Train new model (100k steps)")
    print("2. Train with custom steps")
    print("3. Resume training from latest checkpoint")
    print("4. List all checkpoints")
    print("5. Analyze what the AI has learned")
    print("6. Configure AI goals")
    print("7. Load and test existing model")
    print("8. Just test game interface")
    choice = input("Enter choice (1-8): ").strip()
    
    if choice == '1':
        print("\nStarting NEW training (100,000 steps)...")
        print("Archiving old checkpoints to preserve them...\n")
        archive_old_checkpoints()
        countdown_to_game(seconds=5)
        try:
            agent.train(total_timesteps=100000)
            agent.save_checkpoint("final_100k")
            print("\nModel checkpoint saved!")
        except Exception as e:
            print(f"\nError during training: {e}")
            print("Make sure Elden Ring is running and in focus!")
    
    elif choice == '2':
        steps = int(input("Enter number of training steps: "))
        print(f"\nStarting NEW training ({steps} steps)...")
        print("Archiving old checkpoints to preserve them...\n")
        archive_old_checkpoints()
        countdown_to_game(seconds=5)
        try:
            agent.train(total_timesteps=steps)
            agent.save_checkpoint(f"final_{steps}")
            print("\nModel checkpoint saved!")
        except Exception as e:
            print(f"\nError during training: {e}")
            print("Make sure Elden Ring is running and in focus!")
    
    elif choice == '3':
        if agent.load_latest_checkpoint():
            steps = int(input("Enter number of additional training steps: "))
            print(f"\nResuming training for {steps} more steps...")
            countdown_to_game(seconds=5)
            try:
                agent.train(total_timesteps=steps, resume=True)
                agent.save_checkpoint("resumed")
                print("\nModel checkpoint saved!")
            except Exception as e:
                print(f"\nError during training: {e}")
                print("Make sure Elden Ring is running and focused!")
        else:
            print("No checkpoints to resume from!")
    
    elif choice == '4':
        agent.list_checkpoints()
    
    elif choice == '5':
        print("\nOpening AI learning analyzer...")
        # Import and run analyzer directly
        from analyze import AIAnalyzer
        analyzer = AIAnalyzer(agent)
        
        print("\nWhat would you like to analyze?")
        print("1. View policy summary (what actions it prefers)")
        print("2. Test behavior (watch what it does)")
        print("3. Generate full report")
        print("4. Do all of the above")
        sub_choice = input("Enter choice (1-4): ").strip()
        
        if sub_choice in ['1', '4']:
            analyzer.get_policy_summary()
        
        if sub_choice in ['2', '4']:
            episodes = int(input("How many episodes to test? (default 3): ") or "3")
            analyzer.test_behavior(episodes=episodes)
        
        if sub_choice in ['3', '4']:
            analyzer.generate_report()
    
    elif choice == '6':
        print("\nOpening goal configuration...")
        import subprocess
        import os
        script_path = os.path.join(os.path.dirname(__file__), 'configure_goals.py')
        subprocess.Popen([r"\Python\Python312\python.exe", script_path])
    
    elif choice == '7':
        try:
            agent.load('models/elden_ring_ppo')
            print("\nTesting trained model (5 episodes)...\n")
            agent.play(episodes=5, render=True)
        except FileNotFoundError:
            print("Model not found! Train a model first.")
    
    elif choice == '8':
        print("\nTesting game interface...")
        test_interface = GameInterface()
        screenshot = test_interface.capture_screen()
        processed = test_interface.process_screen(screenshot)
        print(f"✓ Game interface working!")
        print(f"  - Screenshot shape: {screenshot.shape}")
        print(f"  - Processed shape: {processed.shape}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()

