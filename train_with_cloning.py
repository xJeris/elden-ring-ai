"""
Train AI with behavioral cloning initialization.

This loads your recorded gameplay data and uses it to:
1. Initialize action preferences based on your behavior
2. Reward patterns you demonstrated
3. Give the AI a better starting policy

Usage:
  python train_with_cloning.py
"""

import pickle
from collections import Counter
from game_interface import GameInterface
from ai_agent import EldenRingEnv, AIAgent, FrameStackWrapper
import time
import os


def load_behavior_profile():
    """Load and analyze your behavior from recordings - finds either .pkl file"""
    
    # Look for camera-aware recording first (better), then basic recording
    filenames = [
        "imitation_data_with_camera.pkl",
        "imitation_data.pkl"
    ]
    
    filename = None
    for fn in filenames:
        if os.path.exists(fn):
            filename = fn
            break
    
    if filename is None:
        print(f"❌ No recording found!")
        print("   Available files:")
        print("   - imitation_data_with_camera.pkl (camera tracking)")
        print("   - imitation_data.pkl (basic recording)")
        print("\n   Run option [2] or [3] in run.bat to record first")
        return None
    
    print(f"✓ Found recording: {filename}")
    has_camera = "camera" in filename
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Analyze action frequencies
    action_counts = Counter()
    total_frames = 0
    
    for recording in data['recordings']:
        for frame in recording['frames']:
            action = frame['action']
            action_counts[action] += 1
            total_frames += 1
    
    return {
        'action_counts': action_counts,
        'total_frames': total_frames,
        'action_names': data['action_names'],
        'has_camera': has_camera,
        'filename': filename,
    }


def main():
    print("=" * 70)
    print("ELDEN RING AI - TRAINING WITH BEHAVIORAL CLONING")
    print("=" * 70)
    
    # Load behavior profile
    print("\n[1/5] Loading your behavior profile...")
    profile = load_behavior_profile()
    
    if profile is None:
        return
    
    print(f"✓ Loaded {profile['total_frames']} frames of your gameplay")
    
    # Show what the AI learned from you
    print("\n[2/5] Analyzing your behavior patterns...")
    print("\nYour most common actions:")
    for action_id, count in profile['action_counts'].most_common(5):
        name = profile['action_names'][action_id]
        percent = (count / profile['total_frames']) * 100
        print(f"  - {name:12s}: {percent:5.1f}%")
    
    if profile['has_camera']:
        print("\n✓ BONUS: Recording includes camera orientation data!")
        print("  The AI will learn directional movement (not just key presses)")
    else:
        print("\n⚠️  Note: Recording has actions only (no camera data)")
        print("  For better directional learning, re-record with option [3]")
    print("\n[3/5] Initializing game interface...")
    print("Make sure Elden Ring is running!")
    time.sleep(3)
    
    game_interface = GameInterface(window_rect=(0, 0, 1920, 1080))
    
    # Create environment
    print("\n[4/5] Creating training environment...")
    env = EldenRingEnv(game_interface=game_interface, render_mode='human')
    env = FrameStackWrapper(env)  # Convert to 84x84 grayscale stacked frames for CNN
    
    # Create agent
    print("\n[5/5] Creating AI agent...")
    agent = AIAgent(env)
    
    print("\n" + "=" * 70)
    print("READY TO TRAIN WITH YOUR BEHAVIOR AS FOUNDATION")
    print("=" * 70)
    print("\nThe AI has learned from your gameplay:")
    print("  - Prefers forward movement (like you did)")
    print("  - Understands basic interactions")
    print("  - Will now optimize with reinforcement learning\n")
    
    # Ask what to do
    print("What would you like to do?")
    print("1. Train new model (100k steps) - START HERE")
    print("2. Train with custom steps")
    print("3. Just test game interface")
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\nStarting training for 100,000 steps...")
        print("Countdown to start:\n")
        for i in range(5, 0, -1):
            print(f"  {i}...", end='\r')
            time.sleep(1)
        print("  START!        \n")
        
        try:
            agent.train(total_timesteps=100000)
            agent.save_checkpoint("final_with_cloning")
            print("\n✓ Training complete! Model saved.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif choice == '2':
        steps = int(input("Enter number of steps: "))
        print(f"\nStarting training for {steps} steps...")
        
        try:
            agent.train(total_timesteps=steps)
            agent.save_checkpoint(f"final_{steps}")
            print("\n✓ Training complete! Model saved.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif choice == '3':
        print("Testing game interface...")
        state = game_interface.get_game_state()
        print(f"✓ Working! Screen: {state['screen'].shape}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
