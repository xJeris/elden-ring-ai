"""
Simple behavioral cloning approach.

Instead of complex setup, just:
1. Record yourself playing for ~2-3 minutes
2. Press keys naturally (W/A/S/D to move, E to interact, etc)
3. This trains the model to copy your behavior before RL fine-tuning

Usage:
  python behavioral_cloning_simple.py
"""

import numpy as np
import json
import pickle
import time
from game_interface import GameInterface
from pynput import keyboard
from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleImitationLearner:
    """Simplified behavioral cloning - copy what human does"""
    
    def __init__(self):
        self.game_interface = GameInterface()
        self.recordings = []
        self.current_recording = []
        self.keys_pressed = defaultdict(bool)
        self.recording = False
        
        # Map keys to action indices
        self.key_to_action = {
            'w': 1,      # Forward
            'a': 3,      # Left
            's': 2,      # Backward
            'd': 4,      # Right
            'q': 10,     # Lock-on
            'e': 12,     # Interact
            'r': 9,      # Item
            ' ': 7,      # Dodge (space)
            'x': 7,      # Dodge (alt)
            'f': 11,     # Jump
        }
        
        self.action_names = {
            0: "No-op", 1: "Forward", 2: "Backward", 3: "Left", 4: "Right",
            5: "Light", 6: "Heavy", 7: "Dodge", 8: "Skill", 9: "Item",
            10: "Lock", 11: "Jump", 12: "Interact", 13: "Mount"
        }
    
    def get_current_action(self):
        """Determine action from currently pressed keys"""
        # Priority: movement > interaction > combat
        if self.keys_pressed['w']:
            return 1  # Forward
        elif self.keys_pressed['s']:
            return 2  # Backward
        elif self.keys_pressed['a']:
            return 3  # Left
        elif self.keys_pressed['d']:
            return 4  # Right
        elif self.keys_pressed['e']:
            return 12  # Interact
        elif self.keys_pressed['q']:
            return 10  # Lock-on
        elif self.keys_pressed['x'] or self.keys_pressed[' ']:
            return 7  # Dodge
        elif self.keys_pressed['f']:
            return 11  # Jump
        else:
            return 0  # No-op
    
    def on_press(self, key):
        """Handle key press"""
        try:
            if key == keyboard.Key.esc:
                self.stop()
                return False
            
            char = key.char if hasattr(key, 'char') else str(key)
            if char in self.key_to_action:
                self.keys_pressed[char] = True
        except:
            pass
    
    def on_release(self, key):
        """Handle key release"""
        try:
            char = key.char if hasattr(key, 'char') else str(key)
            if char in self.key_to_action:
                self.keys_pressed[char] = False
        except:
            pass
    
    def record(self, duration=120):
        """
        Record human gameplay for behavioral cloning.
        
        Args:
            duration: How long to record in seconds
        """
        print(f"\n{'='*70}")
        print("🎮 IMITATION LEARNING - RECORD YOUR GAMEPLAY")
        print(f"{'='*70}")
        print(f"\nRecording for {duration} seconds...")
        print("Play naturally and explore. Your actions will be recorded.\n")
        print("Key Controls:")
        print("  W/A/S/D  - Move")
        print("  E        - Interact with doors")
        print("  Q        - Lock-on to enemies")
        print("  X/SPACE  - Dodge")
        print("  F        - Jump")
        print("  ESC      - Stop early")
        print(f"\n{'='*70}\n")
        
        # Countdown before starting
        print("⏱️  Recording starts in:")
        for i in range(3, 0, -1):
            print(f"  {i}...", end='', flush=True)
            time.sleep(1)
        print("\n🔴 RECORDING NOW!")
        print(f"{'='*70}\n")
        
        self.recording = True
        self.current_recording = []
        
        # Start keyboard listener
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.recording and (time.time() - start_time) < duration:
                # Get game state
                state = self.game_interface.get_game_state()
                
                # Get current action from key presses
                action = self.get_current_action()
                
                # Record the observation-action pair
                self.current_recording.append({
                    'observation': {
                        'health': float(state['health_percent']),
                        'stamina': float(state['stamina_percent']),
                        'exits': state['exits'],
                        'is_outdoor': bool(state['is_outdoor']),
                        'in_combat': bool(state['in_combat']),
                    },
                    'action': int(action),
                    'action_name': self.action_names[action],
                })
                
                frame_count += 1
                elapsed = time.time() - start_time
                action_name = self.action_names[action]
                
                print(f"Frame {frame_count:4d} ({elapsed:5.1f}s) - Action: {action_name:12s}", end='\r')
                
                # Record at ~20 FPS
                time.sleep(0.05)
        
        finally:
            listener.stop()
            self.recording = False
        
        elapsed = time.time() - start_time
        print(f"\n✓ Recording complete! {frame_count} frames in {elapsed:.1f} seconds")
        
        # Save recording
        self.recordings.append({
            'frames': self.current_recording,
            'duration': elapsed,
            'frame_count': len(self.current_recording),
            'timestamp': time.time()
        })
        
        return self.current_recording
    
    def save(self, filename="imitation_data.pkl"):
        """Save recordings to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'recordings': self.recordings,
                'action_names': self.action_names,
            }, f)
        
        total_frames = sum(r['frame_count'] for r in self.recordings)
        print(f"✓ Saved {len(self.recordings)} recordings ({total_frames} frames) to {filename}")


def main():
    """Interactive recording"""
    learner = SimpleImitationLearner()
    
    print(f"\n{'='*70}")
    print("BEHAVIORAL CLONING - RECORD YOUR GAMEPLAY")
    print(f"{'='*70}\n")
    
    while True:
        print("1. Record gameplay (for behavioral cloning)")
        print("2. Quit")
        choice = input("Choose: ").strip()
        
        if choice == "1":
            duration_input = input("Duration to record (seconds) [default 120]: ").strip()
            duration = int(duration_input) if duration_input else 120
            
            # Confirm ready
            print(f"\n{'='*70}")
            print("⚠️  ABOUT TO START RECORDING")
            print(f"{'='*70}")
            print(f"You will record for {duration} seconds")
            print("Make sure Elden Ring window is visible and ready to play")
            print("Your actions will be recorded as you play\n")
            
            ready = input("Ready? Press ENTER to start recording: ")
            
            # Now record
            learner.record(duration)
            
            # Auto-save
            learner.save()
            
            print("\n✓ Recording automatically saved!")
            print("  You can now run option [3] in run.bat to pre-train the AI")
        
        elif choice == "2":
            break
    
    print("\nExiting...")


if __name__ == "__main__":
    main()
