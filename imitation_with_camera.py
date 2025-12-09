"""
Enhanced behavioral cloning that captures CAMERA POSITION + ACTIONS.

This records:
1. Your button presses (W/A/S/D)
2. Your mouse movements (camera angle)
3. Game state (health, stamina, etc)

The AI learns that "W press + mouse moved right = move northeast"
instead of just "W press = move forward"

Usage:
  python imitation_with_camera.py
"""

import numpy as np
import pickle
import time
from game_interface import GameInterface
from pynput import keyboard, mouse
from collections import defaultdict


class CameraAwareImitationLearner:
    """Records gameplay with camera orientation tracking"""
    
    def __init__(self):
        self.game_interface = GameInterface()
        self.recordings = []
        self.current_recording = []
        self.keys_pressed = defaultdict(bool)
        self.recording = False
        self.mouse_controller = mouse.Controller()
        
        # Last known mouse position to detect movement
        self.last_mouse_pos = None
        self.camera_angle_degrees = 0  # Estimated from mouse movement
        self.frame_mouse_delta = (0, 0)  # Mouse movement in this frame
        
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
    
    def get_current_action(self, camera_moved=False):
        """
        Determine action from currently pressed keys AND camera movement.
        
        Args:
            camera_moved: bool - did the mouse/camera move this frame?
        
        Returns:
            action_id: int (0-13)
        """
        # Movement keys take priority
        if self.keys_pressed['w']:
            return 1  # Forward
        elif self.keys_pressed['s']:
            return 2  # Backward
        elif self.keys_pressed['a']:
            return 3  # Left
        elif self.keys_pressed['d']:
            return 4  # Right
        # Combat/interaction keys
        elif self.keys_pressed['e']:
            return 12  # Interact
        elif self.keys_pressed['q']:
            return 10  # Lock-on
        elif self.keys_pressed['x'] or self.keys_pressed[' ']:
            return 7  # Dodge
        elif self.keys_pressed['f']:
            return 11  # Jump
        # Camera-only movement counts as adjusting direction, not no-op
        # Since no movement key is pressed, just camera rotation = still no-op for character action
        # But we track the camera angle separately
        else:
            return 0  # True no-op - no keys pressed and presumably no camera movement
    
    def update_camera_angle(self):
        """
        Estimate camera angle from mouse position.
        Returns: (camera_moved_bool, new_angle)
        """
        try:
            current_pos = self.mouse_controller.position
            
            if self.last_mouse_pos is None:
                self.last_mouse_pos = current_pos
                self.frame_mouse_delta = (0, 0)
                return False
            
            # Calculate mouse delta
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
            self.frame_mouse_delta = (dx, dy)
            
            # Update camera angle (rough estimate)
            # In Elden Ring: X movement rotates camera horizontally
            # Each pixel of mouse movement ≈ 0.5-1 degree
            self.camera_angle_degrees += dx * 0.5  # Horizontal rotation
            self.camera_angle_degrees %= 360  # Keep in 0-360 range
            
            self.last_mouse_pos = current_pos
            
            # Return True if mouse moved significantly (more than 1 pixel)
            movement = (dx*dx + dy*dy) ** 0.5  # Euclidean distance
            return movement > 1.0  # Consider movement if > 1 pixel delta
        except:
            self.frame_mouse_delta = (0, 0)
            return False
    
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
        Record human gameplay with camera orientation.
        
        Args:
            duration: How long to record in seconds
        """
        print(f"\n{'='*70}")
        print("🎮 ENHANCED RECORDING - GAMEPLAY + CAMERA ORIENTATION")
        print(f"{'='*70}")
        print(f"\nRecording for {duration} seconds...")
        print("This version captures:")
        print("  ✓ Your button presses (W/A/S/D)")
        print("  ✓ Your mouse movements (camera angle)")
        print("  ✓ Game state (health, stamina, etc)\n")
        print("Key Controls:")
        print("  W/A/S/D  - Move (combined with camera = direction)")
        print("  Mouse    - Rotate camera (IMPORTANT!)")
        print("  E        - Interact")
        print("  Q        - Lock-on")
        print("  X/SPACE  - Dodge")
        print("  F        - Jump")
        print("  ESC      - Stop early")
        print(f"\n{'='*70}\n")
        
        # Countdown
        print("⏱️  Recording starts in:")
        for i in range(3, 0, -1):
            print(f"  {i}...", end='', flush=True)
            time.sleep(1)
        print("\n🔴 RECORDING NOW! (Camera movements being tracked)")
        print(f"{'='*70}\n")
        
        self.recording = True
        self.current_recording = []
        self.camera_angle_degrees = 0
        self.last_mouse_pos = self.mouse_controller.position
        
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
                # Update camera angle from mouse and detect if camera moved
                camera_moved = self.update_camera_angle()
                
                # Get game state
                state = self.game_interface.get_game_state()
                
                # Get current action (pass camera movement info)
                action = self.get_current_action(camera_moved=camera_moved)
                
                # Record with RAW SCREENSHOT for CNN + LSTM behavioral cloning
                # Store the raw_screen image along with action and camera data
                self.current_recording.append({
                    'raw_screen': state['raw_screen'],  # Raw RGB image for CNN processing
                    'observation': {
                        'health': float(state['health_percent']),
                        'stamina': float(state['stamina_percent']),
                        'exits': state['exits'],
                        'is_outdoor': bool(state['is_outdoor']),
                        'in_combat': bool(state['in_combat']),
                    },
                    'action': int(action),
                    'action_name': self.action_names[action],
                    'camera_angle': float(self.camera_angle_degrees),
                    'camera_moved': bool(camera_moved),  # NEW: Track if camera actually moved
                    'timestamp': time.time() - start_time,
                })
                
                frame_count += 1
                elapsed = time.time() - start_time
                action_name = self.action_names[action]
                
                print(f"Frame {frame_count:4d} ({elapsed:5.1f}s) - {action_name:12s} | Camera: {self.camera_angle_degrees:6.1f}°", end='\r')
                
                time.sleep(0.05)  # ~20 FPS
        
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
    
    def save(self, filename="imitation_data_with_camera.pkl"):
        """
        Save recordings to file with compressed image data.
        Images are stored efficiently to avoid huge file sizes.
        """
        import cv2
        
        # Compress images before saving to reduce file size
        compressed_recordings = []
        for recording in self.recordings:
            compressed_frames = []
            for frame_data in recording['frames']:
                compressed_frame = {
                    'action': frame_data['action'],
                    'action_name': frame_data['action_name'],
                    'camera_angle': frame_data['camera_angle'],
                    'camera_moved': frame_data['camera_moved'],
                    'observation': frame_data['observation'],
                }
                
                # Compress raw screen image using PNG (lossless)
                # Store as JPEG encoded bytes to save space
                if 'raw_screen' in frame_data:
                    raw_screen = frame_data['raw_screen']
                    # Encode to JPEG (lossy but much smaller)
                    success, buffer = cv2.imencode('.jpg', raw_screen, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        compressed_frame['raw_screen_jpg'] = buffer.tobytes()
                    else:
                        print(f"Warning: Failed to compress frame")
                
                compressed_frames.append(compressed_frame)
            
            compressed_recordings.append({
                'frames': compressed_frames,
                'duration': recording['duration'],
                'frame_count': recording['frame_count'],
                'timestamp': recording['timestamp']
            })
        
        # Save compressed data
        with open(filename, 'wb') as f:
            pickle.dump({
                'recordings': compressed_recordings,
                'action_names': self.action_names,
                'has_camera_data': True,
                'has_raw_screens': True,  # NEW: Mark that raw screens are included
                'image_format': 'jpg',  # Format used for compression
            }, f)
        
        total_frames = sum(r['frame_count'] for r in self.recordings)
        print(f"✓ Saved {len(self.recordings)} recordings ({total_frames} frames) to {filename}")
        print(f"  - Includes camera orientation data")
        print(f"  - Includes raw screen images (JPEG compressed)")
        print(f"  - Ready for CNN + LSTM behavioral cloning!")


def main():
    """Interactive recording"""
    learner = CameraAwareImitationLearner()
    
    print(f"\n{'='*70}")
    print("ENHANCED BEHAVIORAL CLONING")
    print("Records button presses + camera movements")
    print(f"{'='*70}\n")
    
    while True:
        print("1. Record gameplay with camera tracking")
        print("2. Quit")
        choice = input("Choose: ").strip()
        
        if choice == "1":
            duration_input = input("Duration to record (seconds) [default 120]: ").strip()
            duration = int(duration_input) if duration_input else 120
            
            print(f"\n{'='*70}")
            print("⚠️  ABOUT TO START RECORDING")
            print(f"{'='*70}")
            print(f"You will record for {duration} seconds")
            print("IMPORTANT: Move your camera (mouse) as you play!")
            print("The AI needs to see the camera movements to understand direction\n")
            
            ready = input("Ready? Press ENTER to start: ")
            
            learner.record(duration)
            learner.save()
            
            print("\n✓ Recording saved with camera data!")
            print("  File: imitation_data_with_camera.pkl")
        
        elif choice == "2":
            break
    
    print("\nExiting...")


if __name__ == "__main__":
    main()
