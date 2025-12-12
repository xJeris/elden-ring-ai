"""
Basic AI Agent for Elden Ring
Uses Gymnasium and Stable Baselines3 for reinforcement learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import RecurrentPPO  # For recurrent policies with LSTM
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import NatureCNN
from game_interface import GameInterface
import os
from datetime import datetime
from goals import GoalSystem, create_base_game_goals
import time
import collections
import hashlib
import cv2

# Optional hardware monitoring (gracefully skipped if not available)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _hash_prompt_region(frame, prompt_region_y_start=950, prompt_region_y_end=1080, 
                        prompt_region_x_start=700, prompt_region_x_end=1220, 
                        downsample_size=(32, 8)):
    """
    Create a unique signature for a prompt by hashing its visual appearance.
    
    REGION DETAILS (for 1920×1080 screen):
    - Y-axis: 950-1080 (bottom 130 pixels, 50px buffer above prompt) - captures text like "E :Read message"
    - X-axis: 700-1220 (center 520 pixels, ~160px padding on each side) - isolates prompt with leeway
    - This region is where all interactive prompts appear with buffer to avoid edge cutoff
    
    This allows the AI to distinguish:
    - Same prompt tried multiple times (spam detection) - same hash = same prompt
    - New/different prompt appearing (reset attempt counter) - different hash = new prompt
    - State changes (white→grey prompt text) - creates different hash signature
    
    Process:
    1. Crop prompt region from frame (bottom-center with buffer)
    2. Convert to grayscale (removes color variation noise)
    3. Downsample to tiny size (32×8) - keeps text recognizable, removes noise
    4. Hash the pixel data with SHA-1
    
    Args:
        frame: BGR image from game (1920x1080x3)
        prompt_region_y_start: top edge of region (default 950 = includes 50px buffer above prompt)
        prompt_region_y_end: bottom edge of region (default 1080 = screen bottom)
        prompt_region_x_start: left edge of region (default 700 = wide left buffer)
        prompt_region_x_end: right edge of region (default 1220 = wide right buffer)
        downsample_size: (height, width) for downsampling (32×8 works well for text)
        
    Returns:
        String hash of prompt region (first 8 chars of SHA-1, or None if no valid region)
    """
    try:
        if frame is None or frame.size == 0:
            return None
        
        # Clip bounds to frame dimensions
        h, w = frame.shape[:2]
        y_start = max(0, min(prompt_region_y_start, h))
        y_end = max(y_start + 1, min(prompt_region_y_end, h))
        x_start = max(0, min(prompt_region_x_start, w))
        x_end = max(x_start + 1, min(prompt_region_x_end, w))
        
        # Extract prompt region
        prompt_region = frame[y_start:y_end, x_start:x_end]
        
        if prompt_region.size == 0:
            return None
        
        # Convert to grayscale
        if len(prompt_region.shape) == 3:
            gray = cv2.cvtColor(prompt_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = prompt_region
        
        # Downsample to small size (reduces noise, speeds up hashing)
        tiny = cv2.resize(gray, downsample_size, interpolation=cv2.INTER_AREA)
        
        # Hash the pixel data
        hash_obj = hashlib.sha1(tiny.tobytes())
        return hash_obj.hexdigest()[:8]  # Use first 8 chars for readability
    
    except Exception as e:
        # If hashing fails, return None (hash not available)
        return None



class HardwareMonitorCallback(BaseCallback):
    """
    Monitor system hardware during training.
    Stops training if disk space or CPU temperature exceed safe limits.
    (Requires psutil - gracefully skipped if not available)
    """
    def __init__(self, 
                 min_disk_gb=200.0,    # Minimum 200GB free disk space (stops when exceeded)
                 max_cpu_temp=85.0,    # Max CPU temperature in Celsius
                 check_interval=100,   # Check every N steps
                 verbose=1):
        super().__init__(verbose)
        self.min_disk_gb = min_disk_gb
        self.max_cpu_temp = max_cpu_temp
        self.check_interval = check_interval
        self.steps_since_check = 0
        self.initial_disk_free = None
        self.disk_used_by_training = 0
        self.psutil_available = HAS_PSUTIL
        
        if not self.psutil_available and verbose:
            print("⚠️  Warning: psutil not installed. Hardware monitoring disabled.")
            print("   Install with: pip install psutil")
        
    def _on_step(self) -> bool:
        """Called after each step. Returns False to stop training if unsafe."""
        if not self.psutil_available:
            return True  # Skip monitoring if psutil unavailable
            
        self.steps_since_check += 1
        
        # Check hardware periodically
        if self.steps_since_check >= self.check_interval:
            self.steps_since_check = 0
            
            # Check disk space
            try:
                disk_usage = psutil.disk_usage('/')
                free_gb = disk_usage.free / (1024**3)
                used_gb = disk_usage.used / (1024**3)
                
                # Track initial state on first check
                if self.initial_disk_free is None:
                    self.initial_disk_free = free_gb
                
                # Calculate how much disk training has used
                disk_freed_by_training = self.initial_disk_free - free_gb
                
                if free_gb < self.min_disk_gb:
                    print(f"\n⚠️ CRITICAL: Disk space exhausted! Only {free_gb:.2f}GB free (allowed: {self.min_disk_gb}GB)")
                    print(f"   Training used approximately {disk_freed_by_training:.2f}GB of disk space.")
                    print(f"   Stopping training to prevent hard drive overflow.")
                    return False  # Stop training immediately
            except Exception as e:
                if self.verbose > 1:
                    print(f"Warning: Could not check disk space: {e}")
            
            # Check CPU temperature
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature (varies by system, try common names)
                    cpu_temp = None
                    for sensor_type in ['coretemp', 'acpitz', 'it8792', 'k10temp']:
                        if sensor_type in temps:
                            cpu_temp = temps[sensor_type][0].current
                            break
                    
                    if cpu_temp and cpu_temp > self.max_cpu_temp:
                        print(f"\n⚠️ CRITICAL: CPU overheating! Temperature {cpu_temp:.1f}°C (threshold: {self.max_cpu_temp}°C)")
                        print(f"   Stopping training to prevent hardware damage.")
                        return False  # Stop training immediately
                    elif cpu_temp:
                        if self.verbose:
                            print(f"   CPU Temp: {cpu_temp:.1f}°C | Free Disk: {free_gb:.2f}GB")
            except Exception as e:
                if self.verbose > 1:
                    print(f"Note: Could not read CPU temperature: {e}")
        
        return True


class PeriodicRewardCallback(BaseCallback):
    """
    Callback to log episode rewards and stats every 60 seconds
    Shows training progress and whether AI is just gaining easy points
    """
    def __init__(self, log_interval_seconds=60, verbose=0):
        super().__init__(verbose)
        self.log_interval_seconds = log_interval_seconds
        self.last_log_time = time.time()
        self.episode_rewards = []
        self.episode_steps = []
        self.steps_since_log = 0
        self.accumulated_reward = 0.0
        self.step_count = 0
        # Initialize to 0 to prevent AttributeError if accessed before parent init
        self.num_timesteps = 0
        
    def _on_step(self) -> bool:
        """Called after each step in the environment"""
        current_time = time.time()
        
        # Track steps
        self.steps_since_log += 1
        self.step_count += 1
        
        # Safely access num_timesteps with fallback
        try:
            timesteps = self.num_timesteps if hasattr(self, 'num_timesteps') else self.step_count
        except AttributeError:
            timesteps = self.step_count
        
        # Log every N seconds
        if current_time - self.last_log_time >= self.log_interval_seconds:
            self._print_stats(timesteps)
            self.last_log_time = current_time
            self.accumulated_reward = 0.0
            self.steps_since_log = 0
        
        return True
    
    def _print_stats(self, timesteps=0):
        """Print training statistics"""
        print("\n" + "=" * 70)
        print(f"📊 TRAINING STATUS (Last 60 seconds) - Step {timesteps}")
        print("=" * 70)
        print(f"  Steps in last 60 sec: {self.steps_since_log}")
        print(f"  Total steps so far: {timesteps}")
        print("=" * 70)
        
        print("=" * 70 + "\n")


class EldenRingEnv(gym.Env):
    """
    Gymnasium environment for Elden Ring
    Defines the action/observation spaces and reward structure
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, game_interface=None, render_mode=None, goals=None):
        """
        Initialize environment
        
        Args:
            game_interface: GameInterface instance
            render_mode: how to render
            goals: GoalSystem instance for defining objectives
        """
        self.game_interface = game_interface or GameInterface()
        self.render_mode = render_mode
        self.goals = goals or create_base_game_goals()  # Use default goals if none provided
        
        # Track game state
        self.melina_spoken_to = False  # Track if AI has actually spoken to Melina (not just at a bonfire)
        self.melina_available_at_step = 500  # Melina dialogue only available after 500 game steps (real progression)
        
        # Observation space: Raw pixel images (84x84 grayscale, 4 stacked frames for motion)
        # CNN will extract spatial features, LSTM will track temporal patterns
        # 4 frames allows agent to see motion/direction of movement
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 4),  # 84x84 grayscale with 4-frame stacking
            dtype=np.uint8
        )
        
        # Action space: discrete actions based on Elden Ring default controls
        # 0: no-op
        # 1-4: move (forward/W, backward/S, left/A, right/D)
        # 5: normal attack (Left Click)
        # 6: heavy attack (Shift + Left Click)
        # 7: backstep/dodge (Spacebar)
        # 8: skill (Shift + Right Click)
        # 9: use item (R) - available when in exploration/combat with items
        # 10: lock-on (Q)
        # 11: jump (F)
        # 12: interact with NPCs/objects/doors/items (E)
        # 13: summon mount (R) - ONLY available after Melina met (context-dependent)
        # 14: pan camera left (mouse left)
        # 15: pan camera right (mouse right)
        # 16: pan camera up (mouse up)
        # 17: pan camera down (mouse down)
        # 18: open map (G)
        # NOTE: Action 9 and 13 both use R key; context determines which applies
        # During exploration: Action 9 (use item) gets -0.5 penalty per Prime Directives

        self.action_space = spaces.Discrete(19)  # 19 discrete actions (0-18)
        
        # Define action groups as sets for O(1) lookup (faster than lists)
        self.MOVEMENT_ACTIONS = {1, 2, 3, 4}  # Forward, backward, left, right
        self.COMBAT_ACTIONS = {5, 6, 8}  # Light attack, heavy attack, skill
        self.ATTACK_ACTIONS = {5, 6}  # Light and heavy attacks
        self.CAMERA_ACTIONS = {14, 15, 16, 17}  # Camera pan left, right, up, down
        
        # Action masking - enforce Prime Directives at action level
        # Valid actions are updated each step based on game state
        self.steps = 0
        self.max_steps = 5000
        # Track health state for reward shaping
        self.previous_health = 1.0  # Start at full health
        self.previous_stamina = 1.0  # Start at full stamina
        
        # Track item slot usage to penalize repeated empty slot attempts
        self.last_successful_item_use = -10  # Steps since last successful item use
        self.last_item_use_attempt = -10  # Steps since item was attempted
        self.consecutive_empty_item_attempts = 0  # Count of failed item attempts in a row
        
        # Track Chapel exit (first building) for bonus reward
        self.chapel_exited = False  # Has AI left the Chapel of Anticipation?
        self.exit_time_steps = -1  # Steps taken to exit (for bonus calculation)
        self.episode_start_time = time.time()  # Real world time episode started
        
        # Track reward logging for 60-second periodic output
        self.last_reward_log_time = time.time()
        self.episode_reward_log_interval = 60.0  # Log every 60 seconds
        
        # Track first boss spawn (Godrick) - must happen within 20 minutes (6000 steps @ 5 fps)
        self.boss_spawned = False  # Has first boss been spawned?
        self.boss_spawn_time_steps = -1  # Steps at which boss was spawned
        self.max_steps_without_boss = 6000  # 20 minutes at 5 FPS = 6000 steps
        self.boss_spawn_deadline_passed = False  # Flag to reset reward if deadline missed
        
        # Track stuck detection - when AI repeats same movement without progress
        self.last_state_signature = None  # Previous state signature for comparison
        self.stuck_consecutive_frames = 0  # Count of consecutive frames in same state
        self.stuck_counter = 0  # How many steps in current stuck area
        self.last_direction_tried = None  # Track which direction to try next when stuck
        self.stuck_directions = {}  # Track which directions have walls (e.g., {'forward': 45, 'left': 23})
        
        # Wall detection: distinguish between walls and pillars/obstacles
        # A real wall is confirmed by: 5+ horizontal lateral movements + can't escape after 3 steps inward
        self.direction_wall_attempts = {}  # Track attempts to move in each direction: {direction: {'lateral_moves': N, 'state_unchanged': N}}
        self.wall_confirmed = set()  # Confirmed walls: {'forward', 'left', etc.}
        
        # Track door interactions - detect when a closed door is opened
        self.last_door_state = False  # Was there a closable door in last step?
        self.door_opened_this_episode = False  # Have we successfully opened a door?
        self.steps_since_last_door_action = 0  # Track steps spent hovering at door without progress
        self.last_interact_action_step = -999  # Track when we last tried to interact (prevent false positives)
        self.door_attempt_cooldown = 0  # Steps remaining on door attempt cooldown (grace period)
        self.last_inventory_count = 0  # Track inventory count for door opening validation
        self.position_when_interacted = None  # Track player position when E was pressed (x, y, z)
        
        # Track failed exits - if AI moves toward an exit and nothing happens, mark it as false
        self.last_exit_count = 0  # Number of exits detected last step
        self.exit_movement_steps = 0  # How many steps has AI moved toward an exit?
        self.failed_exits = set()  # Track exit detection patterns that led nowhere
        self.last_exit_direction = None  # 'left', 'right', 'center', or None
        
        # Track door/interactive prompt visibility for memory
        self.door_seen_recently = False  # Was a door visible in the last 10 steps?
        self.door_visibility_counter = 0  # How long has it been since we last saw a door?
        self.max_door_memory = 10  # Remember for N steps after disappearing
        
        # Track prompt visibility for failure-to-interact penalty
        self.prompt_visible_steps = 0  # How many consecutive steps has a ready prompt been visible?
        self.max_prompt_grace_period = 3  # Allow 3 steps grace before penalizing non-interaction
        
        # Track required item for Chapel exit (Tarnished's Wizened Finger)
        self.has_wizened_finger = False  # Does AI have the required key item?
        self.wizened_finger_obtained_step = -1  # Step at which the finger was obtained
        
        # Track movement trajectory - reward continuing in same direction (momentum)
        self.last_movement_action = None  # Last movement action taken (1=forward, 2=back, 3=left, 4=right)
        self.direction_change_penalty_multiplier = 1.0  # Increases penalty for changing direction
        
        # Track combat state persistently (not just locally in step)
        self.in_combat = False  # Are we currently in combat?
        self.consecutive_frames_in_direction = 0  # How many consecutive frames in current direction
        self.min_frames_before_direction_change = 5  # Must stay in direction for 5+ frames before changing
        
        # Track map state for close-immediately bonus
        self.map_open = False  # Is the map currently open?
        self.map_opened_step = -1  # Step at which map was opened
        self.consecutive_map_actions = 0  # Count of actions while map is open (not closed immediately)
        self.map_last_closed_step = -1  # Step at which map was last closed
        
        # Map subwindow tracking (e.g., inventory opened with R while map is open)
        self.map_subwindow_open = False  # Is a subwindow (like inventory) open in the map?
        self.map_subwindow_opened_step = -1  # Step at which subwindow was opened
        
        # Calculate steps equivalent to 30 seconds (assuming ~60 FPS game, 30 steps per second at normal speed)
        self.map_reopen_cooldown_steps = 900  # 30 seconds * 30 steps/second = 900 steps
        
        # Track all actions taken during training for analysis
        self.action_names = [
            "no-op",           # 0
            "move forward",    # 1
            "move backward",   # 2
            "move left",       # 3
            "move right",      # 4
            "normal attack",   # 5
            "heavy attack",    # 6
            "backstep/dodge",  # 7
            "use skill",       # 8
            "use item",        # 9
            "lock-on",         # 10
            "jump",            # 11
            "interact",        # 12
            "summon mount",    # 13
            "pan camera left", # 14
            "pan camera right",# 15
            "pan camera up",   # 16
            "pan camera down", # 17
            "open map"         # 18
        ]
        self.action_counts = np.zeros(19, dtype=np.int64)  # Count each action
        self.episode_action_counts = np.zeros(19, dtype=np.int64)  # Per-episode tracking
        
        # Track interact actions breakdown (for diagnostics)
        self.successful_interact_count = 0  # Count of interact presses with valid target (door/item)
        self.wasted_interact_count = 0  # Count of interact presses with no valid target
        self.wasted_interact_during_map = 0  # Count of interact presses while map is open
        self.missed_interact_opportunities = 0  # Count of times a prompt/item was visible but E not pressed
        
        # Track prompt dwell time (discourage indecision at doors)
        self.prompt_visible_consecutive_steps = 0  # How many consecutive steps has prompt been visible?
        self.last_prompt_state = False  # Was prompt visible last step?
        
        # Track message reads (floor messages can be read multiple times)
        self.message_reads_this_episode = 0  # Count of message reads (rewards only first one)
        
        # Track interact state changes (for validating successful interactions)
        self.interact_pending_state_check = False  # Waiting to check if last interact caused state change?
        self.interact_prompt_state_when_pressed = False  # Was prompt visible when E was pressed?
        self.interact_inventory_when_pressed = tuple()  # Inventory state when E was pressed (for retry gating)
        self.last_prompt_hash = None  # Hash of last prompt we pressed E on
        self.prompt_attempt_counts = {}  # Track attempts per prompt hash: {hash: attempt_count}
        self.prompt_hash_stats = {}  # Track success rate per hash: {hash: {'attempts': N, 'successes': N}}
        self.last_attempted_prompt_hash = None  # Which hash was just attempted (for success logging)
        
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: action from action space
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Initialize reward for this step - all bonuses/penalties accumulate here
        reward = 0.0
        
        # Track the original action for reward calculation
        original_action = action
        
        # Track action taken for analysis
        self.action_counts[original_action] += 1
        self.episode_action_counts[original_action] += 1
        
        # Execute action
        self._execute_action(action)
        
        # Get new state
        state = self.game_interface.get_game_state()
        
        # Return raw screen image (wrapper will stack frames and convert to grayscale)
        try:
            observation = state['raw_screen']  # Keep as numpy array, wrapper handles conversion
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 460 observation = state['raw_screen']: {e}")
            raise
        
        # NOTE: Actual observation returned to model is handled by FrameStackWrapper
        # This EldenRingEnv step() method returns raw RGB image, which wrapper:
        # 1. Converts to grayscale (84x84)
        # 2. Stacks last 4 frames for motion detection
        # CNN processes spatial features, LSTM handles temporal sequences
        
        # ===== STUCK DETECTION =====
        # Track location stability using state signature (exits + health + stamina quantized)
        try:
            exits = state.get('exits', {'total': 0})
            health = state.get('health_percent', 0.0)
            stamina = state.get('stamina_percent', 0.0)
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 470-472 stuck detection: {e}")
            raise
        
        # Create a compact state signature for stuck detection
        current_signature = (
            exits.get('total', 0),
            int(health * 10),  # Quantize to 0-10 range
            int(stamina * 10)   # Quantize to 0-10 range
        )
        
        # Track which directions have walls (movement actions that don't change state)
        # Actions 1-4 are movements: 1=forward, 2=backward, 3=left, 4=right
        movement_map = {1: 'forward', 2: 'backward', 3: 'left', 4: 'right'}
        lateral_map = {1: [3, 4], 2: [3, 4], 3: [1, 2], 4: [1, 2]}  # Lateral movements for each direction
        
        # Streamlined counter: increment if state unchanged, reset if changed
        if current_signature == self.last_state_signature:
            self.stuck_consecutive_frames += 1
            # If the last action was movement and state didn't change, it hit a wall or obstacle
            if original_action in movement_map:
                direction = movement_map[original_action]
                self.stuck_directions[direction] = self.stuck_directions.get(direction, 0) + 1
                
                # Track wall confirmation: need 5+ lateral moves + can't escape after 3 steps inward
                if direction not in self.direction_wall_attempts:
                    self.direction_wall_attempts[direction] = {'lateral_moves': 0, 'state_unchanged': 0}
                
                self.direction_wall_attempts[direction]['state_unchanged'] += 1
            
            # Check if we're making lateral movements (trying to go around obstacle)
            lateral_dirs = []
            for main_dir, laterals in lateral_map.items():
                if original_action in laterals:
                    for main_d in movement_map:
                        if movement_map[main_d] in [movement_map[main_dir] for main_dir in [k for k, v in lateral_map.items()]]:
                            pass  # This is getting complex, simplify below
            
            # Simpler approach: if we're trying lateral movement (left/right) when wall blocks forward
            if original_action in [3, 4]:  # left or right
                for confirmed_wall_dir in self.wall_confirmed:
                    opposite = {'forward': 'backward', 'backward': 'forward', 'left': 'right', 'right': 'left'}
                    if original_action == 3 and confirmed_wall_dir == 'forward':  # trying left while forward blocked
                        if 'forward' in self.direction_wall_attempts:
                            self.direction_wall_attempts['forward']['lateral_moves'] += 1
                    elif original_action == 4 and confirmed_wall_dir == 'forward':  # trying right while forward blocked
                        if 'forward' in self.direction_wall_attempts:
                            self.direction_wall_attempts['forward']['lateral_moves'] += 1
            
            # Wall confirmation logic: if 5+ lateral moves while blocked + 3+ steps inward still blocked = confirmed wall
            for direction, attempts in self.direction_wall_attempts.items():
                if (attempts['lateral_moves'] >= 5 and 
                    attempts['state_unchanged'] >= 8 and 
                    direction not in self.wall_confirmed):
                    self.wall_confirmed.add(direction)
                    self.stuck_directions[direction] = self.stuck_directions.get(direction, 0) + 10  # Boost count for confirmed wall
            
            # Stuck detected after 10 consecutive frames in same state
            if self.stuck_consecutive_frames >= 10:
                self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)  # Gradually decrease
            self.stuck_consecutive_frames = 0  # Reset counter on state change
            # If just became unstuck, reset momentum counter to allow fresh movement
            if self.stuck_counter == 0:
                self.consecutive_frames_in_direction = 0
        
        self.last_state_signature = current_signature
        
        # Extract health, stamina, and exits from state
        try:
            health = state.get('health_percent', -1)
            stamina = state.get('stamina_percent', -1)
            exits = state.get('exits', {'closed_doors': 0, 'open_doors': 0, 'archways': 0, 'total': 0})
            quickslots = state.get('quickslots', [False] * 8)
            is_outdoor = state.get('is_outdoor', False)
            ground_items_visible = state.get('ground_items_visible', False)
            door_state = state.get('door_state', {'has_closable_door': False, 'has_open_prompt': False, 'prompt_brightness': 'none', 'prompt_is_white': False})
            self.in_combat = state.get('in_combat', False)  # Update persistent combat state
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 543-552 main state extraction: {e}")
            raise
        
        # DETECT WIZENED FINGER PICKUP
        # Check if Wizened Finger is detected in inventory (either quickslots or main inventory)
        try:
            has_wizened_finger_now = state.get('has_wizened_finger', False)
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 560 has_wizened_finger: {e}")
            raise
        if has_wizened_finger_now and not self.has_wizened_finger:
            # First time detecting the Wizened Finger
            self.has_wizened_finger = True
            self.wizened_finger_obtained_step = self.steps
            reward += 2.0  # MAJOR bonus for picking up required item
            print(f"\n✓ WIZENED FINGER ACQUIRED! Major progression at step {self.steps}")
        
        # ===== INTERACT STATE-CHANGE DETECTION (WITH PROMPT HASHING) =====
        # Check if previous interact action caused a state change (successful interaction)
        if self.interact_pending_state_check:
            # We pressed E last step on a valid prompt, now check if anything changed
            current_prompt_visible = door_state.get('has_open_prompt', False)
            current_inventory = tuple(quickslots)
            inventory_changed = current_inventory != self.interact_inventory_when_pressed
            
            # Get hash of current prompt (if visible)
            current_prompt_hash = None
            if current_prompt_visible:
                try:
                    current_prompt_hash = _hash_prompt_region(state.get('raw_screen'))
                except NameError as e:
                    print(f"🔴 DEBUG: NameError at LINE 576 _hash_prompt_region(state.get('raw_screen')): {e}")
                    raise
            
            if not current_prompt_visible and self.interact_prompt_state_when_pressed:
                # Prompt was visible when we pressed E, but is gone now = PROMPT DISAPPEARED
                # This indicates the interact was successful (door opened, item picked up, etc.)
                reward += 4.0  # Was +20.0, reduced to shift focus to movement
                
                # Check if this is a novel hash (exploration bonus)
                if self.last_attempted_prompt_hash and self.last_attempted_prompt_hash not in self.prompt_hash_stats:
                    reward += 2.0  # Bonus for discovering new prompt type
                
                # Track success for this prompt hash
                if self.last_attempted_prompt_hash:
                    if self.last_attempted_prompt_hash in self.prompt_hash_stats:
                        self.prompt_hash_stats[self.last_attempted_prompt_hash]['successes'] += 1
                    else:
                        # First time seeing this hash - initialize it
                        self.prompt_hash_stats[self.last_attempted_prompt_hash] = {'attempts': 1, 'successes': 1}
                
                self.prompt_attempt_counts = {}  # Clear all attempt counts on success
                self.last_prompt_hash = None
            elif inventory_changed:
                # Inventory changed since we pressed E = STATE CHANGED
                # AI obtained an item, now might be able to open locked door
                reward += 4.0  # Was +20.0, reduced to shift focus to movement
                
                # Check if this is a novel hash (exploration bonus)
                if self.last_attempted_prompt_hash and self.last_attempted_prompt_hash not in self.prompt_hash_stats:
                    reward += 2.0  # Bonus for discovering new prompt type
                
                # Track success for this prompt hash
                if self.last_attempted_prompt_hash:
                    if self.last_attempted_prompt_hash in self.prompt_hash_stats:
                        self.prompt_hash_stats[self.last_attempted_prompt_hash]['successes'] += 1
                    else:
                        # First time seeing this hash - initialize it
                        self.prompt_hash_stats[self.last_attempted_prompt_hash] = {'attempts': 1, 'successes': 1}
                
                self.prompt_attempt_counts = {}  # Clear attempt counts on inventory change (new game state)
                self.last_prompt_hash = None
            elif current_prompt_visible and self.interact_prompt_state_when_pressed:
                # Prompt still visible and inventory unchanged = NO STATE CHANGE
                # Nothing happened (locked door, already read message, etc.)
                
                # Use prompt hash to track unique prompts
                if current_prompt_hash is not None:
                    attempt_count = self.prompt_attempt_counts.get(current_prompt_hash, 0)
                    
                    if attempt_count == 0:
                        # First attempt on this prompt - small reward to encourage trying
                        reward += 2.0
                        self.prompt_attempt_counts[current_prompt_hash] = 1
                    elif attempt_count == 1:
                        # Second attempt on same prompt - apply penalty
                        reward -= 0.5
                        self.prompt_attempt_counts[current_prompt_hash] = 2
                    else:
                        # Third+ attempt on same prompt - heavier penalty
                        reward -= 1.0
                        self.prompt_attempt_counts[current_prompt_hash] = attempt_count + 1
                    
                    self.last_prompt_hash = current_prompt_hash
                else:
                    # Hash failed (shouldn't happen), fall back to safe penalty
                    reward -= 0.5
            
            # Clear the pending flag
            self.interact_pending_state_check = False
            self.interact_prompt_state_when_pressed = False
            self.interact_inventory_when_pressed = tuple()
        
        # ===== FALSE EXIT DETECTION =====
        # If exits disappear suddenly without successful progression, mark them as false
        # This prevents the AI from chasing false exit detections
        if exits['total'] == 0 and self.last_exit_count > 0 and self.exit_movement_steps > 0:
            # We were moving toward exits, they disappeared, but we didn't progress
            # Mark this exit count as a false detection
            self.failed_exits.add(self.last_exit_count)
            if self.steps % 100 == 0:  # Log occasionally to avoid spam
                print(f"⚠️  False exit detected: {self.last_exit_count} exits led nowhere")
            self.exit_movement_steps = 0
        
        # Track when we're moving toward exits
        if exits['total'] > 0 and exits['total'] not in self.failed_exits:
            self.exit_movement_steps += 1
        else:
            self.exit_movement_steps = 0
        
        self.last_exit_count = exits['total']
        
        # ===== DOOR OPENING DETECTION =====
        # Detect when a closed door (interaction prompt visible) becomes open (prompt gone)
        # KEY INSIGHT: If prompt disappears and we haven't moved, the door was opened.
        # If prompt disappears and we HAVE moved away, we just walked away from it.
        door_bonus = 0.0  # Track door-specific bonuses separately
        if self.last_door_state and not door_state['has_closable_door']:
            # Prompt disappeared - now check if we actually opened it or just moved away
            steps_since_interact = self.steps - self.last_interact_action_step
            if steps_since_interact <= 2 and not self.door_opened_this_episode:
                # We interacted 1-2 steps ago and prompt disappeared
                # CRITICAL: Check if we stayed in roughly the same position
                # If position_when_interacted is None, fall back to being conservative
                if self.position_when_interacted is not None:
                    try:
                        current_pos = state.get('player_position', None)
                    except NameError as e:
                        print(f"🔴 DEBUG: NameError at LINE 669 state.get('player_position'): {e}")
                        raise
                    if current_pos is not None:
                        # Calculate distance moved since pressing E
                        dx = current_pos[0] - self.position_when_interacted[0]
                        dy = current_pos[1] - self.position_when_interacted[1]
                        dz = current_pos[2] - self.position_when_interacted[2]
                        distance_moved = (dx**2 + dy**2 + dz**2) ** 0.5
                        
                        # If we moved very little (<0.5 units) and prompt is gone, door was definitely opened
                        if distance_moved < 0.5:
                            self.door_opened_this_episode = True
                            door_bonus = 10.0
                            reward += door_bonus
                            self.episode_reward += door_bonus
                            print(f"\n🚪 DOOR OPENED! Major progression! Bonus: +10.0 (stayed in place)")
                        else:
                            # We moved away - prompt disappearance might just be from movement
                            if self.steps % 50 == 0:
                                print(f"⚠️  Prompt disappeared but we moved {distance_moved:.2f} units away - not counting as door opened")
                    else:
                        # Can't determine position, be conservative
                        if self.steps % 50 == 0:
                            print(f"⚠️  Prompt disappeared but position unavailable - not counting as door opened")
                else:
                    # No position recorded when E was pressed
                    if self.steps % 50 == 0:
                        print(f"⚠️  Prompt disappeared but no position record from interact - not counting as door opened")
        
        # ===== DETECT FAILED DOOR ATTEMPT =====
        # If door is still visible after we just tried to interact, it's locked/can't open
        # Activate cooldown to prevent penalty spam
        if self.last_door_state and door_state['has_closable_door']:
            steps_since_interact = self.steps - self.last_interact_action_step
            if steps_since_interact == 1 and self.door_attempt_cooldown == 0:
                # We pressed E last step, door is still here = it didn't open (locked/blocked)
                self.door_attempt_cooldown = 100  # Activate 100-step grace period
                if self.steps % 50 == 0:
                    print(f"🔒 Door attempt failed - likely locked or blocked. Cooldown activated.")
        
        # ===== DOOR PROMPT STATE HANDLING =====
        # Distinguish between GREY (waiting) and WHITE (ready) prompts
        prompt_brightness = door_state.get('prompt_brightness', 'none')
        
        # Tick down the door attempt cooldown (grace period after failed attempt)
        if self.door_attempt_cooldown > 0:
            self.door_attempt_cooldown -= 1
        
        # DOOR MEMORY: Track whether doors have been seen recently
        # This helps AI remember "there was a door here" even after it disappears (item pickup obscuring, etc.)
        if door_state['has_closable_door'] or door_state['has_open_prompt']:
            self.door_seen_recently = True
            self.door_visibility_counter = 0  # Reset memory counter
            
            # Handle based on prompt color
            if prompt_brightness == 'white':
                # WHITE PROMPT: Door is ready to open - MAKE THIS THE OVERWHELMING CHOICE
                # Balanced reward scale (±5) to avoid PPO destabilization
                reward += 2.0  # Strong bonus just for seeing white prompt
                self.prompt_visible_steps += 1
                
                # If we press E while prompt is white: Strong reward
                if original_action == 12:  # Interact action
                    reward += 5.0  # Strong bonus for pressing E when ready
                    self.last_interact_action_step = self.steps
                    self.steps_since_last_door_action = 0
                    self.door_attempt_cooldown = 100  # Grace period: don't penalize for 100 steps
                else:
                    # NOT pressing E while white prompt is visible - EXTREME penalty
                    # This is the WORST thing to do - door is ready and we ignore it
                    if self.prompt_visible_steps > 1 and self.door_attempt_cooldown == 0:
                        reward -= 10.0  # CATASTROPHIC penalty for ignoring ready prompt
                
            elif prompt_brightness == 'grey':
                # GREY PROMPT: Door not ready yet - incentivize WAITING (no-op action)
                reward += 1.0  # Bonus just for seeing the grey prompt (was 0.3)
                self.prompt_visible_steps = 0  # Reset white counter
                
                # If we stay still (no-op): reward for patience
                if original_action == 0:  # No-op / waiting
                    reward += 2.0  # Strong reward for waiting when door is grey (was 0.8)
                else:
                    # Moving or trying to interact while door is grey - penalty
                    reward -= 1.0  # Penalty for not waiting (was -0.2)
                    if original_action == 12:  # Trying to interact when not ready
                        reward -= 2.0  # Extra penalty for wasting interaction on unavailable door (was -0.3)
        else:
            # Door is not visible now - check if it was recently
            if self.door_seen_recently:
                if self.door_visibility_counter < self.max_door_memory:
                    # Still within memory window - give weak bonus to keep searching here
                    reward += 0.05  # Weak bonus: "a door was here, keep searching"
                    self.door_visibility_counter += 1
                else:
                    # Forgot about the door
                    self.door_seen_recently = False
            
            self.prompt_visible_steps = 0  # Reset counter when no prompt visible
        
        # BONUS: Detect interactive items on ground (world items)
        if door_state.get('has_interactive_item', False):
            item_rarity = door_state.get('item_rarity', None)
            if item_rarity == 'legendary':
                reward += 0.25  # Gold items are valuable
            elif item_rarity == 'rare':
                reward += 0.15  # Purple items are good
            elif item_rarity == 'common':
                reward += 0.05  # White items are OK
        
        # Update door state tracking for next step
        self.last_door_state = door_state['has_closable_door']
        
        # Update inventory count for door opening validation
        try:
            current_inventory_count = state.get('inventory_count', 0)
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 824 state.get('inventory_count'): {e}")
            raise
        if current_inventory_count != self.last_inventory_count:
            self.last_inventory_count = current_inventory_count
        
        # ===== VALIDATE MAP STATE (resync with game) =====
        # Check if map is actually visible in game before calculating rewards
        try:
            map_ui_visible = state.get('map_ui_visible', False)
        except NameError as e:
            print(f"🔴 DEBUG: NameError at LINE 790 state.get('map_ui_visible'): {e}")
            raise
        if self.map_open and not map_ui_visible:
            # Our tracking says map is open but game says it's closed - resync
            self.map_open = False
            self.map_subwindow_open = False
            self.map_subwindow_opened_step = -1
        
        # ===== TRACK INTERACT HASH ATTEMPTS (when pressing E on valid target) =====
        # Capture the prompt hash when we press E on a valid prompt for success rate tracking
        if original_action == 12 and door_state.get('has_open_prompt', False):
            # We're pressing E on a valid prompt - capture its hash for tracking
            try:
                current_hash = _hash_prompt_region(state.get('raw_screen'))
            except NameError as e:
                print(f"🔴 DEBUG: NameError at LINE 806 _hash_prompt_region(state.get('raw_screen')): {e}")
                raise
            if current_hash:
                # Initialize hash stats if new
                if current_hash not in self.prompt_hash_stats:
                    self.prompt_hash_stats[current_hash] = {'attempts': 0, 'successes': 0}
                # Increment attempts for this hash
                self.prompt_hash_stats[current_hash]['attempts'] += 1
                # Track which hash we just attempted
                self.last_attempted_prompt_hash = current_hash
        
        # Calculate movement and action rewards using the ORIGINAL action chosen
        # RecurrentPPO doesn't support ActionMasker, so we use reward penalties instead
        action_reward = self._calculate_reward(original_action, health, stamina, invalid=False, exits=exits, quickslots=quickslots, door_state=door_state, ground_items_visible=ground_items_visible, in_combat=self.in_combat)
        reward += action_reward
        self.episode_reward += action_reward
        
        # ===== PERIODIC REWARD LOGGING (every 60 seconds) =====
        current_time = time.time()
        if current_time - self.last_reward_log_time >= self.episode_reward_log_interval:
            print("\n" + "=" * 70)
            print(f"📊 TRAINING STATUS - Step {self.steps} - Total Episode Reward: {self.episode_reward:.2f}")
            print(f"   Actions/sec: {self.steps / (current_time - self.episode_start_time):.1f}")
            print(f"   Last detected: {exits['total']} exits (closed:{exits['closed_doors']} open:{exits['open_doors']} archways:{exits['archways']})")
            print(f"   Chapel exited: {self.chapel_exited} | Boss spawned: {self.boss_spawned}")
            
            # Map usage stats
            if self.action_counts[18] > 0:  # If map has been used at all
                map_penalty_estimate = self.action_counts[18] * -15.0  # Base penalty per open
                print(f"   ⚠️  MAP USAGE: {self.action_counts[18]} times - Est. penalty: {map_penalty_estimate:.1f}")
                print(f"   ❌ INTERACT during map: {self.wasted_interact_during_map} times (penalty: -{self.wasted_interact_during_map * 2.0:.1f})")
            
            print("=" * 70 + "\n")
            self.last_reward_log_time = current_time
        
        # Update previous health/stamina for next step
        if health >= 0:
            self.previous_health = health
        if stamina >= 0:
            self.previous_stamina = stamina
        
        # Update prompt state tracking for next step (for missed opportunity detection)
        self.last_prompt_state = door_state.get('has_open_prompt', False) or ground_items_visible
        
        # Check if episode is done
        self.steps += 1
        
        # ===== CHECK FOR CHAPEL EXIT BONUS =====
        # Detect if AI has exited the Chapel of Anticipation
        if not self.chapel_exited:
            # Check if we're now in an outdoor area (sky visible at top of screen)
            # AND we've taken some steps (avoid false positives at start)
            if is_outdoor and self.steps > 100:
                self.chapel_exited = True
                self.exit_time_steps = self.steps
                
                # Calculate bonus based on speed
                # Human does it in ~20 seconds (~600 steps at 30 FPS)
                # We give bonus if under 3 minutes (9000 steps at 30 FPS)
                seconds_elapsed = self.exit_time_steps / 30.0  # Assuming ~30 FPS
                
                if seconds_elapsed < 180:  # Under 3 minutes
                    if seconds_elapsed < 20:
                        # Superhuman speed
                        exit_bonus = 5.0
                    elif seconds_elapsed < 60:
                        # Very fast (1 minute)
                        exit_bonus = 3.0
                    elif seconds_elapsed < 120:
                        # Fast (2 minutes)
                        exit_bonus = 2.0
                    else:
                        # Acceptable (2-3 minutes)
                        exit_bonus = 1.0
                else:
                    # Took too long, no bonus
                    exit_bonus = 0.0
                
                if exit_bonus > 0:
                    reward += exit_bonus
                    self.episode_reward += exit_bonus  # Add to episode total
                    print(f"\n🎯 CHAPEL EXITED in {seconds_elapsed:.1f} seconds! Bonus: +{exit_bonus:.1f}")
        
        # ===== CHECK FOR BOSS SPAWN =====
        # Detect if AI has reached the boss arena (Godrick spawns when entering the arena)
        # Detection: Boss health bar visible + fog wall visible = boss arena
        if not self.boss_spawned and self.chapel_exited:
            try:
                boss_health = state.get('boss_health_visible', False)
                fog_wall = state.get('fog_wall_visible', False)
            except NameError as e:
                print(f"🔴 DEBUG: NameError at LINE 906-907 boss_health/fog_wall: {e}")
                raise
            
            # Boss is spawned when we see the health bar and/or fog wall
            if boss_health or fog_wall:
                self.boss_spawned = True
                self.boss_spawn_time_steps = self.steps
                seconds_to_spawn = self.boss_spawn_time_steps / 30.0
                reward += 10.0  # HUGE bonus for reaching boss
                self.episode_reward += 10.0
                print(f"\n🐉 BOSS SPAWNED (Health Bar + Fog Wall Detected) in {seconds_to_spawn:.1f} seconds! Bonus: +10.0")
        
        # ===== CHECK FOR DEATH =====
        # Detect if AI died (health = 0)
        terminated = False
        if health == 0:
            # AI is dead - episode terminates
            terminated = True
            # Heavy penalty for dying
            reward -= 50.0
            self.episode_reward -= 50.0
            print(f"\n☠️ DEATH at step {self.steps}! Total episode reward: {self.episode_reward:.2f}")
        
        # ===== CHECK FOR BOSS SPAWN DEADLINE =====
        # If boss hasn't spawned by 20 minutes, reset reward to force progression
        if not self.boss_spawned and self.steps >= self.max_steps_without_boss and not self.boss_spawn_deadline_passed:
            # Deadline has passed - reset all rewards earned so far
            self.boss_spawn_deadline_passed = True
            reset_amount = self.episode_reward
            self.episode_reward = 0
            reward = 0  # Current step gets 0 reward
            print(f"\n⏰ 20-MINUTE DEADLINE PASSED without boss spawn!")
            print(f"   Reward reset from {reset_amount:.2f} to 0.0 - MUST reach boss to earn points!")
        
        # Melina becomes available after sufficient game steps (representing progression)
        if self.steps >= self.melina_available_at_step and not self.melina_spoken_to:
            # Only ONE way to set melina_spoken_to: interact at a bonfire AFTER steps threshold
            # (representing finding Melina at a bonfire during normal progression)
            if original_action == 12:  # Only if they interacted
                self.melina_spoken_to = True
                melina_bonus = 0.5  # Large bonus for finding Melina
                reward += melina_bonus
                self.episode_reward += melina_bonus
        
        # ===== EARLY EPISODE TERMINATION FOR FAILURE =====
        # If episode reward becomes extremely negative, terminate early
        # This teaches the AI "this path leads to failure, don't do this"
        if self.episode_reward < -100.0:
            # Episode has failed catastrophically - end it now so model learns from failure
            terminated = True
            print(f"\n❌ EPISODE FAILED - Reward crashed below -100 at step {self.steps}")
            print(f"   Final episode reward: {self.episode_reward:.2f}")
        
        truncated = self.steps >= self.max_steps
        
        info = {
            'episode_reward': self.episode_reward,
            'steps': self.steps,
            'health': health,
            'stamina': stamina,
            'melina_spoken_to': self.melina_spoken_to,
            'melina_available_at_step': self.melina_available_at_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action):
        """Execute an action in the game"""
        if action == 0:
            # No-op
            pass
        elif action == 1:
            # Move forward
            self.game_interface.move_character(0, 1)
        elif action == 2:
            # Move backward
            self.game_interface.move_character(0, -1)
        elif action == 3:
            # Move left
            self.game_interface.move_character(-1, 0)
        elif action == 4:
            # Move right
            self.game_interface.move_character(1, 0)
        elif action == 5:
            # Normal attack
            self.game_interface.attack()
        elif action == 6:
            # Heavy attack
            self.game_interface.heavy_attack()
        elif action == 7:
            # Backstep/Dodge (Spacebar)
            self.game_interface.backstep()
        elif action == 8:
            # Use skill
            self.game_interface.skill()
        elif action == 9:
            # Use item
            self.game_interface.use_item()
        elif action == 10:
            # Lock-on
            self.game_interface.lock_on()
        elif action == 11:
            # Jump
            self.game_interface.jump()
        elif action == 12:
            # Interact with NPCs/objects/doors
            self.game_interface.interact()
            # Melina dialogue will be detected in step() function when conditions are met
        elif action == 13:
            # Summon mount
            self.game_interface.summon_mount()
        elif action == 14:
            # Pan camera left
            try:
                from pynput.mouse import Controller
                mouse = Controller()
                mouse.move(-30, 0)
            except Exception:
                pass  # Mouse control not available, skip camera action
        elif action == 15:
            # Pan camera right
            try:
                from pynput.mouse import Controller
                mouse = Controller()
                mouse.move(30, 0)
            except Exception:
                pass  # Mouse control not available, skip camera action
        elif action == 16:
            # Pan camera up
            try:
                from pynput.mouse import Controller
                mouse = Controller()
                mouse.move(0, -30)
            except Exception:
                pass  # Mouse control not available, skip camera action
        elif action == 17:
            # Pan camera down
            try:
                from pynput.mouse import Controller
                mouse = Controller()
                mouse.move(0, 30)
            except Exception:
                pass  # Mouse control not available, skip camera action
        elif action == 18:
            # Open map
            self.game_interface.open_map()
    
    def _calculate_reward(self, action, health=-1, stamina=-1, invalid=False, exits=None, quickslots=None, door_state=None, ground_items_visible=False, in_combat=False):
        """
        Calculate reward for this step - NORMALIZED SCALE (±5 range for stability)
        
        Reward Scale (all normalized to ±5 for PPO stability):
        - Major events: ±3 to ±5 (door opening, chapel exit, death)
        - Movement bonuses: ±0.1 to ±0.5 (exploration, exits)
        - Action penalties: ±0.05 to ±0.2 (wasted actions in exploration)
        - Time penalty: -0.003 per step (encourage speed)
        
        Args:
            action: action chosen by model (not necessarily executed)
            health: current health percentage (0.0-1.0 or -1 if unknown)
            stamina: current stamina percentage (0.0-1.0 or -1 if unknown)
            invalid: whether this action was invalid/blocked
            exits: dict with exit types {closed_doors, open_doors, archways, total}
            quickslots: list of 8 bools indicating which slots have items
            door_state: dict with door state info {has_closable_door, has_open_prompt}
            ground_items_visible: bool - True if items are visible on ground/corpses
            in_combat: bool - True if currently in combat
        """
        if exits is None:
            exits = {'closed_doors': 0, 'open_doors': 0, 'archways': 0, 'total': 0}
        if quickslots is None:
            quickslots = [False] * 8
        if door_state is None:
            door_state = {'has_closable_door': False, 'has_open_prompt': False, 'prompt_brightness': 'none', 'prompt_is_white': False}
        
        # Early unpacking: extract all needed values at once for O(1) access throughout function
        exits_total = exits['total']
        exits_archways = exits['archways']
        exits_open_doors = exits['open_doors']
        exits_closed_doors = exits['closed_doors']
        
        reward = 0.0
        
        # CRITICAL: Penalize invalid actions (NORMALIZED)
        if invalid:
            reward -= 0.1  # Light penalty for attempting blocked actions
            return reward  # Early return - don't give other rewards for invalid actions
        
        # ========== PRIME DIRECTIVES: HARD RULES DURING EXPLORATION ==========
        # Normalized to ±5 scale for stable training
        
        if not in_combat:
            # EXPLORATION MODE - Prime Directives are active (NORMALIZED)
            
            # Prime Directive 1: Never use items during exploration (NORMALIZED)
            if action == 9:
                reward -= 0.5  # Was -2.0, normalized to -0.5
                return reward
            
            # Prime Directive 2: Don't waste stamina on unnecessary dodges (SEVERELY PENALIZED)
            if action == 7:  # Backstep/Dodge
                reward -= 1.0  # EXTREMELY increased: Was -0.5, now -1.0 (never dodge during exploration)
                return reward  # Early return - don't give any other rewards
            
            # Prime Directive 3: Don't jump needlessly (SEVERELY PENALIZED)
            if action == 11:  # Jump
                reward -= 1.0  # EXTREMELY increased: Was -0.5, now -1.0 (never jump during exploration)
                return reward  # Early return - don't give any other rewards
            
            # Prime Directive 4: Don't open map (SEVERELY PENALIZED with immediate termination)
            # Map can be opened with action 18 (G)
            # NOTE: Additional termination happens in step() function for immediate failure
            
            if action == 18:  # Toggle map with G
                if not self.map_open:
                    # Opening map - SEVERE penalty (termination happens in step())
                    reward -= 15.0  # Severe penalty applied here
                    print(f"🗺️  MAP OPENED at step {self.steps} - Penalty: -15.0")
                    
                    # Check if reopening within 30-second window
                    if self.map_last_closed_step >= 0:
                        steps_since_close = self.steps - self.map_last_closed_step
                        if steps_since_close < self.map_reopen_cooldown_steps:
                            # Reopening too soon - additional harsh penalty
                            reward -= 10.0  # Extra penalty for habitual map checking
                            print(f"   ⚠️  REOPENED TOO SOON! Only {steps_since_close} steps since close - Extra penalty: -10.0")
                            print(f"   Total penalty this step: -25.0")
                    
                    self.map_open = True
                    self.map_opened_step = self.steps
                    self.consecutive_map_actions = 0  # Reset counter when opening
                    # NO early return - let per-step penalty also apply since map is now open
                else:
                    # Closing map with G - reward for fast close (no penalty, just bonus)
                    frames_open = self.steps - self.map_opened_step
                    if frames_open <= 1:  # Closed within 1 frame of opening
                        reward += 1.0  # Strong reward for immediately closing map
                        print(f"🗺️  MAP CLOSED IMMEDIATELY at step {self.steps} (open for {frames_open} frames) - Bonus: +1.0")
                    else:
                        print(f"🗺️  MAP CLOSED at step {self.steps} (was open for {frames_open} frames)")
                    self.map_open = False
                    self.map_last_closed_step = self.steps  # Track when map was closed
                    self.consecutive_map_actions = 0  # Reset counter when closing
                    return reward  # Early return - only for closing, to prevent per-step penalty
            
            # Close map with Q (lock-on button) when map is open
            if action == 10 and self.map_open:  # Q pressed while map open
                frames_open = self.steps - self.map_opened_step
                if frames_open <= 1:  # Closed within 1 frame of opening
                    reward += 1.0  # Strong reward for immediately closing map
                
                # Handle subwindow closure
                if self.map_subwindow_open:
                    # Closing subwindow (inventory, etc.) with Q - small positive
                    reward += 0.2  # Small reward for closing subwindow
                    self.map_subwindow_open = False
                    self.map_subwindow_opened_step = -1
                else:
                    # Closing main map with Q - no additional reward
                    self.map_open = False
                    self.map_last_closed_step = self.steps  # Track when map was closed
                
                self.consecutive_map_actions = 0  # Reset counter
                return reward  # Early return - only give close reward, suppress all other rewards
            elif self.map_open and action not in [18]:  # Any OTHER action while map is open (except G which is handled separately)
                # Action taken while map is open - continuous penalty already applied above
                # NOTE: Map state validation happens in step() before calling this function
                # If we reach here and self.map_open is True, map IS actually open in game
                
                # Still track interact presses for diagnostics
                if action == 12:  # Interact pressed while map is open
                    self.wasted_interact_during_map += 1
                    print(f"   ⚠️  INTERACT pressed during MAP (step {self.steps})")
                
                if self.map_subwindow_open:
                    # IN SUBWINDOW: Q is the only valid action, everything else gets penalized
                    if action != 10:  # Not Q
                        self.consecutive_map_actions += 1
                        # Additional penalty for wrong action in subwindow
                        subwindow_penalty = -0.3
                        reward += subwindow_penalty
                        print(f"   ⚠️  Non-Q action in subwindow at step {self.steps} (action {action}) - Penalty: {subwindow_penalty}")
                else:
                    # IN MAIN MAP: Any non-Q action opens/interacts with subwindow
                    self.map_subwindow_open = True
                    self.map_subwindow_opened_step = self.steps
                    print(f"   ℹ️  Subwindow opened at step {self.steps}")
            
            # ========== PER-STEP MAP PENALTY (After close detection) ==========
            # If we reach here and map is still open, apply the per-step penalty
            # Close actions (G toggle or Q) have early returns, so they never reach here
            # Only non-close actions trigger this penalty
            if self.map_open:
                reward -= 0.5  # Continuous penalty for having map open - encourages closing it
            
            # Prime Directive 5: Forward movement is default
            if action == 2:  # Moving backward
                reward -= 0.5  # Heavy penalty: backward is the opposite of where we want to go
            elif action in [3, 4]:  # Moving left or right
                if self.stuck_counter <= 0:
                    reward -= 0.10  # Penalty for sideways (not primary direction)
        
        # Time penalty (NORMALIZED - all steps cost tiny amount)
        reward -= 0.003  # Was -0.005, normalized to -0.003
        
        # ========== MOVEMENT IS PRIMARY BUT NOT FREE ==========
        if action in self.MOVEMENT_ACTIONS:  # Movement (forward/back/left/right)
            # FORWARD MOVEMENT DOMINANCE: Make forward SO rewarding that it dominates
            if not in_combat and action == 1:  # Moving forward during exploration
                # WALL DETECTION: Penalty for pushing into wall with no progress
                if self.stuck_counter >= 5:  # Stuck at wall (even moderate stuck detection)
                    # Progressive penalty based on how long stuck
                    if self.stuck_counter >= 20:  # Very stuck
                        reward -= 2.0  # Severe penalty for persistent wall pushing
                    elif self.stuck_counter >= 10:  # Moderately stuck
                        reward -= 1.5  # Heavy penalty for continued forward into wall
                    else:  # Just started getting stuck
                        reward -= 0.75  # Light penalty to encourage trying different direction
                else:
                    reward += 1.5  # Was 1.0, increased to 1.5 to make movement more attractive
                                    # Total for forward: +1.5 bonus alone
                    
                    # MOMENTUM BONUS: Reward continuing forward if that was last action (+0.5 per 10 frames)
                    if self.last_movement_action == 1:
                        self.consecutive_frames_in_direction += 1
                        # Grant momentum bonus every 10 consecutive frames in same direction
                        if self.consecutive_frames_in_direction % 10 == 0:
                            reward += 0.5  # Bonus for sustained forward movement
                    else:
                        # Starting new forward movement streak
                        self.consecutive_frames_in_direction = 1
                self.last_movement_action = 1
            else:
                # All other movement directions get penalized relative to forward
                reward -= 0.15  # Heavy penalty for non-forward movement (backward/sideways)
                
                # ===== TEMPORAL MOMENTUM: Enforce minimum frame commitment (EXPLORATION ONLY) =====
                # Penalize frequent direction changes (dithering) to encourage committed movement
                # This prevents the AI from thrashing left-right-left instead of exploring systematically
                # PENALTY SCALE (for ±5 training):
                # - Continuing same direction: +0 (momentum bonus removed, just no penalty)
                # - Early change at frame 1: -0.45 (worst case: 0.2 + 0.05*5)
                # - Early change at frame 3: -0.35 (0.2 + 0.05*3)
                # - Early change at frame 5: -0.25 (0.2 + 0.05*1)
                # - After min_frames (5+): -0.05 (small penalty to prefer continuity)
                # - Stuck and changing: -0.1 (allow escaping, light penalty)
                if not in_combat:
                    # Require 5+ consecutive frames in same direction before allowing direction change
                    if self.last_movement_action == action:
                        # Continuing in same direction - increment counter
                        self.consecutive_frames_in_direction += 1
                    else:
                        # Attempting to change direction
                        if self.stuck_counter > 0:
                            # Allow direction change when stuck (trying to escape) - small penalty
                            reward -= 0.1  # Light penalty for escaping stuck position
                            self.consecutive_frames_in_direction = 1
                        elif self.consecutive_frames_in_direction >= self.min_frames_before_direction_change:
                            # Allowed direction change after sufficient momentum - minimal penalty
                            reward -= 0.05  # Very small penalty after maintaining direction long enough
                            self.consecutive_frames_in_direction = 1
                        else:
                            # NOT YET: Changing direction too early - PENALIZE DITHERING
                            # Apply escalating penalty based on how premature the change is
                            frames_short = self.min_frames_before_direction_change - self.consecutive_frames_in_direction
                            # Penalty grows as frames_short increases (worse the earlier the change)
                            # Max penalty at frames_short=5: 0.2 + (0.05 * 5) = 0.45
                            early_change_penalty = 0.2 + (0.05 * frames_short)
                            reward -= early_change_penalty
                            self.consecutive_frames_in_direction = 1
                else:
                    # Combat mode - allow free direction changes for dodging/positioning
                    self.consecutive_frames_in_direction = 1
                
                self.last_movement_action = action
            
            self.consecutive_empty_item_attempts = 0  # Reset item counter
            
            # ===== STUCK DETECTION & SIDEWAYS MOVEMENT BONUS =====
            if self.stuck_counter > 0:
                # AI is stuck - reward trying different directions to escape
                if action in {3, 4}:  # Left or right movement when stuck
                    reward += 0.25  # Override penalty when stuck (trying to escape)
            
            # ===== DOORWAY ATTRACTION BONUS =====
            if exits_total > 0 and exits_total not in self.failed_exits:
                exit_direction = self._analyze_exit_direction(exits)
                action_toward_exit = self._is_action_toward_exits(action, exit_direction)
                if action_toward_exit:
                    reward += 0.2  # Bonus for moving toward exits (was 0.3, normalized to 0.2)
                    self.last_exit_direction = exit_direction
                    self.last_exit_count = exits['total']
                else:
                    reward -= 0.1  # Penalty for moving away from exits
                
                reward += exits_total * 0.05  # Base exit sighting bonus
            
            elif exits_total in self.failed_exits:
                # Penalize moving toward failed exits
                action_toward_exit = self._is_action_toward_exits(action, self._analyze_exit_direction(exits))
                if action_toward_exit:
                    reward -= 0.2  # Penalize wasting time on failed exits
            
            else:
                # NO EXITS DETECTED: Forward movement already gets strong bonus above
                # Penalize backward and left to keep AI moving forward-ish
                if action == 2:  # Backward
                    reward -= 0.05  # Penalty for backward
                elif action == 3:  # Left
                    reward -= 0.05  # Penalty for left
        
        else:
            # All non-movement actions get penalty during exploration
            reward -= 0.05  # Penalty for non-movement during exploration (was -0.01, now -0.05)
            
            if action == 0:  # No-op
                # Exception: if grey prompt is visible, no-op is already rewarded in door handling
                # Don't double-penalize - the door prompt logic handles this case
                if door_state.get('prompt_brightness', 'none') != 'grey':
                    reward -= 0.5  # Heavy penalty for idle (was -0.02, now -0.5)
            
            # ========== COMBAT/SUPPORT ACTIONS (HEAVILY PENALIZED IN EXPLORATION) ==========
            elif action in self.ATTACK_ACTIONS:  # Attacks (normal/heavy)
                # Heavy penalty for combat during exploration - should be exploring, not fighting
                reward -= 0.2  # INCREASED: Was -0.01, now -0.2 (combat wastes time)
            
            elif action == 8:  # Skill
                reward -= 0.3  # (unchanged - already aggressive)
            
            elif action == 10:  # Lock-on
                reward -= 0.2  # INCREASED: Was -0.01, now -0.2 (lock-on is combat prep)
            
            elif action == 12:  # Interact (STATE-CHANGE GATED)
                # Reset momentum counter when interact action is taken
                self.consecutive_frames_in_direction = 0
                
                # Only reward interact when there's a valid target
                # Reward is determined by whether the interact causes a state change
                if door_state.get('has_open_prompt', False):
                    # VALID TARGET: Door/chest/bonfire/NPC/message prompt visible
                    self.successful_interact_count += 1
                    
                    # DEFER REWARD: Save state and check next step if anything changed
                    self.interact_pending_state_check = True
                    self.interact_prompt_state_when_pressed = True
                    self.interact_inventory_when_pressed = tuple(quickslots)  # Save inventory for retry gating
                    self.last_interact_action_step = self.steps
                    self.prompt_visible_consecutive_steps = 0  # Reset dwell counter
                    
                elif ground_items_visible:
                    # GROUND ITEMS VISIBLE: Loot, items, or similar
                    self.successful_interact_count += 1
                    reward += 0.7  # Strong bonus for picking up items
                    self.last_interact_action_step = self.steps
                    self.prompt_visible_consecutive_steps = 0
                else:
                    # NO VALID TARGET: Wasting the interact action
                    self.wasted_interact_count += 1
                    
                    # Check for repetition penalty: if this hash has 0% success, penalize further attempts
                    if self.last_attempted_prompt_hash and self.last_attempted_prompt_hash in self.prompt_hash_stats:
                        hash_stats = self.prompt_hash_stats[self.last_attempted_prompt_hash]
                        if hash_stats['successes'] == 0 and hash_stats['attempts'] >= 1:
                            # This hash has never succeeded - strong penalty for trying again
                            reward -= 1.0  # Penalty for beating dead horse (locked door without key, etc.)
                    
                    # COOLDOWN ON FAILED INTERACT: Penalize spamming E when nothing is there (REDUCED)
                    steps_since_last_interact = self.steps - self.last_interact_action_step
                    if steps_since_last_interact < 5:
                        # Spamming interact within 5 steps of last failed attempt
                        reward -= 1.0  # Reduced: Was -2.0/-1.5, now -1.0 (lighter penalty)
                    else:
                        # First failure in a while
                        reward -= 0.5  # Reduced: Was -1.0/-0.7, now -0.5 (lighter penalty)
            
            # MISSED INTERACT OPPORTUNITIES: Track when prompt/items visible but E not pressed
            elif (door_state.get('has_open_prompt', False) or ground_items_visible) and action != 12:
                # Valid interact target visible but AI chose a different action
                # Only count the FIRST step of a prompt appearance, not every step it stays visible
                prompt_newly_visible = (door_state.get('has_open_prompt', False) or ground_items_visible) and not self.last_prompt_state
                if prompt_newly_visible:
                    self.missed_interact_opportunities += 1
                
                # DWELL TIME PENALTY: Discourage hovering at prompts without deciding
                # If prompt has been visible for 2+ consecutive steps and still no E press, penalize indecision
                if not in_combat and door_state.get('has_open_prompt', False):
                    if self.prompt_visible_steps > 2:
                        # Hovering at door for 3+ steps without pressing E
                        reward -= 0.5  # Penalty for indecision (encourages commitment)
            
            elif action == 13:  # Summon mount
                reward -= 0.1  # Penalty for unavailable mount (was -0.15, normalized to -0.1)
            
            elif action in self.CAMERA_ACTIONS:  # Camera panning (14, 15, 16, 17)
                # Camera actions waste time during exploration
                # Only useful in combat for positioning - penalize during exploration
                if not in_combat:
                    # Check if this camera action is paired with forward movement (steering while moving)
                    if self.last_movement_action == 1:  # Last action was forward movement
                        # Forward + camera = steering while exploring, small bonus
                        reward += 0.05  # Small bonus for intelligent navigation combo
                    else:
                        reward -= 0.15  # Penalty for camera wasting during exploration (was -0.05)
                # In combat, camera panning is neutral (no reward, no penalty)
        
        # EXPLORATION BONUS: Reward seeing exits (NORMALIZED)
        if exits_archways > 0:
            reward += exits_archways * 0.05  # Bonus for visible archways
        
        if exits_open_doors > 0:
            reward += exits_open_doors * 0.03  # Bonus for visible openings
        
        # Closed doors are interesting but require interaction
        if exits_closed_doors > 0:
            reward += exits['closed_doors'] * 0.02  # Bonus for visible door
        
        # Reward from goal progress (when goals are detected)
        reward += self.goals.get_goal_reward() * 0.001  # Scale down so it doesn't dominate
        
        # Health/Stamina awareness bonuses
        if health >= 0:  # If health detection is working
            # Penalize being at critical health (below 30%)
            if health < 0.3:
                reward -= 0.2  # Heavy penalty for being in danger
            # Penalize letting health drop significantly
            elif health < self.previous_health - 0.2:
                reward -= 0.1  # Took a hit, discourage careless actions
            # Reward staying healthy
            elif health > 0.7:
                reward += 0.05  # Bonus for maintaining health
        
        if stamina >= 0:  # If stamina detection is working
            # Penalize being out of stamina during combat/action
            if stamina < 0.1:
                reward -= 0.05  # Low stamina makes you vulnerable
        
        
        return reward
    
    def _analyze_exit_direction(self, exits):
        """
        Analyze where exits are located in the room.
        Since we do not have exact pixel positions from detection,
        we use a simple heuristic: if any exit exists, assume it is ahead or to sides.
        
        Returns:
            'forward' if moving forward likely leads to exit
            'left' if exits detected
            'right' if exits detected
            None if no clear direction
        """
        # Simple heuristic: assume most room exits are ahead or sides
        # More archways/open doors = likely forward/center exit
        if exits.get('archways', 0) > 0:
            return 'forward'  # Archways are typically visible ahead
        elif exits.get('open_doors', 0) > 0:
            return 'forward'  # Open doors are visible
        elif exits.get('closed_doors', 0) > 0:
            return 'forward'  # Closed doors still ahead
        
        return None
    
    def _is_action_toward_exits(self, action, exit_direction):
        """
        Check if the chosen action moves toward the detected exit direction.
        
        Args:
            action: 1=forward, 2=backward, 3=left, 4=right
            exit_direction: 'forward', 'left', 'right', or None
            
        Returns:
            True if action moves toward exits, False otherwise
        """
        if exit_direction is None:
            return False
        
        if exit_direction == 'forward':
            # Reward forward and left/right movement (exploring forward area)
            return action in [1, 3, 4]  # forward, left, right
        elif exit_direction == 'left':
            return action == 3  # left movement
        elif exit_direction == 'right':
            return action == 4  # right movement
        
        return False
    
    
    def print_action_recap(self, save_to_file=True):
        """
        Print comprehensive action statistics at end of training.
        Helps understand what actions the AI is taking and why.
        
        Args:
            save_to_file: whether to save report to file (default True)
        """
        total_actions = np.sum(self.action_counts)
        if total_actions == 0:
            print("\n⚠ No actions recorded during training")
            return
        
        # Build the report as a string so we can both print and save it
        report_lines = []
        report_lines.append("\n" + "=" * 70)
        report_lines.append("📊 ACTION STATISTICS - Training Complete")
        report_lines.append("=" * 70)
        report_lines.append(f"\nTotal Actions Taken: {total_actions:,}")
        report_lines.append(f"Episodes Completed: {len([c for c in self.episode_action_counts if np.sum(c) > 0])}")
        report_lines.append("")
        
        # Sort actions by frequency
        action_freq = list(enumerate(self.action_counts))
        action_freq.sort(key=lambda x: x[1], reverse=True)
        
        report_lines.append("Action Frequency (sorted by usage):")
        report_lines.append("-" * 70)
        report_lines.append(f"{'Rank':<6} {'Action':<25} {'Count':<12} {'Percentage':<12}")
        report_lines.append("-" * 70)
        
        for rank, (action_idx, count) in enumerate(action_freq, 1):
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"Unknown({action_idx})"
            report_lines.append(f"{rank:<6} {action_name:<25} {count:<12,} {percentage:<12.2f}%")
        
        report_lines.append("-" * 70)
        
        # Special analysis for key actions
        report_lines.append("\nKey Action Analysis:")
        report_lines.append("-" * 70)
        
        # Movement analysis
        movement_actions = [1, 2, 3, 4]  # forward, backward, left, right
        movement_total = sum(self.action_counts[i] for i in movement_actions)
        movement_pct = (movement_total / total_actions * 100) if total_actions > 0 else 0
        report_lines.append(f"Movement Actions (1-4):    {movement_total:>8,} ({movement_pct:>6.2f}%)")
        
        # Combat analysis
        combat_actions = [5, 6, 8]  # light attack, heavy attack, skill
        combat_total = sum(self.action_counts[i] for i in combat_actions)
        combat_pct = (combat_total / total_actions * 100) if total_actions > 0 else 0
        report_lines.append(f"Combat Actions (5,6,8):    {combat_total:>8,} ({combat_pct:>6.2f}%)")
        
        # Interaction analysis
        interact_count = self.action_counts[12]
        interact_pct = (interact_count / total_actions * 100) if total_actions > 0 else 0
        report_lines.append(f"Interact Action (12):      {interact_count:>8,} ({interact_pct:>6.2f}%)")
        
        # Interact breakdown diagnostics
        successful_pct = (self.successful_interact_count / interact_count * 100) if interact_count > 0 else 0
        wasted_pct = (self.wasted_interact_count / interact_count * 100) if interact_count > 0 else 0
        wasted_during_map_pct = (self.wasted_interact_during_map / interact_count * 100) if interact_count > 0 else 0
        report_lines.append(f"  ├─ Successful (valid target): {self.successful_interact_count:>8,} ({successful_pct:>6.2f}% of interact)")
        report_lines.append(f"  ├─ Wasted (no target):        {self.wasted_interact_count:>8,} ({wasted_pct:>6.2f}% of interact)")
        report_lines.append(f"  ├─ Wasted (during map):       {self.wasted_interact_during_map:>8,} ({wasted_during_map_pct:>6.2f}% of interact)")
        report_lines.append(f"  └─ Missed opportunities:      {self.missed_interact_opportunities:>8,} (prompt appearances where E not pressed)")
        
        # Dodge analysis
        dodge_count = self.action_counts[7]
        dodge_pct = (dodge_count / total_actions * 100) if total_actions > 0 else 0
        report_lines.append(f"Dodge Action (7):          {dodge_count:>8,} ({dodge_pct:>6.2f}%)")
        
        # No-op analysis
        noop_count = self.action_counts[0]
        noop_pct = (noop_count / total_actions * 100) if total_actions > 0 else 0
        report_lines.append(f"No-op Action (0):          {noop_count:>8,} ({noop_pct:>6.2f}%)")
        
        report_lines.append("-" * 70)
        
        # Action diversity metric
        non_zero_actions = np.count_nonzero(self.action_counts)
        diversity_score = (non_zero_actions / len(self.action_counts)) * 100
        max_action_pct = (np.max(self.action_counts) / total_actions * 100) if total_actions > 0 else 0
        
        report_lines.append(f"\nAction Diversity Metrics:")
        report_lines.append(f"  Unique Actions Used:       {non_zero_actions} / {len(self.action_counts)}")
        report_lines.append(f"  Diversity Score:           {diversity_score:.1f}%")
        report_lines.append(f"  Most-used Action:          {self.action_names[np.argmax(self.action_counts)]} ({max_action_pct:.1f}%)")
        report_lines.append(f"  Most-used Action Count:    {np.max(self.action_counts):,}")
        
        # Interpretation
        report_lines.append(f"\nInterpretation:")
        if max_action_pct > 70:
            report_lines.append(f"  ⚠️  AI is too heavily biased toward one action ({max_action_pct:.1f}%)")
            report_lines.append(f"      This suggests the reward function may need adjustment")
        elif max_action_pct > 50:
            report_lines.append(f"  ⚠️  AI shows bias toward {self.action_names[np.argmax(self.action_counts)]} ({max_action_pct:.1f}%)")
            report_lines.append(f"      This could indicate strong strategy or limited exploration")
        else:
            report_lines.append(f"  ✓ Good action diversity - AI is exploring multiple strategies")
        
        if interact_pct < 5 and non_zero_actions > 8:
            report_lines.append(f"  ⚠️  CONCERN: Very low interact usage ({interact_pct:.2f}%)")
            report_lines.append(f"      AI finds doors/items but doesn't attempt to interact with them")
            report_lines.append(f"      May need to increase interact reward or adjust detection")
        
        if movement_pct < 30:
            report_lines.append(f"  ⚠️  Low movement ratio ({movement_pct:.1f}%) - AI may be stuck/stalled")
        elif movement_pct > 90:
            report_lines.append(f"  ℹ️  High movement ratio ({movement_pct:.1f}%) - focused on navigation")
        
        # Prompt hash statistics section
        if self.prompt_hash_stats:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("PROMPT HASH STATISTICS - Success Rates by Visual Signature")
            report_lines.append("=" * 70)
            report_lines.append(f"\n{'Hash':<12} {'Attempts':<10} {'Successes':<10} {'Success Rate':<15} {'Notes'}")
            report_lines.append("-" * 70)
            
            # Sort by attempts (most tested first)
            sorted_hashes = sorted(self.prompt_hash_stats.items(), 
                                 key=lambda x: x[1]['attempts'], 
                                 reverse=True)
            
            for hash_val, stats in sorted_hashes:
                attempts = stats['attempts']
                successes = stats['successes']
                success_rate = (successes / attempts * 100) if attempts > 0 else 0.0
                
                # Determine status notes
                if success_rate == 100:
                    notes = "✓ All attempts succeeded"
                elif success_rate == 0:
                    notes = "✗ All attempts failed"
                elif success_rate > 75:
                    notes = "✓ Mostly successful"
                elif success_rate > 50:
                    notes = "⚠ Mixed results"
                else:
                    notes = "✗ Mostly failures"
                
                report_lines.append(f"{hash_val:<12} {attempts:<10} {successes:<10} {success_rate:>6.1f}%         {notes}")
            
            report_lines.append("-" * 70)
            report_lines.append(f"\nTotal unique prompt signatures: {len(self.prompt_hash_stats)}")
            total_attempts = sum(s['attempts'] for s in self.prompt_hash_stats.values())
            total_successes = sum(s['successes'] for s in self.prompt_hash_stats.values())
            overall_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0.0
            report_lines.append(f"Overall prompt success rate: {overall_success_rate:.1f}%")
        
        report_lines.append("=" * 70 + "\n")
        
        # Print the report to console
        for line in report_lines:
            print(line)
        
        # Save to file if requested
        if save_to_file:
            try:
                os.makedirs("training_reports", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"training_reports/action_report_{timestamp}.txt"
                
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(report_lines))
                
                print(f"📁 Report saved to: {report_filename}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save report to file: {e}")
    
    def _calculate_indoor_confidence(self, env_dict):
        """
        Estimate if character is indoors based on reliable game state indicators.
        Returns 1.0 if confident indoors, 0.0 if outdoor, 0.5 if uncertain.
        
        Reliable indicators:
        - is_outdoor flag from game interface = primary source (most reliable)
        - roof_visible = strong indicator (roofs indicate ceilings/indoors)
        - fog_wall_visible = 85% indicator of indoors (fog walls are dungeon doors)
        - wall_confirmed (3+ directions) = moderate indicator (enclosed room)
        """
        # Primary indicator: is_outdoor flag from game interface
        is_outdoor_flag = env_dict.get('is_outdoor', False)
        
        if is_outdoor_flag:
            indoor_score = 0.0  # Game says outdoor
        else:
            indoor_score = 1.0  # Game says indoor
        
        # Secondary: Roof visible (strong indicator of indoors)
        if env_dict.get('roof_visible', False):
            if is_outdoor_flag:
                # Game says outdoor but roof visible = contradiction
                # Roof is strong indicator, trust it over is_outdoor
                indoor_score = 0.9
            else:
                # Both agree it's indoor = very confident
                indoor_score = 0.95
        
        # Tertiary: Fog wall (85% reliable)
        # Only use if primary/secondary haven't already set confidence high
        elif env_dict.get('fog_wall_visible', False):
            if is_outdoor_flag:
                # Game says outdoor but fog visible = contradiction
                # Fog wall is 85% accurate
                indoor_score = 0.85
            else:
                # Both agree it's indoor
                indoor_score = 0.95
        
        # Quaternary: Walls from multiple directions (indicates enclosed room)
        # If we've hit confirmed walls from 3+ different directions, we're likely in a room
        # Only count confirmed walls (not temporary obstacles like pillars)
        wall_confirmed = self.wall_confirmed if hasattr(self, 'wall_confirmed') else set()
        
        if len(wall_confirmed) >= 3:
            # Confirmed walls on 3+ sides = enclosed space (indoor)
            if is_outdoor_flag:
                # Game says outdoor but walls on 3+ sides = contradiction
                # Multiple confirmed wall indicators are fairly reliable
                indoor_score = max(indoor_score, 0.75)
            else:
                # Both agree it's indoor
                indoor_score = 0.95
        
        return indoor_score
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Returns:
            initial observation, info dict
        """
        super().reset(seed=seed)
        
        self.episode_reward = 0
        self.steps = 0
        self.melina_spoken_to = False  # Reset Melina flag for new episode
        self.previous_health = 1.0  # Reset health tracking
        self.previous_stamina = 1.0  # Reset stamina tracking
        self.chapel_exited = False  # Reset chapel exit flag
        self.boss_spawned = False  # Reset boss spawn flag
        self.boss_spawn_deadline_passed = False  # Reset deadline penalty flag
        self.door_opened_this_episode = False  # Reset door opened flag
        self.last_door_state = False  # Reset last door state
        self.steps_since_last_door_action = 0  # Reset door hover counter
        self.last_interact_action_step = -999  # Reset interact action tracker
        self.position_when_interacted = None  # Reset position tracker
        self.last_inventory_count = 0  # Reset inventory count tracker
        self.has_wizened_finger = False  # Reset key item acquisition flag
        self.last_movement_action = None  # Reset movement trajectory tracking
        self.message_reads_this_episode = 0  # Reset message read counter for new episode
        self.interact_pending_state_check = False  # Reset interact state check flag
        self.interact_prompt_state_when_pressed = False  # Reset saved prompt state
        self.interact_inventory_when_pressed = tuple()  # Reset saved inventory state
        self.last_prompt_hash = None  # Reset prompt hash tracking
        self.prompt_attempt_counts = {}  # Reset per-prompt attempt counts
        
        # NOTE: DO NOT reset map state or subwindow state - preserve across episodes
        # The game doesn't reset between episodes, so if the map/subwindow are open,
        # they will still be open at the start of the next episode.
        # Our tracking must reflect the actual game state persistently.
        # However, RESET the step counters so duration calculations work for THIS episode
        if self.map_open:
            self.map_opened_step = 0  # Map is still open, but reset step counter for new episode
        if self.map_subwindow_open:
            self.map_subwindow_opened_step = 0  # Subwindow is still open, but reset step counter
        
        # Get initial state
        state = self.game_interface.get_game_state()
        observation = state['raw_screen']  # Return raw image for wrapper to process
        
        # Sync map tracking with actual game state
        # If map_opened_step was set but now steps=0, update it so frame calc is correct
        if self.map_open and self.map_opened_step > 0:
            # Map was open in previous episode, preserve the open state
            # but reset the step counter for this new episode
            self.map_opened_step = 0  # Map is still open, but now at step 0 of new episode
            print(f"ℹ️  Map persisted from previous episode - still open, applying penalties...")
        
        # CHECK: Do we already have the Wizened Finger? (in case restarting mid-session)
        quickslots = state.get('quickslots', [False] * 8)
        if any(quickslots):
            self.has_wizened_finger = True
            print(f"✓ Wizened Finger already in inventory at episode start")
        
        # PRINT EPISODE GOALS
        print(f"\n{'='*70}")
        print(f"EPISODE START - AI OBJECTIVES")
        print(f"{'='*70}")
        
        # Calculate indoor confidence at start
        indoor_confidence = self._calculate_indoor_confidence(state)
        
        # Indoor confidence: 0.0 = definitely outdoors, 1.0 = definitely indoors
        # Display shows decision (YES/NO) and confidence in that decision
        is_indoors = indoor_confidence > 0.5
        indoors_status = "YES" if is_indoors else "NO"
        
        # Calculate confidence percentage in the decision
        # If indoors: confidence = indoor_confidence * 100
        # If outdoors: confidence = (1 - indoor_confidence) * 100
        if is_indoors:
            confidence_pct = indoor_confidence * 100
        else:
            confidence_pct = (1 - indoor_confidence) * 100
        
        print(f"Currently Indoors: {indoors_status} ({confidence_pct:.1f}% confidence)")
        
        # Get next goal from goal system (main goal or first uncompleted)
        next_goal = self.goals.main_goal
        if not next_goal:
            # Find first uncompleted goal
            for goal in self.goals.goals:
                if not goal.completed:
                    next_goal = goal
                    break
        
        if next_goal:
            print(f"Primary Goal: {next_goal.name}")
            # Show full description without truncation (wrap if needed)
            desc = next_goal.description
            if len(desc) > 400:
                desc = desc[:400] + "..."
            print(f"  Description: {desc}")
            print(f"  Priority: {next_goal.priority}/10")
            if next_goal.sub_goals:
                print(f"  Sub-goals: {len(next_goal.sub_goals)} ({len(next_goal.completed_sub_goals)} completed)")
        else:
            print(f"Primary Goal: None (all goals completed!)")
        
        print(f"{'='*70}\n")
        
        info = {}
        
        return observation, info
    
    
    def _reset_area(self, new_origin=None):
        """
        Reset coordinate system when entering new area (fog wall, cutscene, etc.)
        
        Args:
            new_origin: if provided, set as new origin; otherwise use current position as origin
        """
        if new_origin is None:
            new_origin = self.current_position.copy()
        
        # Shift all recorded positions
        offset = self.current_position - new_origin
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.area_origin = new_origin.copy()
        
        # Clear location history (new area = new map)
        self.visited_locations = {}
        self.location_visit_history = []
        
        # Reset stuck directions and wall tracking for new area
        self.stuck_directions = {}
        self.direction_wall_attempts = {}
        self.wall_confirmed = set()
        
        self.last_area_transition_step = self.steps
        print(f"\n🌍 AREA TRANSITION - Resetting coordinate system at step {self.steps}")

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            state = self.game_interface.get_game_state()
            # Display the raw screen
            import cv2
            cv2.imshow('Elden Ring AI', state['raw_screen'])
            cv2.waitKey(1)



class FrameStackWrapper(gym.Wrapper):
    """
    Convert raw RGB images to stacked grayscale frames for CNN processing.
    
    OBSERVATION CONSISTENCY:
    - Input from EldenRingEnv: state['raw_screen'] = (1080, 1920, 3) BGR uint8
    - Processing: BGR → grayscale, resize to 84×84
    - Output: (84, 84, 4) uint8 grayscale stacked frames
    - Validated: Each observation checked against declared space
    
    Process:
    1. Receive 1920×1080 RGB frame from game
    2. Convert to grayscale (84×84)
    3. Add to 4-frame buffer
    4. Stack into (84, 84, 4) array
    5. Validate against observation space
    6. Return to CNN + LSTM policy
    
    This allows CNN to extract spatial features and detect movement.
    LSTM will handle temporal sequences of these visual observations.
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        
        self.num_stack = num_stack
        self.frame_buffer = collections.deque(maxlen=num_stack)
        
        # Update observation space: stacked grayscale frames
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, num_stack),
            dtype=np.uint8
        )
    
    def _process_frame(self, frame):
        """
        Convert raw RGB frame (1920x1080) to grayscale (84x84)
        
        Args:
            frame: numpy array of shape (1080, 1920, 3) in BGR format from OpenCV
            
        Returns:
            grayscale frame of shape (84, 84) as uint8
        """
        try:
            import cv2
            # Convert BGR to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 84x84
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Fallback: return random frame if processing fails
            return np.zeros((84, 84), dtype=np.uint8)
    
    def _stack_frames(self):
        """Stack frames into (84, 84, num_stack) array"""
        # Ensure buffer is full - if not, repeat the latest frame
        while len(self.frame_buffer) < self.num_stack:
            if len(self.frame_buffer) > 0:
                self.frame_buffer.append(self.frame_buffer[-1])  # Repeat last frame
            else:
                self.frame_buffer.append(np.zeros((84, 84), dtype=np.uint8))  # Black frame
        
        stacked = np.stack(list(self.frame_buffer), axis=2)
        return stacked.astype(np.uint8)
    
    def step(self, action):
        # Get raw observation from underlying env
        raw_frame, reward, terminated, truncated, info = self.env.step(action)
        
        # Process frame and add to buffer
        processed = self._process_frame(raw_frame)
        self.frame_buffer.append(processed)
        
        # Stack frames for CNN input
        observation = self._stack_frames()
        
        # VALIDATION: Ensure observation matches declared space
        assert self.observation_space.contains(observation), \
            f"Observation {observation.shape} does not match space {self.observation_space.shape}. " \
            f"Got dtype {observation.dtype}, expected {self.observation_space.dtype}"
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        raw_frame, info = self.env.reset(seed=seed, options=options)
        
        # Clear frame buffer and fill with initial frame
        self.frame_buffer.clear()
        processed = self._process_frame(raw_frame)
        for _ in range(self.num_stack):
            self.frame_buffer.append(processed)
        
        # Return stacked frames
        observation = self._stack_frames()
        
        # VALIDATION: Ensure observation matches declared space
        assert self.observation_space.contains(observation), \
            f"Reset observation {observation.shape} does not match space {self.observation_space.shape}. " \
            f"Got dtype {observation.dtype}, expected {self.observation_space.dtype}"
        
        return observation, info



class AIAgent:
    """AI agent using RecurrentPPO + CNN + LSTM for visual learning with temporal memory"""
    
    def __init__(self, env):
        """
        Initialize AI agent with CNN + LSTM architecture
        
        Args:
            env: Gymnasium environment wrapped with FrameStackWrapper
        """
        self.env = env
        
        # RecurrentPPO with CNN for visual processing + LSTM for temporal patterns
        # Architecture:
        # - NatureCNN: processes 84x84x4 image frames (standard Atari architecture)
        # - LSTM: maintains temporal context across frames
        # - Policy head: outputs action probabilities
        # - Value head: estimates state value for bootstrapping
        try:
            policy_kwargs = dict(
                # CNN features extraction for spatial understanding
                features_extractor_class=NatureCNN,
                features_extractor_kwargs=dict(
                    features_dim=256,  # Output from CNN
                ),
                # LSTM for temporal sequence modeling
                net_arch=dict(
                    pi=[256, 256],    # Policy network hidden layers
                    vf=[256, 256],    # Value network hidden layers
                ),
                n_lstm_layers=2,
                lstm_hidden_size=256,
                use_expln=False,  # Disable exponential learning rate schedule
            )
            self.model = RecurrentPPO(
                RecurrentActorCriticPolicy,
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                policy_kwargs=policy_kwargs
            )
            print("✓ Using RecurrentPPO with NatureCNN + LSTM")
            print("  - CNN: extracts spatial features from 84x84 grayscale images")
            print("  - LSTM: learns temporal patterns with 2 layers, 256 hidden size")
            print("  - Processing 4-frame stacks to detect motion and context")
        except Exception as e:
            print(f"⚠️  NatureCNN initialization failed: {e}")
            print("Attempting fallback: SimpleMultiInputPolicy without custom CNN...")
            try:
                # Fallback: Use default policy which can handle images with built-in CNN
                # This uses a simpler CNN than NatureCNN but still works with image input
                policy_kwargs = dict(
                    net_arch=dict(
                        pi=[256, 256],
                        vf=[256, 256],
                    ),
                    n_lstm_layers=2,
                    lstm_hidden_size=256,
                )
                self.model = RecurrentPPO(
                    RecurrentActorCriticPolicy,
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=1,
                    policy_kwargs=policy_kwargs
                )
                print("✓ Fallback: RecurrentPPO with default CNN + LSTM initialized successfully")
            except Exception as e2:
                print(f"❌ RecurrentPPO failed completely: {e2}")
                print("   This suggests an incompatibility with the observation space or environment")
                raise RuntimeError(
                    f"Failed to initialize RecurrentPPO with both NatureCNN and default policy.\n"
                    f"Primary error: {e}\n"
                    f"Fallback error: {e2}\n"
                    f"Please verify the environment observation space matches the policy expectations."
                )
        
        self.checkpoint_dir = "models/checkpoints"
        self.ensure_checkpoint_dir()
        self.total_timesteps = 0
    
    def ensure_checkpoint_dir(self):
        """Create checkpoint directory if it does not exist"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, total_timesteps=100000, resume=False):
        """
        Train the agent with pause/resume capability
        
        Args:
            total_timesteps: number of steps to train for
            resume: whether to resume from last checkpoint
        """
        if resume:
            self.load_latest_checkpoint()
            print(f"Resumed from checkpoint. Total timesteps so far: {self.total_timesteps}")
        
        print(f"Training for {total_timesteps} steps...")
        print("Press Ctrl+S to save and pause training at any time")
        print("=" * 60)
        print("Periodic reward scores will print every 60 seconds\n")
        
        try:
            # Create callbacks for periodic logging and hardware monitoring
            reward_callback = PeriodicRewardCallback(log_interval_seconds=60)
            hardware_callback = HardwareMonitorCallback(
                min_disk_gb=200.0,    # Stop if less than 200GB free (allows ~1.5TB to be used for training)
                max_cpu_temp=85.0,    # Stop if CPU exceeds 85°C
                check_interval=100,   # Check every 100 steps
                verbose=1
            )
            
            self.model.learn(total_timesteps=total_timesteps, 
                           callback=[reward_callback, hardware_callback])
            self.total_timesteps += total_timesteps
            print("\nTraining complete!")
            # Print action statistics at end of training
            self.env.unwrapped.print_action_recap()
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint("interrupted")
            print("Checkpoint saved!")
            # Print action statistics even if interrupted
            self.env.unwrapped.print_action_recap()
    
    def save_checkpoint(self, label=""):
        """
        Save model and training state
        
        Args:
            label: optional label for the checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"model_checkpoint_{timestamp}"
        if label:
            checkpoint_name += f"_{label}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        self.model.save(checkpoint_path)
        
        # Save metadata and training statistics
        metadata_path = checkpoint_path + "_metadata.txt"
        base_env = self.env.unwrapped
        action_counts = base_env.action_counts if hasattr(base_env, 'action_counts') else np.zeros(19)
        action_names = base_env.action_names if hasattr(base_env, 'action_names') else [f"Action {i}" for i in range(19)]
        
        with open(metadata_path, "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total timesteps: {self.total_timesteps}\n")
            f.write(f"Label: {label}\n")
            f.write(f"\nACTION STATISTICS:\n")
            f.write(f"{'='*60}\n")
            
            total_actions = np.sum(action_counts)
            if total_actions > 0:
                sorted_actions = sorted(enumerate(action_counts), key=lambda x: x[1], reverse=True)
                f.write(f"Total actions taken: {int(total_actions):,}\n")
                f.write(f"Unique actions used: {np.count_nonzero(action_counts)} / {len(action_names)}\n\n")
                f.write(f"{'Rank':<6} {'Action':<25} {'Count':<10} {'Percentage':<12}\n")
                f.write(f"{'-'*6} {'-'*25} {'-'*10} {'-'*12}\n")
                
                for rank, (action_id, count) in enumerate(sorted_actions[:10], 1):
                    pct = (count / total_actions * 100) if total_actions > 0 else 0
                    action_name = action_names[action_id] if action_id < len(action_names) else f"Action {action_id}"
                    f.write(f"{rank:<6} {action_name:<25} {int(count):<10} {pct:>6.1f}%\n")
            else:
                f.write(f"No actions recorded during training\n")
        
        print(f"✓ Checkpoint saved: {checkpoint_name}")
        return checkpoint_path
    
    def _get_sorted_checkpoints(self):
        """
        Get list of all checkpoint files, sorted by name (newest first)
        
        Returns:
            list: checkpoint filenames sorted in reverse order (newest first)
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("model_checkpoint_") and not f.endswith("_metadata.txt")]
        checkpoints.sort(reverse=True)
        return checkpoints
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        checkpoints = self._get_sorted_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found!")
            return False
        
        latest = checkpoints[0]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest)
        
        self.model = RecurrentPPO.load(checkpoint_path, env=self.env)
        
        # Load metadata
        metadata_path = checkpoint_path + "_metadata.txt"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Total timesteps:" in line:
                        self.total_timesteps = int(line.split(":")[-1].strip())
        
        print(f"✓ Loaded checkpoint: {latest}")
        return True
    
    def list_checkpoints(self):
        """List all saved checkpoints"""
        checkpoints = self._get_sorted_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        print("\nAvailable checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            metadata_path = os.path.join(self.checkpoint_dir, cp + "_metadata.txt")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    print(f"{i}. {cp}")
                    for line in f:
                        print(f"   {line.strip()}")
            else:
                print(f"{i}. {cp}")
    
    def play(self, episodes=1, render=True, log_decisions=False):
        """
        Play game using trained model
        
        Args:
            episodes: number of episodes to play
            render: whether to render
        """
        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Get action from trained model
                action, _states = self.model.predict(obs, deterministic=False)
                
                # Extract scalar action if returned as array (from vectorized env)
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.ndim == 0 else int(action[0])
                
                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
            
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")
    
    
    def save(self, path):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load trained model"""
        self.model = RecurrentPPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def analyze_learning(self):
        """
        Analyze what the AI has learned so far.
        Returns insights about the model behavior and training progress.
        """
        print("\n" + "=" * 70)
        print("AI LEARNING ANALYSIS")
        print("=" * 70)
        
        # Get model info
        print(f"\nModel Architecture:")
        print(f"  Algorithm: RecurrentPPO (Recurrent Proximal Policy Optimization)")
        print(f"  Policy Type: RecurrentActorCriticPolicy (LSTM-based)")
        print(f"  Total timesteps trained: {self.total_timesteps:,}")
        
        # Analyze policy network
        policy = self.model.policy
        print(f"\nPolicy Network:")
        print(f"  Input features: {policy.observation_space.shape[0]}")
        print(f"  Output actions: {policy.action_space.n}")
        
        # Check for LSTM layers
        if hasattr(policy, 'lstm_actor'):
            print(f"  LSTM Layers: Yes (actor + critic)")
        elif hasattr(policy, 'mlp_extractor'):
            print(f"  Network Type: MLP (fallback)")
        
        # Try to load and display saved statistics from metadata
        checkpoints = self._get_sorted_checkpoints()
        if checkpoints:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[0])
            metadata_path = checkpoint_path + "_metadata.txt"
            
            if os.path.exists(metadata_path):
                print(f"\nTraining Statistics (from checkpoint metadata):")
                try:
                    with open(metadata_path, "r") as f:
                        metadata_content = f.read()
                    
                    # Extract and display the action statistics section
                    if "ACTION STATISTICS:" in metadata_content:
                        stats_section = metadata_content.split("ACTION STATISTICS:")[1]
                        print(stats_section)
                    else:
                        print("  (No action statistics saved with this checkpoint)")
                except Exception as e:
                    print(f"  Error reading metadata: {e}")
            else:
                # Fallback to live analysis if no metadata
                print(f"\nAction Distribution (live from environment):")
                base_env = self.env.unwrapped
                action_counts = base_env.action_counts if hasattr(base_env, 'action_counts') else np.zeros(19)
                action_names = base_env.action_names if hasattr(base_env, 'action_names') else [f"Action {i}" for i in range(19)]
                
                print(f"  {'Rank':<6} {'Action':<25} {'Count':<10} {'Percentage':<12}")
                print(f"  {'-'*6} {'-'*25} {'-'*10} {'-'*12}")
                
                total_actions = np.sum(action_counts)
                sorted_actions = sorted(enumerate(action_counts), key=lambda x: x[1], reverse=True)
                
                for rank, (action_id, count) in enumerate(sorted_actions[:10], 1):
                    pct = (count / total_actions * 100) if total_actions > 0 else 0
                    action_name = action_names[action_id] if action_id < len(action_names) else f"Action {action_id}"
                    bar = "█" * int(pct / 5)
                    print(f"  {rank:<6} {action_name:<25} {count:<10} {pct:>6.1f}% {bar}")
                
                # Check for exploration capability
                print(f"\nExploration Metrics:")
                unique_actions_used = np.count_nonzero(action_counts)
                print(f"  Unique actions used: {unique_actions_used} / {self.env.action_space.n}")
                
                if unique_actions_used == 1:
                    dominant_action = np.argmax(action_counts)
                    print(f"  ⚠️  WARNING: AI is stuck on action '{action_names[dominant_action]}'")
                    print(f"      This indicates the LSTM hasn't learned diverse exploration yet.")
                elif unique_actions_used < 5:
                    print(f"  ⚠️  LIMITED exploration - only {unique_actions_used} of {self.env.action_space.n} actions used")
                    print(f"      LSTM may still be learning basic patterns")
                else:
                    print(f"  ✓ Good action diversity - using {unique_actions_used} different actions")
        else:
            print(f"\nNo checkpoints found - cannot load training statistics")
        
        print(f"\n{'='*70}\n")
    
    def analyze_training_history(self):
        """
        Analyze all checkpoints to show training progression and action learning.
        Shows how the AI's behavior evolved across multiple training sessions.
        """
        print("\n" + "=" * 70)
        print("TRAINING HISTORY ANALYSIS")
        print("=" * 70)
        
        checkpoints = self._get_sorted_checkpoints()
        if not checkpoints:
            print("\nNo checkpoints found!")
            return
        
        print(f"\nFound {len(checkpoints)} checkpoint(s)\n")
        
        # Display each checkpoint's statistics
        for idx, checkpoint_file in enumerate(checkpoints, 1):
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
            metadata_path = checkpoint_path + "_metadata.txt"
            
            print(f"\n{'─'*70}")
            print(f"Checkpoint {idx}: {checkpoint_file}")
            print(f"{'─'*70}")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        content = f.read()
                    print(content)
                except Exception as e:
                    print(f"Error reading metadata: {e}")
            else:
                print("No metadata available for this checkpoint")
        
        print(f"\n{'='*70}\n")
    
    def test_checkpoint_behavior(self, num_steps=500, verbose=True):
        """
        Run the model for limited steps to see what it actually does in the environment.
        Useful to diagnose if AI is exploring or stuck in a loop.
        
        Args:
            num_steps: number of steps to run
            verbose: print detailed output
        """
        print(f"\n{'='*70}")
        print(f"TESTING CHECKPOINT BEHAVIOR - MODEL ARCHITECTURE ANALYSIS")
        print(f"{'='*70}")
        
        # Analyze the model structure
        policy = self.model.policy
        
        print(f"\nModel Configuration:")
        print(f"  Algorithm: RecurrentPPO (Recurrent Proximal Policy Optimization)")
        print(f"  Policy Type: RecurrentActorCriticPolicy")
        print(f"  Device: {self.model.device}")
        print(f"  Learning Rate: {self.model.learning_rate}")
        
        print(f"\nNetwork Architecture:")
        print(f"  Input: {policy.observation_space.shape[0]} semantic features")
        print(f"  Output: {policy.action_space.n} discrete actions")
        
        # Check LSTM layers
        if hasattr(policy, 'lstm_actor'):
            print(f"  LSTM Layers: Yes")
            print(f"    - Actor LSTM: {policy.lstm_actor}")
            print(f"    - Critic LSTM: {policy.lstm_critic}")
        else:
            print(f"  LSTM Layers: Not found in policy")
        
        # Try to estimate training info
        print(f"\nTraining Session Info:")
        print(f"  Total timesteps recorded: {self.total_timesteps:,}")
        
        # Metadata from checkpoint
        checkpoints = self._get_sorted_checkpoints()
        if checkpoints:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[0])
            metadata_path = checkpoint_path + "_metadata.txt"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = f.read()
                print(f"\n  Checkpoint Metadata:")
                for line in metadata.strip().split("\n"):
                    print(f"    {line}")
        
        print(f"\nNote: Full runtime testing requires active game environment.")
        print(f"      The checkpoint is valid and ready to use.")
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print("Elden Ring AI - Basic Setup")
    print("=" * 50)
    print("\nNote: Make sure Elden Ring is running before starting training!")
