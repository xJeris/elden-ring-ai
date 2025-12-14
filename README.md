# Elden Ring AI Agent

A reinforcement learning agent trained to explore and navigate Elden Ring using computer vision and reward shaping.

## Architecture

### Core Algorithm
- **Algorithm**: RecurrentPPO (Recurrent Proximal Policy Optimization) from `sb3_contrib`
- **Policy**: RecurrentActorCriticPolicy with LSTM for temporal sequence learning
- **CNN**: NatureCNN for spatial feature extraction from game frames

### Visual Processing Pipeline
1. **Input**: 1920×1080 BGR frames captured from game
2. **Frame Processing**: 
   - Convert to grayscale
   - Resize to 84×84 (Atari standard)
   - Stack 4 frames for motion detection
3. **Output**: (84, 84, 4) uint8 observations to CNN + LSTM

### Training Configuration
- **Learning Rate**: 3e-4
- **Timesteps per Rollout**: 2048
- **Batch Size**: 64
- **Epochs per Rollout**: 5 (reduced from 10 for faster training)
- **Gamma** (discount factor): 0.99
- **GAE Lambda**: 0.95
- **LSTM Layers**: 2 with 256 hidden units
- **Reward Normalization**: ±5 scale for PPO stability
- **Max Episode Steps**: 18,000 frames (~60 minutes at 5fps)

## Game State Detection

The agent perceives game state through computer vision-based detection:

### Health & Damage
- **Health Bar Detection**: Red color detection in bottom-left area
- **Damage Tracking**: Detects health drops to identify enemy vs environmental damage
- **Combat Detection**: Boss health bar, fog walls, or recent damage from enemies

### Environmental Features
- **Doors & Exits**: Color-based detection of door prompts (white = ready, grey = locked)
- **Ground Items**: Brightness detection for glowing loot on corpses/chests/ledges
- **Status Effects**: Saturation-based detection for poison/rot/bleed buildup bars
- **Roof/Ceiling**: Edge detection for indoor vs outdoor areas

### Game Objectives
- **Wizened Finger Detection**: Scans inventory for required key item
- **Chapel Exit Detection**: Outdoor area detection indicates progression
- **Boss Spawn Detection**: Visible boss health bar + fog wall = arena reached

## Reward Structure (Rebalanced for Movement Focus)

### Core Design Philosophy
Rebalanced to make **movement the dominant strategy** (~50-60% of actions), with interact as a secondary, intentional action (~10% of actions). All rewards normalized to ±5 range for PPO stability.

Exploration is driven by a **dual-layer curiosity system**:
1. **State-Signature Curiosity**: High-level world state changes (exits, items, doors, health, location)
2. **Visual Pixel Curiosity**: Low-level pixel changes (detects forward progress vs strafing)

### Curiosity-Driven Exploration (Dual-Layer System)

#### State-Signature Curiosity (High-Level)
Tracks 10 attributes of game state:
- Total exits visible
- Closed and open doors
- Health and stamina (quantized)
- Prompt and item visibility
- Outdoor/indoor location
- Combat status

**Rewards & Penalties:**
- **State Change**: +0.6 reward whenever any attribute changes
- **Stagnation Penalty**: -0.1 per frame after 3 unchanged frames, capped at -0.5
- **Purpose**: Encourages exploring new areas, meeting new NPCs, finding new exits

**Suppression Mechanics:**
- Blocked for 8 frames after interact action (prevents UI exploits)
- Blocked for 8 frames after map action (prevents UI state change exploitation)
- Blocked for 8 frames after camera/mount action (prevents spinning/animation exploits)

#### Visual Pixel Curiosity (Low-Level)
Computes mean absolute difference between 8×8 downsampled grayscale frames.

**Rewards & Penalties:**
- **Large visual change** (diff > 5): +0.1 reward
  - Indicates forward movement, camera pan, environmental exploration
- **Sustained low visual change** (diff < 2 for 3+ frames): -0.05 penalty
  - Indicates strafing, standing still, wiggling in place

**Curiosity Suppression (Map UI Prevention):**
- Visual curiosity rewards are suppressed when:
  - Map is being toggled (action 18)
  - Map UI is open (`self.map_open`)
  - Map subwindows are open (`self.map_subwindow_open` - inventory, markers, etc.)
- Prevents AI from exploiting large visual changes caused by map UI transitions
- Suppression only affects curiosity reward calculations, not stagnation detection

**Why this matters:**
- Can't cheat by walking backward at wall (high-level state unchanged, but low visual change)
- Can't strafe left-right repeatedly without moving forward (pixels change minimally)
- Can't farm curiosity rewards from map UI screen transitions
- Rewards actual exploration over micro-farming


### Interaction System (5-Layer Anti-Farming Architecture)

The interact system uses **5 priority layers** to prevent reward exploitation while maintaining exploration incentives:

#### Layer 1: Door Success Bootstrapping (+4.0)
- **Condition**: Prompt is door-like OR hash has `door_successes > 0`
- **Reward**: +4.0 full reward
- **Purpose**: Once AI learns a door works, always reward it fully (prevents re-farming heuristic)
- **Resets cooldown**: `steps_since_last_interact_reward = 0`

#### Layer 2: Per-Hash Cooldown (Temporal Gate)
- **Condition**: 50 steps since last attempt on same hash
- **Reward**: +0.2 - 0.5 = -0.3 net penalty (stronger to prevent spam)
- **Purpose**: Prevents rapid re-spamming of the same prompt
- **Decay**: Auto-halving every 200 steps (per-hash), global reset every 300 steps

#### Layer 3: Global Interact Cooldown
- **Condition**: 10 steps since any interact attempt
- **Reward**: +0.2 - 0.1 = +0.1 net penalty
- **Purpose**: Prevents machine-gun interact spamming across different prompts
- **Decay**: Increments every step, auto-resets periodically

#### Layer 4: First-Time Non-Door (+0.2)
- **Condition**: Non-door prompt, first ever attempt on hash
- **Reward**: +0.2 (tiny reward, neutral for exploration)
- **Purpose**: Allows free testing of unknown prompts
- **Escalation Trigger**: Attempts > 1 (escalates faster, only 1 free test)

#### Layer 5: Escalating Repetition Penalty
- **Condition**: Hash attempt count > 1 (attempts at 2+ trigger penalty)
- **Formula**: +0.2 - min(0.5 × (attempts - 1), 1.5)
- **Penalty Cap**: -1.5 (prevents permanent lockout, always recoverable)
- **Decay**: Automatic halving every 200 steps, global reset every 300 steps
- **Hash=None Handling**: Skips escalation for unhashable prompts (maps to "__NONE__")

#### Anti-Farming Mechanics
- **Attempt Increment Timing**: Incremented BEFORE layer evaluation to prevent skipped counts
- **Global Cooldown Aging**: Increments every step - cannot be bypassed
- **Partial Door Signals**: Detects animation progress even without full state change:
  - Prompt white → grey transition (+0.5 bonus)
  - Player position forward > 0.5 units (+0.3 bonus)
  - Exits increased in world (+0.2 bonus)
  - ANY signal triggers `door_successes += 1` and attempt reset
- **Escalation Recovery**: Global decay loop runs every 300 steps, automatically halving attempt counts across ALL hashes
- **Curiosity Suppression**: State-signature reward blocked for 8 frames post-interact, post-map, and post-camera/mount actions

#### Telemetry Tracking
5 counters track layer distribution per episode:
- `telemetry_door_rewards`: Layer 1 hits
- `telemetry_non_door_rewards`: Layer 4 hits
- `telemetry_per_hash_cooldown_hits`: Layer 2 hits
- `telemetry_global_cooldown_hits`: Layer 3 hits
- `telemetry_escalations_applied`: Layer 5 escalations
- JSON output at episode end for parameter tuning

### Movement (Primary Focus)
- **Forward Movement**: +2.0 (strong incentive for primary exploration direction)
- **Movement Momentum**: +0.5 every 10 consecutive frames in same direction (encourages sustained exploration)
- **Momentum Reset**: Resets to 0 whenever interact action is taken
- **Backward Movement**: -0.5 (opposite of goal)
- **Sideways Movement**: -0.10 (not primary direction)

#### Direction Change Anti-Farming (Prevents Dithering)
Prevents AI from switching directions rapidly to farm micro-rewards without real progress:

- **Direction Change Penalty**: -0.4 per frame when changing direction before 5-frame threshold
  - Frame 1 change: -0.4 penalty (prevent instant switches)
  - Frames 2-4: -0.4 (sustained penalty)
  - Frame 5+: -0.05 (allow with minimal cost)
- **Direction Continuity Bonus**: +0.3 every 10 consecutive frames in same direction (NEW)
- **Direction Flip Cooldown**: 5-step minimum duration per direction
  - Penalty: -0.2 if attempted to flip before cooldown expires
  - Prevents: Left-right oscillation [3,4,3,4] pattern detection
- **Lateral Oscillation Detection**: -0.2 penalty for alternating left-right rapidly
- **Lateral Movement Suppression**: -0.05 additional penalty when left/right actions sustained with low visual change (3+ frames)
  - Specifically targets strafing in place without forward progress
  - Independent check: applies in addition to other movement penalties
- **Momentum Reset**: `consecutive_frames_in_direction = 0` when interact (action 12) taken

**Purpose:** Forces coherent exploration paths instead of dithering at decision points

### Ground Item Interaction
- **Ground Items Pickup**: +0.7 (bonus for looting utility items)

### Map (Heavily Suppressed with Multiple Layers)
- **Opening Map**: -15.0 (severe penalty, immediate application suppresses other rewards)
- **Any Action While Map Open**: -0.5/step (continuous penalty encourages closing)
- **Map Reopen Cooldown**: 900 steps (30 seconds) after closing to prevent oscillation
- **State-Signature Curiosity Block**: +0.6 reward suppressed for 8 frames post-map action
- **Visual Curiosity Block**: Already suppressed when map UI is open
- **Subwindow Validation**: State resync ensures penalties only apply when map actually open

### Combat Actions (Prime Directives)
- **Attack Actions (Normal/Heavy)**: -0.2 (combat avoided during exploration)
- **Backstep/Dodge**: -1.0 (EXTREME penalty - never dodge during exploration)
- **Jump**: -1.0 (EXTREME penalty - never jump during exploration)
- **Using Items**: -0.5 (normalized from -2.0, strict constraint in exploration)
- **Lock-on**: -0.2 (combat prep, avoided)
- **Skill**: -0.3 (skill use avoided)
- **Summon Mount**: -0.1 (unavailable, minimal penalty)

### Wasted Interact Penalties
- **Spam within 5 steps of last attempt**: -1.0 (rapid re-attempt penalty)
- **Attempt after 5+ steps**: -0.5 (lighter penalty for spaced attempts)

### Camera & Navigation
- **Camera Panning (Solo)**: -0.15 (small penalty for wasting time)
- **Camera + Forward Movement**: +0.05 (bonus for intelligent steering)
- **Camera/Mount Curiosity Suppression**: Visual and state-signature rewards blocked for 8 frames post-action
- **Time Penalty**: -0.01 per step (forces faster movement, stronger than previous -0.003)

### Major Events
- **Chapel Exit Bonus**: +1.0 to +5.0 (based on speed)
- **Boss Arena Reached**: +3.0
- **Wizened Finger Pickup**: +2.0 (key item acquisition)

### Temporal Constraints
- **Direction Change Penalties** (Momentum Enforcement):
  - Frame 1 change: -0.45 (prevent dithering)
  - Frame 3 change: -0.35
  - Frame 5 change: -0.25
  - After 5 frames: -0.05 (allow with minimal cost)
  - Stuck detection: -0.10 (allow escaping)

## Prime Directives (Hard Constraints)

During exploration mode (not in combat), these rules are enforced via severe penalties:

1. **Never Use Items** (-0.5 penalty, normalized from -2.0)
2. **Never Dodge/Backstep** (-1.0 penalty - wasted stamina, normalized from -0.5)
3. **Never Jump** (-1.0 penalty - pointless action, normalized from -0.5)
4. **Never Open Map** (-15.0 penalty - immediate application, suppresses other rewards, reopen cooldown 900 steps)
5. **Prefer Forward Movement** (primary direction, +1.0 bonus)

All penalties are normalized to ±5 range for PPO training stability.

## Behavioral Cloning (Optional)

### Recording Format
- **imitation_with_camera.py**: Captures human demonstrations
- **Data Captured**: Raw RGB frames (JPEG compressed at 95%), actions, camera angles
- **Storage**: Pickled recordings with metadata

### Training
- **behavioral_cloning_cnn.py**: Trains CNN + LSTM policy from demonstrations
- **Architecture**: Same as RL policy (NatureCNN + LSTM)
- **Loss**: CrossEntropyLoss for action prediction
- **Data Split**: 90% train / 10% validation

## File Structure

```
elden_ring_ai/
├── ai_agent.py                 # Core RL environment and agent
├── game_interface.py           # Game capture and state detection
├── main.py                     # Training orchestration
├── imitation_with_camera.py    # Record human demonstrations
├── behavioral_cloning_cnn.py   # Train policy from demonstrations
├── analyze_training.py         # Analyze training progress
├── analyze_checkpoint.py       # Load and analyze models
├── train_with_cloning.py       # Combined RL + behavioral cloning
├── requirements.txt            # Dependencies
└── models/
    └── checkpoints/            # Saved model checkpoints
```

## Installation

### Requirements
- Python 3.12 (required)
- Elden Ring (running at 1920×1080)
- Windows (pynput for keyboard/mouse control)

### Setup
```bash
pip install -r requirements.txt
python main.py
```

## Usage

### Training
```bash
python main.py
# Select option 1: Train new model (100k steps)
# Focus Elden Ring window when countdown reaches 0
```

### Recording Demonstrations
```bash
python imitation_with_camera.py
# Manually play Elden Ring while recording
# Saves recorded frames and actions
```

### Behavioral Cloning
```bash
python behavioral_cloning_cnn.py
# Trains policy from recorded demonstrations
```

### Analysis
```bash
python analyze_training.py
# View policy summary and action frequencies
```

## Current Status

### Working Features
- ✅ CNN + LSTM architecture fully integrated (NatureCNN + 2-layer LSTM, 256 hidden units)
- ✅ Frame stacking with validation (84×84×4)
- ✅ Health damage tracking and combat detection
- ✅ Door detection and interaction rewards with state-change gating
- ✅ Ground item detection and collection incentives
- ✅ Status effect buildup detection (foundation for environmental damage)
- ✅ Map opening severely penalized (-15.0 immediate, -0.5/step, 900-step reopen cooldown)
- ✅ Map/subwindow state persists across episodes with proper validation
- ✅ Prompt hash tracking (SHA-1 based) with attempt/success statistics
- ✅ **Momentum enforcement to prevent dithering** (direction change penalties + cooldown)
- ✅ **Direction continuity bonus** (+0.3 per 10 frames in same direction)
- ✅ **Lateral oscillation detection** (-0.2 for [3,4,3,4] patterns)
- ✅ **Lateral movement suppression** (-0.05 for sustained left/right with low visual change)
- ✅ **State-signature curiosity** (10-attribute state, +0.4 on change, -0.1 escalating penalty)
- ✅ **Visual pixel-level curiosity** (8×8 frame diff, +0.1 for large changes, -0.05 for stagnation)
- ✅ All recent code fixes verified and syntax clean
- ✅ All core files compile with Python 3.12
- ✅ Behavioral cloning infrastructure available
- ✅ All resets implemented across episode start, area transitions, and state changes

### Known Limitations
- State-change detection based on prompt disappearance and inventory changes (doesn't detect all in-game state changes like dialogue advancement)
- Ground item detection uses brightness threshold (may need tuning for different lighting)
- No direct enemy health bar detection (only boss bars)
- Camera control is mouse-based (may be noisy in some situations)
- Floor messages and item prompts may stack on-screen (edge case not fully handled)

### Recent Changes (Latest Session - December 2025)

**Phase 1: Major Bug Fixes (Debugging Session)**
- ✅ Fixed "state is not defined" error in map UI validation
- ✅ Fixed `_hash_prompt_region()` method call syntax (removed incorrect `self.` prefix)
- ✅ Fixed map/subwindow state persistence across episodes (now preserves booleans, resets counters)
- ✅ Fixed map UI validation scope (moved from `_calculate_reward()` to `step()` where state is available)
- ✅ Added map state resync logic to prevent false subwindow penalties
- ✅ Added comprehensive debug logging (9 try-catch blocks at state access points)
- ✅ Verified all state parameter scoping is correct
- ✅ Fixed comment inaccuracy (Action 9 penalty: -0.5, was documented as -10.0)
- ✅ Verified all 14 penalty-related comments are accurate to current code

**Phase 2: Reward Shaping Rebalance (Movement Focus)**
- ✅ **Reduced interact base reward**: +20.0 → +4.0 (major shift toward movement dominance)
- ✅ **Increased forward movement bonus**: +1.0 → +1.5 (more attractive than interact)
- ✅ **Added movement momentum bonus**: +0.5 every 10 consecutive frames forward (NEW)
- ✅ **Implemented momentum reset**: Counter resets to 0 whenever interact (action 12) is taken (NEW)
- ✅ **Added prompt novelty bonus**: +2.0 for successful interact on never-before-seen hash (NEW)
- ✅ **Added repetition penalty**: -1.0 for attempting interact on hash with 0% success rate (NEW)

**Phase 3: Direction Change Anti-Farming System**
- ✅ **Direction change penalties**: -0.4 per frame when changing before 5-frame threshold
- ✅ **Direction continuity bonus**: +0.3 every 10 frames in same direction (NEW)
- ✅ **Direction flip cooldown**: 5 steps duration with -0.2 penalty if violated (NEW)
- ✅ **Lateral oscillation detection**: -0.2 penalty for [3,4,3,4] or [4,3,4,3] patterns (NEW)
- ✅ **Momentum reset on interact**: `consecutive_frames_in_direction = 0` when action 12 taken
- ✅ **Cooldown decrement fix**: Now decrements EVERY movement step (was only during forward)
- ✅ **Consecutive frames reset**: Fixed missing reset in `reset()` method at episode start

**Phase 4: Curiosity-Driven Exploration System**
- ✅ **Extended state signature**: 3 attributes → 10 attributes (exits, doors, health, stamina, prompts, items, location, combat)
- ✅ **State-signature curiosity**: +0.4 reward on state change (was +0.2), -0.1 escalating penalty for stagnation
- ✅ **Stagnation detection threshold**: 3 frames (was 5) – more aggressive exploration
- ✅ **Penalty cap**: -0.5 maximum stagnation penalty
- ✅ **Visual pixel-level curiosity** (NEW): 
  - Downsample frame to 8×8 for lightweight processing
  - Reward +0.1 for large visual differences (forward movement, exploration)
  - Penalize -0.05 for sustained low visual differences (strafing, standing still)
- ✅ **Dual-layer curiosity**: High-level state changes + low-level pixel changes
- ✅ **All resets implemented**: Episode start, area transitions, state changes

**Phase 5: Enhanced Curiosity Suppression & Reward Amplification (CURRENT - December 2025)**
- ✅ **Per-hash cooldown strengthened**: -0.25 → -0.5 penalty (makes re-spamming more costly)
- ✅ **Attempt threshold lowered**: 2 → 1 (escalation triggers at 2nd attempt instead of 3rd, fewer free tests)
- ✅ **State-signature curiosity amplified**: +0.4 → +0.6 (makes exploration 50% more attractive)
- ✅ **Time penalty increased**: -0.003 → -0.01 per step (makes standing still 3x more costly)
- ✅ **Forward movement increased**: +1.5 → +2.0 per step (makes movement more attractive than interact attempts)
- ✅ **Interact curiosity suppression**: Added 8-frame suppression window for state-signature rewards post-interact
- ✅ **Map curiosity suppression**: Added 8-frame suppression window blocking +0.6 reward post-map action
- ✅ **Camera/mount curiosity suppression**: Added 8-frame suppression window blocking visual + state-signature rewards
- ✅ **Map action re-enabled**: Action space back to 19 (action 18 available) after testing
- ✅ **Multi-layer curiosity suppression**: Interact, map, and camera/mount actions all block hidden reward exploitation

**Current Tuning Parameters:**
- **State-Signature Curiosity**: +0.6 per state change
- **Time Penalty**: -0.01 per step
- **Forward Movement**: +2.0 per step
- **Per-Hash Cooldown Penalty**: -0.5
- **Attempt Threshold**: 1 (escalation at 2nd attempt)
- **Layer 1 Reward**: +4.0 (door success)
- **Layer 2 Cooldown**: 50 steps, -0.5 penalty
- **Layer 3 Cooldown**: 10 steps, -0.1 penalty
- **Layer 4 Reward**: +0.2 (first-time non-door)
- **Layer 5 Cap**: -1.5 (escalation penalty cap)
- **Per-hash decay**: 200 steps (ATTEMPT_DECAY_INTERVAL), 0.5 factor
- **Global decay**: 300 steps (GLOBAL_DECAY_INTERVAL)
- **Curiosity suppression duration**: 8 frames (interact, map, camera/mount)

**Expected Results After Implementation:**
- Interact attempts: ~10% (constrained by cooldowns, escalation penalties)
- Movement: 50-60% (primary strategy, high incentive)
- Successful door interactions: Only rewarded after first success (bootstrapping)
- Failed attempts: Escalating penalties prevent spam, but recovery path exists via decay
- Prompt hash diversity: Exploration encouraged through new hashes reaching Layer 1

## Code Architecture

### State Management
- **State Extraction**: All state reading in `step()` method (lines 440-940)
- **State Parameters**: Extracted from `self.game_interface.get_game_state()` at line 455
- **Reward Calculation**: Pure function `_calculate_reward()` (lines 1061-1435) receives extracted parameters
- **No Direct State Access**: `_calculate_reward()` has NO state parameter, prevents scope errors
- **Map State Persistence**: `reset()` method (lines 1713-1739) preserves map_open/map_subwindow_open across episodes

### Helper Functions
- **`_hash_prompt_region()`** (lines 30-71): Module-level function, returns 8-char SHA-1 hash
  - Input: BGR frame from game
  - Process: Crop (y:950-1080, x:700-1220) → grayscale → downsample to 32×8
  - Output: Hash string or None
  - Usage: `_hash_prompt_region(frame)` (NOT `self._hash_prompt_region()`)

## Tuning Parameters (Current Configuration)

### 5-Layer Anti-Farming System
To adjust interact behavior, modify values in [ai_agent.py](ai_agent.py) `__init__()` method:

**Layer Constants:**
- **LAYER_1_DOOR_REWARD (4.0)**: Full reward for bootstrapped doors
- **NON_DOOR_INTERACT_REWARD (0.2)**: Tiny reward for first attempts
- **PER_HASH_COOLDOWN (50)**: Steps before per-hash penalty expires
- **GLOBAL_INTERACT_COOLDOWN (10)**: Steps before global cooldown expires
- **PER_HASH_PENALTY (0.5)**: Penalty when per-hash cooldown active
- **GLOBAL_PENALTY (0.1)**: Penalty when global cooldown active
- **ATTEMPT_THRESHOLD (1)**: Attempts before escalation triggers (escalates at 2nd attempt)
- **ESCALATION_MULTIPLIER (0.5)**: Multiplier for escalating penalty
- **ESCALATION_CAP (1.5)**: Maximum escalation penalty
- **ATTEMPT_DECAY_INTERVAL (200)**: Steps between per-hash decay
- **ATTEMPT_DECAY_FACTOR (0.5)**: Multiplier for decaying attempts
- **GLOBAL_DECAY_INTERVAL (300)**: Steps between global decay

**Layer Reward/Penalty Formulas:**
```python
Layer 1: +4.0 (if is_door_like OR door_successes > 0)
Layer 2: +0.2 - 0.5 (if per_hash_cooldown_active)
Layer 3: +0.2 - 0.1 (if global_cooldown_active)
Layer 4: +0.2 (if first_time_non_door)
Layer 5: +0.2 - min(0.5 × (attempts - 1), 1.5) (else)
```

**Recovery Path:**
- Worst case penalty: +0.2 - 1.5 = -1.3 per step
- Forward movement reward: +2.0 per step (amplified)
- Net: Forward movement dominates escalation, AI can always escape
- Global decay: Every 300 steps, all hashes automatically reset attempts

### Curiosity System (Dual-Layer with Suppression)
**State-Signature Curiosity**:
- **State change reward (+0.6)**: Bonus when any of 10 attributes change
- **Stagnation penalty (-0.1)**: Per frame after 3 unchanged frames
- **Suppression**: Blocked for 8 frames post-interact, post-map, post-camera/mount

**Visual Pixel Curiosity**:
- **Large visual change (+0.1)**: Reward threshold at diff > 5
- **Low visual change penalty (-0.05)**: Applied when diff < 2 for 3+ frames
- **Suppression**: Blocked when map UI open or post-camera/mount action

### Interact Reward System
To adjust interact behavior, modify values in [ai_agent.py](ai_agent.py) step() method (lines 590-625, 1360-1400):
- **Successful state-change (+4.0)**: Base reward for prompt/inventory change (was +20.0)
- **Novelty bonus (+2.0)**: Extra reward for discovering new prompt hash (NEW)
- **Repetition penalty (-1.0)**: Applied when interacting on 0% success hash (NEW)
- **Failed attempt, first try (+0.0)**: Free test on locked doors
- **Failed attempt, retry (-0.5)**: Penalty for re-spamming same prompt
- **Spam within 5 steps (-1.0)**: Rapid re-attempt penalty
- **Dwell penalty (-0.5/step)**: Applied after 2+ steps of visible prompt without E press
- **Ground items (+0.7)**: Bonus for picking up loot

### Movement Reward System
To adjust movement behavior, modify values in [ai_agent.py](ai_agent.py) step() method (lines 1240-1280):
- **Forward movement (+1.5)**: Base reward per step forward (was +1.0)
- **Movement momentum (+0.5)**: Bonus every 10 consecutive frames in same direction (NEW)
- **Momentum reset**: Automatically resets to 0 when interact action (12) is taken (NEW)
- **Backward movement (-0.5)**: Penalty for moving opposite direction
- **Sideways movement (-0.10)**: Light penalty for off-axis movement

### Reward Scaling Guidance
To adjust overall AI behavior, modify reward values:
- **Lower state-change reward** (currently +0.4): Reduces exploration incentive, focus on targets
- **Raise stagnation penalty** (currently -0.1): More aggressive exploration, less tolerance for standing still
- **Adjust visual penalty** (currently -0.05): Tune pixel-level sensitivity
- **Adjust direction penalties** (currently -0.4): Prevent or allow direction switching
- **Adjust curiosity thresholds**: State (3 frames), visual (3 frames) control responsiveness
- **Normalization**: All values designed to fit ±5 range for PPO stability

### Hash Tracking System
- **Location**: `self.prompt_hash_stats` dictionary (initialized in `__init__`)
- **Structure**: `{hash: {'attempts': N, 'door_successes': N, 'last_attempt_step': S}}`
- **None Handling**: None hashes map to "__NONE__" sentinel (line 1063), escalation skips this bucket
- **Attempt Increments**: Before layer evaluation (line 1098), happens exactly once per interact
- **Decay Triggers**: 
  - Per-hash: Auto-halving every 200 steps (line 486-489)
  - Global: Auto-halving every 300 steps for ALL hashes (line 490-493)
- **Partial Signals**: Detecting greying/position/exits triggers `door_successes += 1` and attempt reset (lines 846-854)
- **Purpose**: Track unique prompts, success rates, bootstrap learning once door_successes > 0

## Future Improvements

- [ ] Enemy proximity detection for better combat classification
- [ ] Projectile/spell attack detection
- [ ] NPC interaction tracking
- [ ] Multi-objective reward balancing
- [ ] Curriculum learning (progressive difficulty)
- [ ] Demonstration augmentation from model failures
- [ ] Uncertainty estimation for exploration
- [ ] Visualization of hash tracking data for prompt analysis

## License

Educational project for reinforcement learning research.
