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

### Interaction System (State-Change Gated + Novelty Bonus)
The interact system uses **state-change detection** plus **novelty tracking** to determine whether an interaction was successful:

1. **Player presses E** on a valid prompt → defer reward calculation
2. **Next frame check**:
   - **Prompt disappeared** → STATE CHANGED → +4.0 base (was +20.0)
   - **Prompt disappeared + New Hash** → STATE CHANGED → +4.0 + +2.0 novelty = +6.0
   - **Inventory changed** → STATE CHANGED → +4.0 base (was +20.0)
   - **Inventory changed + New Hash** → STATE CHANGED → +4.0 + +2.0 novelty = +6.0
   - **Neither changed, first attempt** → NO STATE CHANGE → +0.0 (neutral test)
   - **Neither changed, retry (0% success hash)** → NO STATE CHANGE → -0.5 to -1.5 (spam penalty + repetition penalty)

This design:
- Base interact reward reduced from +20.0 to +4.0 (major rebalance)
- Rewards discovering new prompts with +2.0 novelty bonus
- Penalizes re-attempting locked doors without key with -1.0 repetition penalty
- Allows one free test attempt on locked doors without penalty
- Resets the free attempt when inventory changes (AI can try again after getting the item)

### Movement (Now Primary Focus)
- **Forward Movement**: +1.5 (was +1.0, increased to compete with lower interact rewards)
- **Movement Momentum**: +0.5 every 10 consecutive frames in same direction (NEW - encourages sustained exploration)
- **Momentum Reset**: Resets to 0 whenever interact action is taken
- **Backward Movement**: -0.5 (opposite of goal)
- **Sideways Movement**: -0.10 (not primary direction)

### Interact Actions (Secondary Strategy)
- **Interact with Valid Prompt (State Change)**: +4.0 base (reduced from +20.0)
- **Interact with NEW Hash (Novelty)**: +2.0 bonus (NEW - discovery reward)
- **Interact on 0% Success Hash (Repetition)**: -1.0 penalty (NEW - stops dead-end spamming)
- **Interact Attempt (No State Change, First Try)**: +0.0 (neutral - test if door is locked)
- **Interact Attempt (No State Change, Retry)**: -0.5 (penalty for re-spamming)
- **Spam within 5 steps of last attempt**: -1.0 (rapid re-attempt penalty)
- **Ground Items Pickup**: +0.7 (unchanged - utility loot)
- **Dwell Time Penalty**: -0.5 per step after 2+ consecutive steps of visible prompt without pressing E (exploration only)

### Item Interaction
- **Interact with Ground Items**: +0.7 (bonus for looting)

### Map (Heavily Suppressed)
- **Opening Map**: -15.0 (severe penalty, immediate application suppresses other rewards)
- **Any Action While Map Open**: -0.5/step (continuous penalty encourages closing)
- **Map Reopen Cooldown**: 900 steps (30 seconds) after closing to prevent oscillation
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
- **Time Penalty**: -0.003 per step (encourages speed, normalized from -0.005)

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
- ✅ Momentum enforcement to prevent dithering (direction change penalties)
- ✅ All recent code fixes verified and syntax clean
- ✅ Comprehensive error handling with debug logging (9 try-catch blocks)
- ✅ All core files compile with Python 3.12
- ✅ Behavioral cloning infrastructure available

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

**Expected Results After Rebalancing:**
- Interact drops from ~25% to target ~10% (+ novelty +2, - repetition -1)
- Movement rises to ~50-60% (primary strategy, +1.5 base + momentum +0.5)
- Dead-end exploration reduced (repetition penalty on locked doors without key)
- Novel discovery encouraged (novelty bonus on new prompts)

## Performance Metrics

### Previous Training Session (26,374 steps, 19 episodes)
- **Interact**: 25.53% (8,995 attempts) - **TARGET: Reduce to ~10%**
- **Move Forward**: 11.52% (4,058 actions) - **TARGET: Increase to 50-60%**
- **Move Left/Right**: 15.71% (5,532 actions combined)
- **Camera Actions**: ~13% combined
- **Combat Actions**: <5% (correctly avoided)
- **Prompt Success Rate**: 48% overall (98 unique hashes, 47 successful)

### Hash Analytics from Previous Training
- Total unique prompt signatures: 98
- Successful prompts (100% success): 47
- Failed prompts (0% success): 49 (locked doors, dead-ends)
- Overall prompt success rate: 48.0%

**Insight**: The 49 failed-hash attempts (50% of prompts) will now be penalized with -1.0, encouraging movement exploration instead of prompt spam.

### Training Characteristics
- **Learned behavior**: AI no longer avoids doors; presses E on valid prompts
- **Spam control**: State-change detection prevents infinite message reading
- **Test-and-retry**: Free attempt on locked doors encourages persistence with inventory changes

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
- **Lower interact base** (currently +4.0): Reduces interact frequency below 10%
- **Raise forward movement** (currently +1.5): Makes movement more attractive relative to interact
- **Adjust momentum bonus** (currently +0.5/10fr): Encourage longer exploration streaks
- **Novelty bonus** (currently +2.0): Increase to reward exploration more, decrease to focus on known targets
- **Repetition penalty** (currently -1.0): Increase to strictly avoid dead-ends, decrease to allow persistence
- **Normalization**: All values designed to fit ±5 range for PPO stability

### Hash Tracking System
- **Location**: `self.prompt_hash_stats` dictionary (initialized in `__init__`)
- **Structure**: `{hash: {'attempts': N, 'successes': N}}`
- **Updates**: Tracked in `step()` at lines 596-625 when interact state changes detected
- **Novelty Check** (line 599-600): `if hash not in self.prompt_hash_stats` triggers +2.0 bonus
- **Repetition Check** (line 1387-1391): `if hash in stats AND successes==0` triggers -1.0 penalty
- **Purpose**: Monitor unique prompts, success rates, and guide exploration behavior across episodes

## Debugging & Error Handling

### Debug Logging
Comprehensive try-catch blocks at critical state access points (marked with 🔴 DEBUG):
- Line 460: observation extraction
- Line 470: stuck detection state access
- Line 549: main state extraction
- Line 564: wizened finger detection
- Line 587: interact state change detection
- Line 680: player position access
- Line 803: map UI visible validation
- Line 821: hash attempt tracking
- Line 830: inventory count access
- Line 906: boss health/fog wall access

Each prints line number before raising exception for rapid error diagnosis.

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
