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

## Reward Structure

### Interaction System (State-Change Gated)
The new interact system uses **state-change detection** to determine whether an interaction was successful:

1. **Player presses E** on a valid prompt → defer reward calculation
2. **Next frame check**:
   - **Prompt disappeared** → STATE CHANGED → +20.0 (successful interaction)
   - **Inventory changed** → STATE CHANGED → +20.0 (obtained item, can retry locked doors)
   - **Neither changed, first attempt** → NO STATE CHANGE → +0.0 (neutral test)
   - **Neither changed, retry** → NO STATE CHANGE → -0.5 (penalty for re-spamming)

This design:
- Rewards successful door openings and item pickups (+20.0)
- Allows one free test attempt on locked doors without penalty
- Penalizes repeated failures on locked doors (-0.5)
- Resets the free attempt when inventory changes (AI can try again after getting the item)

### Movement (Exploration Primary)
- **Forward Movement**: +1.0 (strong incentive for exploration)
- **Momentum Bonus**: +0.25 for continuing forward
- **Backward Movement**: -0.5 (opposite of goal)
- **Sideways Movement**: -0.10 (not primary direction)

### Door Interaction (Critical - State-Change Gated)
- **Interact with Valid Prompt (State Change)**: +20.0 (prompt disappears, door opens, item picked up, etc.)
- **Interact Attempt (No State Change, First Try)**: +0.0 (neutral - test if door is locked without penalty)
- **Interact Attempt (No State Change, Retry)**: -0.5 (penalty for re-spamming locked door)
- **Interact After Inventory Change**: +20.0 (obtained item, can now open locked door)
- **Dwell Time Penalty**: -0.5 per step after 2+ consecutive steps of visible prompt without pressing E (exploration only)

### Item Interaction
- **Interact with Ground Items**: +0.7 (bonus for looting)

### Map (Heavily Suppressed)
- **Opening Map**: -15.0 (severe penalty, returns early to suppress other rewards)
- **Any Action While Map Open**: -0.5 (encourages immediate close)
- **Closing Map Quickly**: +0.1 (minimal recovery, net still -14.9)

### Combat Actions
- **Attack Actions (Normal/Heavy)**: -0.2 (combat avoided during exploration)
- **Backstep/Dodge**: -1.0 (never dodge during exploration)
- **Jump**: -1.0 (never jump during exploration)
- **Using Items**: -10.0 (strict constraint in exploration)
- **Lock-on**: -0.2 (combat prep, avoided)
- **Skill**: -0.3 (skill use avoided)
- **Summon Mount**: -0.1 (unavailable, minimal penalty)

### Wasted Interact Penalties
- **Spam within 5 steps of last attempt**: -1.0 (rapid re-attempt penalty)
- **Attempt after 5+ steps**: -0.5 (lighter penalty for spaced attempts)

### Camera & Navigation
- **Camera Panning**: -0.15 (small penalty, not neutral)
- **Time Penalty**: -0.003 per step (encourage speed)

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

1. **Never Use Items** (-0.5 penalty)
2. **Never Dodge/Backstep** (-1.0 penalty - wasted stamina)
3. **Never Jump** (-1.0 penalty - pointless action)
4. **Never Open Map** (-15.0 penalty - immediate application, suppresses other rewards)
5. **Prefer Forward Movement** (primary direction)

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
- ✅ CNN + LSTM architecture fully integrated
- ✅ Frame stacking with validation (84×84×4)
- ✅ Health damage tracking and combat detection
- ✅ Door detection and interaction rewards
- ✅ Ground item detection and collection incentives
- ✅ Status effect buildup detection (foundation for environmental damage)
- ✅ Map opening severely penalized with immediate penalty application
- ✅ Momentum enforcement to prevent dithering
- ✅ All core files compile with Python 3.12
- ✅ Behavioral cloning infrastructure available

### Known Limitations
- State-change detection based on prompt disappearance and inventory changes (doesn't detect all in-game state changes like dialogue advancement)
- Ground item detection uses brightness threshold (may need tuning for different lighting)
- No direct enemy health bar detection (only boss bars)
- Camera control is mouse-based (may be noisy in some situations)
- Floor messages and item prompts may stack on-screen (edge case not fully handled)

### Recent Changes (Latest Session)
- ✅ Implemented state-change gated interact rewards (replaces time-based logic)
- ✅ Added inventory tracking for retry validation
- ✅ Implemented dwell time penalty (-0.5/step after 2+ steps without pressing E)
- ✅ Added interact diagnostics tracking (successful, wasted, missed opportunities)
- ✅ Removed void detection system (unreliable, replaced with learned behavior from health damage)
- ✅ Fixed missed opportunities counter to count prompt appearances, not persistence steps
- ✅ One free attempt to validate locked doors, penalty only on retry (-0.5)

## Performance Metrics

### Action Distribution (Most Recent Training Session - 26,374 steps)
- **Interact**: 7.46% (1,967 attempts, 100% successful on valid targets)
- **Jump**: 7.21% 
- **Move Left**: 11.93%
- **Move Right**: 7.15%
- **Move Forward**: 7.19%
- **Open Map**: 5.64%
- **Camera Actions**: ~13% combined
- **Combat Actions**: <5% (correctly avoided)

### Interact Breakdown
- **Successful (valid target)**: 1,967 / 1,967 (100%)
- **Wasted (no target)**: 0 (filtered by state-change gating)
- **Missed opportunities**: Varies by episode (prompts seen but E not pressed)

### Training Characteristics
- **Learned behavior**: AI no longer avoids doors; presses E on valid prompts
- **Spam control**: State-change detection prevents infinite message reading
- **Test-and-retry**: Free attempt on locked doors encourages persistence with inventory changes

## Tuning Parameters

### Interact Reward System
To adjust interact behavior, modify values in `ai_agent.py` `step()` → `_calculate_reward()`:
- **Successful interaction (+20.0)**: Base reward for state-change detection success
- **Failed attempt, first try (+0.0)**: Free test on locked doors
- **Failed attempt, retry (-0.5)**: Penalty for re-spamming same prompt
- **Dwell penalty (-0.5/step)**: Applied after 2+ steps of visible prompt without E press
- **Inventory change reset**: Clears failed attempt tracking, allows fresh try

### Reward Scaling
To adjust overall AI behavior, modify reward values:
- Increase forward movement reward to encourage more exploration
- Adjust interact bonuses to shift focus priorities
- Modify penalty values (map, combat, etc.) to change constraint strength

## Future Improvements

- [ ] Enemy proximity detection for better combat classification
- [ ] Projectile/spell attack detection
- [ ] NPC interaction tracking
- [ ] Multi-objective reward balancing
- [ ] Curriculum learning (progressive difficulty)
- [ ] Demonstration augmentation from model failures
- [ ] Uncertainty estimation for exploration

## License

Educational project for reinforcement learning research.
