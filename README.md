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

### Movement (Exploration Primary)
- **Forward Movement**: +1.0 (strong incentive for exploration)
- **Momentum Bonus**: +0.25 for continuing forward
- **Backward Movement**: -0.5 (opposite of goal)
- **Sideways Movement**: -0.10 (not primary direction)

### Door Interaction (Critical)
- **Door with White Prompt + Interact**: +5.0 (highest reward)
- **Ignoring Ready Door**: -10.0 (catastrophic penalty)
- **Door with Grey Prompt**: +1.0 for waiting
- **Door with Wizened Finger**: +1.0 when pressing E

### Item Interaction
- **Interact with Visible Items**: +0.7 (strong incentive to loot)
- **General Interact**: +0.5 (NPCs, other interactions)

### Map (Heavily Suppressed)
- **Opening Map**: -15.0 (severe penalty, returns early to suppress other rewards)
- **Any Action While Map Open**: -0.5 (encourages immediate close)
- **Closing Map Quickly**: +0.1 (minimal recovery, net still -14.9)

### Combat Actions
- **Backstep/Dodge**: -1.0 (never dodge during exploration)
- **Jump**: -1.0 (never jump during exploration)
- **Using Items**: -0.5 (no item use during exploration)
- **Lock-on**: -0.2 (combat prep, avoided)
- **Summon Mount**: -0.1 (unavailable, penalties avoided action)

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
- Status buildup detection needs calibration once agent encounters poison/rot/bleed
- Ground item detection uses brightness threshold (may need tuning for different lighting)
- No direct enemy health bar detection (only boss bars)
- Camera control is mouse-based (may be noisy in some situations)

## Performance Metrics

### Action Distribution (Recent Session)
- Movement (forward/left/right): ~6-7% each (good exploration)
- Interact: ~5-6% (appropriate door/item focus)
- Map: ~5% (still high, working on further suppression)
- Camera adjustments: ~4-6% (controlled with penalties)
- Combat actions: <5% (correctly avoided during exploration)

### Training Efficiency
- Training pause: ~2-3 seconds between rollouts (reduced from 5-6s)
- Frame processing: Real-time capture and grayscale conversion
- Observation validation: Active assertions on every step

## Tuning Parameters

### Reward Scaling
To adjust AI behavior, modify reward values in `ai_agent.py` `_calculate_reward()`:
- Increase forward movement reward to encourage more exploration
- Adjust door/item bonuses to shift focus priorities
- Modify penalty values (map, dodging, etc.) to change constraint strength

### Observation Processing
In `ai_agent.py` `FrameStackWrapper`:
- `frame_size = 84` - Resize target (lower = faster, higher = more detail)
- `num_stack = 4` - Temporal context window (higher = more motion info)

### Training Configuration
In `ai_agent.py` `AIAgent.__init__()`:
- `n_steps` - Trajectory collection length (higher = fewer pauses, more off-policy)
- `n_epochs` - Gradient updates per rollout (higher = more training per pause)
- `learning_rate` - Policy update speed (higher = faster learning, less stable)

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
