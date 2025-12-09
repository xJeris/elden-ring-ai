"""
Behavioral cloning - pre-train the AI to imitate human demonstrations.
This gives the model a good starting policy before RL fine-tuning.
"""

import json
import os
import numpy as np
from ai_agent import EldenRingEnv, FrameStackWrapper
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DemonstrationDataset(Dataset):
    """PyTorch dataset for demonstrations"""
    
    def __init__(self, demonstrations_file):
        with open(demonstrations_file, 'r') as f:
            data = json.load(f)
        
        self.frames = []
        self.actions = []
        
        # Extract all frames and their corresponding actions
        for demo in data['demonstrations']:
            for frame in demo['frames']:
                self.frames.append({
                    'health': frame['health'],
                    'stamina': frame['stamina'],
                    'exits': frame['exits'],
                    'is_outdoor': frame['is_outdoor'],
                    'in_combat': frame['in_combat'],
                })
                self.actions.append(frame['action'])
        
        print(f"Loaded {len(self.frames)} frames from demonstrations")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        action = self.actions[idx]
        
        # Create feature vector from game state
        features = np.array([
            frame['health'],
            frame['stamina'],
            frame['exits']['total'],
            float(frame['is_outdoor']),
            float(frame['in_combat']),
        ], dtype=np.float32)
        
        return features, action


def behavioral_clone_from_demonstrations(demonstrations_file, model_save_path="models/bc_pretrained"):
    """
    Pre-train the policy using behavioral cloning on human demonstrations.
    
    Args:
        demonstrations_file: Path to JSON file with recorded demonstrations
        model_save_path: Where to save the pre-trained model
    """
    print(f"\n{'='*70}")
    print("BEHAVIORAL CLONING - PRE-TRAINING FROM HUMAN DEMONSTRATIONS")
    print(f"{'='*70}\n")
    
    # Create environment and model
    env = EldenRingEnv()
    env = FrameStackWrapper(env)  # Convert to 84x84 grayscale stacked frames for CNN
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    
    # Load demonstrations
    dataset = DemonstrationDataset(demonstrations_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Behavioral cloning training loop
    print(f"Training behavioral cloning on {len(dataset)} demonstration frames...")
    print(f"{'='*70}\n")
    
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for features, actions in dataloader:
            # Convert to tensors
            features = torch.tensor(features, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            
            # Forward pass through policy
            # Note: This is a simplified version - in practice you'd need to 
            # feed actual observations through the network
            optimizer.zero_grad()
            
            # Get action predictions from policy
            logits = policy.mlp_extractor(features)
            loss = loss_fn(logits, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    
    # Save pre-trained model
    model.save(model_save_path)
    print(f"\n✓ Pre-trained model saved to {model_save_path}")
    print(f"  Next step: Use this model as starting point for RL fine-tuning")
    
    env.close()


def main():
    """Run behavioral cloning training"""
    import sys
    
    demonstrations_file = "demonstrations.json"
    
    # Check if demonstrations file exists
    try:
        with open(demonstrations_file, 'r') as f:
            data = json.load(f)
            demo_count = len(data['demonstrations'])
            frame_count = sum(d['frame_count'] for d in data['demonstrations'])
            
            print(f"\nFound demonstrations file:")
            print(f"  Demonstrations: {demo_count}")
            print(f"  Total frames: {frame_count}")
            
    except FileNotFoundError:
        print(f"\n❌ Demonstrations file not found: {demonstrations_file}")
        print("   Run 'python record_demonstrations.py' first to record human gameplay")
        sys.exit(1)
    
    # Train behavioral cloning
    behavioral_clone_from_demonstrations(demonstrations_file)


if __name__ == "__main__":
    main()
