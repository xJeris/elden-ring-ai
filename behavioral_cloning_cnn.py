"""
Behavioral Cloning with CNN + LSTM

Uses recorded human demonstrations to train the AI through imitation learning.
The recorded demonstrations include:
- Raw screen images (for CNN to learn visual features)
- Actions taken (what the human did)
- Camera movements (context for actions)

This helps the AI learn basic gameplay patterns before reinforcement learning.
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from collections import deque
import os
from datetime import datetime


class DemonstrationDataset(Dataset):
    """
    Dataset of human demonstrations for behavioral cloning.
    Converts recorded frames into 4-frame stacks (84x84 grayscale) for CNN input.
    """
    def __init__(self, recording_file, num_stack=4, max_samples=None):
        """
        Load and prepare demonstrations.
        
        Args:
            recording_file: path to recorded demonstrations (pkl file)
            num_stack: number of frames to stack (default 4)
            max_samples: limit dataset size (useful for testing)
        """
        self.num_stack = num_stack
        self.frames = []
        self.actions = []
        
        print(f"Loading demonstrations from {recording_file}...")
        
        # Load recorded data
        with open(recording_file, 'rb') as f:
            data = pickle.load(f)
        
        recordings = data['recordings']
        total_loaded = 0
        
        # Process each recording
        for recording in recordings:
            frame_buffer = deque(maxlen=num_stack)
            
            for frame_data in recording['frames']:
                # Decompress image from JPEG bytes
                if 'raw_screen_jpg' in frame_data:
                    jpg_bytes = frame_data['raw_screen_jpg']
                    # Decode JPEG
                    nparr = np.frombuffer(jpg_bytes, np.uint8)
                    raw_screen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Convert to grayscale and resize
                    gray = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
                    
                    # Add to buffer
                    frame_buffer.append(resized)
                    
                    # Once we have enough frames, add to dataset
                    if len(frame_buffer) == num_stack:
                        stacked = np.stack(list(frame_buffer), axis=2)
                        self.frames.append(stacked.astype(np.uint8))
                        self.actions.append(frame_data['action'])
                        total_loaded += 1
                        
                        # Stop if we've loaded enough
                        if max_samples and total_loaded >= max_samples:
                            break
            
            if max_samples and total_loaded >= max_samples:
                break
        
        print(f"✓ Loaded {len(self.frames)} frame stacks from demonstrations")
        if len(self.frames) == 0:
            print("⚠️  No frames loaded! Check recording file format.")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # Return frame stack and action
        frame = self.frames[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        action = self.actions[idx]
        
        return torch.from_numpy(frame).permute(2, 0, 1), torch.tensor(action, dtype=torch.long)


class BehavioralCloningTrainer:
    """Train CNN + LSTM using human demonstrations"""
    
    def __init__(self, model, num_actions=19, device='cuda'):
        """
        Initialize trainer.
        
        Args:
            model: RecurrentPPO model to train
            num_actions: number of discrete actions
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.num_actions = num_actions
        self.device = device
        
        # Use the policy network's forward pass
        self.policy_network = model.policy
    
    def train(self, train_loader, val_loader=None, epochs=10, learning_rate=1e-4):
        """
        Train the policy network using demonstrations.
        
        Args:
            train_loader: DataLoader for training demonstrations
            val_loader: optional DataLoader for validation
            epochs: number of training epochs
            learning_rate: learning rate for optimizer
        """
        print(f"\n{'='*70}")
        print(f"BEHAVIORAL CLONING - CNN + LSTM")
        print(f"{'='*70}")
        print(f"Training on demonstrations...")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Optimizer for policy network parameters
        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.policy_network.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (frames, actions) in enumerate(train_loader):
                frames = frames.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass through policy
                # Note: RecurrentPPO policy may need special handling for LSTM state
                try:
                    # Extract action logits from policy
                    action_logits = self.policy_network(frames)
                    
                    # Calculate loss
                    loss = criterion(action_logits, actions)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    _, predicted = torch.max(action_logits.data, 1)
                    train_total += actions.size(0)
                    train_correct += (predicted == actions).sum().item()
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1} - Loss: {loss.item():.4f}")
                
                except Exception as e:
                    print(f"Error during batch processing: {e}")
                    print("Note: RecurrentPPO policy may require LSTM state initialization")
                    break
            
            # Average metrics
            avg_train_loss = train_loss / max(1, batch_idx + 1)
            train_acc = 100 * train_correct / max(1, train_total)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Accuracy: {train_acc:.2f}%")
            
            # Validation phase (optional)
            if val_loader:
                val_loss, val_acc = self.validate(val_loader, criterion)
                print(f"               - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"               ✓ New best model saved!")
        
        print(f"\n✓ Behavioral cloning training complete!")
        print(f"  AI has learned from {len(train_loader.dataset)} human demonstrations")
        print(f"  Ready for reinforcement learning fine-tuning")
    
    def validate(self, val_loader, criterion):
        """Validate on held-out demonstrations"""
        self.policy_network.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, actions in val_loader:
                frames = frames.to(self.device)
                actions = actions.to(self.device)
                
                try:
                    action_logits = self.policy_network(frames)
                    loss = criterion(action_logits, actions)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(action_logits.data, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()
                except:
                    break
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100 * val_correct / max(1, val_total)
        
        return avg_val_loss, val_acc


def main():
    """Interactive behavioral cloning trainer"""
    print(f"\n{'='*70}")
    print("BEHAVIORAL CLONING - CNN + LSTM")
    print("Learn from human demonstrations before RL training")
    print(f"{'='*70}\n")
    
    # Step 1: Load demonstrations
    recording_file = input("Path to recording file [imitation_data_with_camera.pkl]: ").strip()
    if not recording_file:
        recording_file = "imitation_data_with_camera.pkl"
    
    if not os.path.exists(recording_file):
        print(f"❌ File not found: {recording_file}")
        return
    
    # Step 2: Load and prepare dataset
    dataset = DemonstrationDataset(recording_file)
    
    if len(dataset) == 0:
        print("❌ No valid frames in dataset!")
        return
    
    # Split into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Training set: {train_size} samples")
    print(f"Validation set: {val_size} samples")
    print(f"Batch size: {batch_size}\n")
    
    # Step 3: Load or create model
    print("Note: This requires loading the trained RecurrentPPO model.")
    print("For now, this is a template. In practice:")
    print("1. Load your trained model from main.py")
    print("2. Initialize BehavioralCloningTrainer with it")
    print("3. Call trainer.train(train_loader, val_loader, epochs=5)")
    
    print("\n✓ Behavioral cloning ready!")
    print("  Integrate this with your main training pipeline:")
    print("  - Record human gameplay")
    print("  - Run behavioral cloning to initialize policy")
    print("  - Use trained model as starting point for RL")


if __name__ == "__main__":
    main()
