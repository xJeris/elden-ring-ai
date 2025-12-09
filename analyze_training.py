#!/usr/bin/env python3
"""
Quick script to analyze training checkpoints without running more training.
Usage:
    python analyze_training.py
"""

from ai_agent import AIAgent, EldenRingEnv, FrameStackWrapper

def main():
    print("\n" + "="*70)
    print("ELDEN RING AI - CHECKPOINT ANALYSIS TOOL")
    print("="*70)
    
    # Create environment and agent
    env = EldenRingEnv()
    env = FrameStackWrapper(env)
    agent = AIAgent(env)
    
    # Load latest checkpoint
    if agent.load_latest_checkpoint():
        print("\n✓ Checkpoint loaded successfully!\n")
        
        print("\n1. LEARNING ANALYSIS (Latest Checkpoint)")
        print("─" * 70)
        agent.analyze_learning()
        
        print("\n2. TRAINING HISTORY (All Checkpoints)")
        print("─" * 70)
        agent.analyze_training_history()
        
        print("\n3. MODEL ARCHITECTURE")
        print("─" * 70)
        agent.test_checkpoint_behavior()
        
    else:
        print("\n✗ No checkpoints found!")
        print("Run training first with: python main.py")

if __name__ == "__main__":
    main()
