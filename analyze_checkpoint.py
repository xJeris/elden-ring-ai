"""
Quick analysis script to examine what a trained checkpoint learned
"""

from ai_agent import AIAgent, EldenRingEnv, FrameStackWrapper
import sys

def analyze_latest_checkpoint():
    """Load and analyze the latest checkpoint"""
    
    # Create environment and agent
    env = EldenRingEnv()
    env = FrameStackWrapper(env)
    agent = AIAgent(env)
    
    print("\n" + "="*70)
    print("CHECKPOINT ANALYSIS TOOL")
    print("="*70)
    
    # Try to load latest checkpoint
    if agent.load_latest_checkpoint():
        print("✓ Checkpoint loaded successfully")
        
        # Run analysis
        print("\n[1] Analyzing learned behavior patterns...")
        agent.analyze_learning()
        
        print("\n[2] Testing behavior in environment for 500 steps...")
        agent.test_checkpoint_behavior(num_steps=500)
        
        print("\nAnalysis complete!")
    else:
        print("✗ No checkpoint found to analyze")
        print("   Train a model first before analyzing")

if __name__ == "__main__":
    analyze_latest_checkpoint()
