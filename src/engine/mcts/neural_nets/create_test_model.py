#!/usr/bin/env python3
"""
Create a dummy neural network model for testing the MCTS implementation.
This creates a small random network that matches the expected interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OthelloNet(nn.Module):
    """
    Simple neural network for Othello.
    
    Input: [batch, 3, 8, 8]
      - Channel 0: Current player's pieces
      - Channel 1: Opponent's pieces  
      - Channel 2: Color to move
    
    Output: (policy, value)
      - policy: [batch, 64] - move probabilities
      - value: [batch, 1] - position evaluation
    """
    
    def __init__(self, num_filters=64):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)  # Convert to probabilities
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy, value


def create_and_export_model(output_path='model.pt', num_filters=64):
    """Create a random model and export it to TorchScript."""
    
    print(f"Creating OthelloNet with {num_filters} filters...")
    model = OthelloNet(num_filters=num_filters)
    model.eval()
    
    # Test with dummy input
    print("Testing model with dummy input...")
    dummy_input = torch.randn(4, 3, 8, 8)  # Batch of 4
    
    with torch.no_grad():
        policy, value = model(dummy_input)
    
    print(f"  Policy shape: {policy.shape} (expected: [4, 64])")
    print(f"  Value shape: {value.shape} (expected: [4, 1])")
    print(f"  Policy sum: {policy[0].sum():.4f} (should be ~1.0)")
    print(f"  Value range: [{value.min():.4f}, {value.max():.4f}] (should be in [-1, 1])")
    
    # Export to TorchScript
    print(f"\nExporting to {output_path}...")
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    
    # Verify exported model
    print("Verifying exported model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    with torch.no_grad():
        policy2, value2 = loaded_model(dummy_input)
    
    # Check outputs match
    assert torch.allclose(policy, policy2, atol=1e-5), "Policy mismatch!"
    assert torch.allclose(value, value2, atol=1e-5), "Value mismatch!"
    
    print("✓ Model exported and verified successfully!")
    print(f"\nYou can now use this model with the C++ code:")
    print(f"  ./othello_mcts {output_path}")
    
    return model


def create_trained_model_template():
    """
    Template for training a real model with self-play.
    This is just a skeleton - you'll need to implement the training loop.
    """
    
    model = OthelloNet(num_filters=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (pseudo-code)
    for epoch in range(100):
        # 1. Generate self-play games using MCTS
        # games = generate_self_play_games(model, num_games=100)
        
        # 2. Extract training examples (state, policy_target, value_target)
        # examples = extract_training_data(games)
        
        # 3. Train on examples
        # for batch in batches(examples):
        #     states, policy_targets, value_targets = batch
        #     
        #     policy_pred, value_pred = model(states)
        #     
        #     policy_loss = cross_entropy(policy_pred, policy_targets)
        #     value_loss = mse_loss(value_pred, value_targets)
        #     loss = policy_loss + value_loss
        #     
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        pass
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create a neural network model for Othello MCTS'
    )
    parser.add_argument(
        '--output', '-o',
        default='model.pt',
        help='Output path for the model (default: model.pt)'
    )
    parser.add_argument(
        '--filters', '-f',
        type=int,
        default=64,
        help='Number of convolutional filters (default: 64)'
    )
    
    args = parser.parse_args()
    
    model = create_and_export_model(args.output, args.filters)
    
    print("\n" + "="*60)
    print("Note: This is a RANDOM model for testing only!")
    print("For good play, you need to train the model with self-play.")
    print("="*60)