import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class OthelloNet(nn.Module):
    """Configurable neural network for Othello with residual blocks"""
    
    # Predefined model configurations
    CONFIGS = {
        'tiny': {
            'num_filters': 32,
            'num_res_blocks': 3,
            'description': '~100K params, good for <5K examples'
        },
        'small': {
            'num_filters': 64,
            'num_res_blocks': 5,
            'description': '~400K params, good for 5K-25K examples'
        },
        'medium': {
            'num_filters': 128,
            'num_res_blocks': 10,
            'description': '~2.5M params, good for 25K-100K examples'
        },
        'large': {
            'num_filters': 256,
            'num_res_blocks': 20,
            'description': '~10M params, good for 100K+ examples'
        },
    }
    
    def __init__(self, num_filters: int = 128, num_res_blocks: int = 10):
        super().__init__()
        
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution (3 input channels: current, opponent, legal moves)
        # With padding=1 and kernel=3, spatial size stays 8x8
        self.conv_input = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower (each block keeps spatial size at 8x8)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)  # 1x1 conv
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Input: [batch, 3, 8, 8]
        
        # Initial convolution: [batch, 3, 8, 8] → [batch, num_filters, 8, 8]
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks: [batch, num_filters, 8, 8] → [batch, num_filters, 8, 8]
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 8, 8]
        policy = policy.view(policy.size(0), -1)              # [batch, 128]
        policy = self.policy_fc(policy)                        # [batch, 64]
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))     # [batch, 1, 8, 8]
        value = value.view(value.size(0), -1)                 # [batch, 64]
        value = F.relu(self.value_fc1(value))                 # [batch, 64]
        value = torch.tanh(self.value_fc2(value))             # [batch, 1]
        
        return policy, value
    
    @classmethod
    def from_config(cls, config_name: str):
        """Create model from predefined configuration"""
        if config_name not in cls.CONFIGS:
            available = ', '.join(cls.CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        
        config = cls.CONFIGS[config_name]
        return cls(
            num_filters=config['num_filters'],
            num_res_blocks=config['num_res_blocks']
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'num_filters': self.num_filters,
            'num_res_blocks': self.num_res_blocks,
            'total_parameters': self.count_parameters()
        }
    
    @staticmethod
    def print_available_configs():
        """Print all available model configurations"""
        print("\nAvailable model configurations:")
        print("-" * 70)
        for name, config in OthelloNet.CONFIGS.items():
            print(f"{name:10s}: {config['description']}")
        print("-" * 70)