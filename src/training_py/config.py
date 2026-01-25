"""
Configuration classes for the training pipeline
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import json


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation"""
    num_games: int = 1000
    simulations_per_move: int = 800
    batch_size: int = 32
    
    # Temperature schedule
    temperature_threshold_move: int = 30
    exploration_temperature: float = 1.0
    exploitation_temperature: float = 0.1
    
    # MCTS parameters
    c_puct: float = 1.5
    use_dirichlet: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Performance
    use_async: bool = True
    chunk_save: int = 50  # Save every N games
    
    verbose: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model architecture
    model_size: str = 'small'  # 'tiny', 'small', 'medium', 'large'
    num_filters: Optional[int] = None
    num_res_blocks: Optional[int] = None
    
    # Training hyperparameters
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Data
    train_split: float = 0.9
    use_augmentation: bool = True
    use_mix_z_q: bool = True
    
    # Fine-tuning
    pretrained_path = None  # Path to checkpoint to continue from
    reset_optimizer = False  # If True, don't load optimizer state


    # Hardware
    device: str = "cuda"  # Will auto-detect if cuda available
    num_workers: int = 4
    
    # Checkpointing
    save_every: int = 1
    log_every: int = 100
    
    verbose: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        return config


@dataclass
class ArenaConfig:
    """Configuration for model evaluation"""
    num_games: int = 100
    simulations_per_move: int = 800
    batch_size: int = 32
    
    # Match settings
    alternate_colors: bool = True
    temperature: float = 0.1  # Low temperature for strong play
    
    # MCTS parameters
    c_puct: float = 1.5
    use_dirichlet: bool = False  # Usually off for evaluation
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    use_async: bool = True
    verbose: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the entire training pipeline"""
    
    # Pipeline settings
    num_iterations: int = 100
    start_iteration: int = 0  # For resuming
    
    # Self-play settings
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    
    # Training settings
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Evaluation settings
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    
    # Model promotion criteria
    promotion_threshold: float = 0.55  # Win rate needed to promote
    require_significance: bool = True  # Require statistical significance (95%)
    
    # Data management
    data_window_size: int = 5  # Keep last N iterations of data
    
    # Paths (as strings, will be converted to Path in __post_init__)
    base_dir: str = "."
    models_dir: str = "models"
    data_dir: str = "data"
    logs_dir: str = "logs"
    checkpoints_dir: str = "models/checkpoints"
    
    # C++ binaries
    selfplay_binary: str = "./build/selfplay"
    arena_binary: str = "./build/arena"
    
    # Initial model (for iteration 0)
    initial_model_path: Optional[str] = None  # If None, train from random
    
    verbose: bool = True
    
    def __post_init__(self):

        root = Path(__file__).resolve().parents[2]  # project root

        def resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (root / p)


        """Convert string paths to Path objects"""
        # Convert all path strings to Path objects
        self.base_dir = resolve(self.base_dir)
        self.models_dir = resolve(self.models_dir)
        self.data_dir = resolve(self.data_dir)
        self.logs_dir = resolve(self.logs_dir)
        self.checkpoints_dir = resolve(self.checkpoints_dir)
    
    def create_directories(self):
        """Create all necessary directories"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "archived").mkdir(exist_ok=True)
        (self.logs_dir / "selfplay").mkdir(exist_ok=True)
        (self.logs_dir / "training").mkdir(exist_ok=True)
        (self.logs_dir / "arena").mkdir(exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization"""
        def convert_value(v):
            if isinstance(v, Path):
                return str(v)
            elif hasattr(v, 'to_dict'):
                return v.to_dict()
            elif hasattr(v, '__dict__'):
                return {k: convert_value(val) for k, val in v.__dict__.items()}
            else:
                return v
        
        return {k: convert_value(v) for k, v in self.__dict__.items()}
    
    def save(self, path: Path):
        """Save configuration to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested configs
        config = cls()
        
        # Update selfplay config
        if 'selfplay' in data and isinstance(data['selfplay'], dict):
            for k, v in data['selfplay'].items():
                if hasattr(config.selfplay, k):
                    setattr(config.selfplay, k, v)
        
        # Update training config
        if 'training' in data and isinstance(data['training'], dict):
            for k, v in data['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        # Update arena config
        if 'arena' in data and isinstance(data['arena'], dict):
            for k, v in data['arena'].items():
                if hasattr(config.arena, k):
                    setattr(config.arena, k, v)
        
        # Update top-level config
        for k, v in data.items():
            if k not in ['selfplay', 'training', 'arena'] and hasattr(config, k):
                setattr(config, k, v)
        
        return config
    
    def get_model_path(self, iteration: int, best: bool = True) -> Path:
        """Get path for a model at given iteration"""
        if best:
            return self.models_dir / f"model_iter_{iteration}_best.pt"
        else:
            return self.checkpoints_dir / f"model_iter_{iteration}.pt"
    
    def get_data_path(self, iteration: int) -> Path:
        """Get path for self-play data at given iteration"""
        return self.data_dir / f"iteration_{iteration}.bin"
    
    def get_arena_results_path(self, iteration: int) -> Path:
        """Get path for arena results"""
        return self.logs_dir / "arena" / f"arena_iter_{iteration}.csv"