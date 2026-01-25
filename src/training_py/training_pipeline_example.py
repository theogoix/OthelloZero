"""
Example script showing how to run the training pipeline
"""
from pathlib import Path
from config import PipelineConfig
from training_pipeline import TrainingPipeline


def run_quick_test():
    """Run a quick test with minimal settings (for testing)"""
    config = PipelineConfig()
    
    # Quick test settings
    config.num_iterations = 3
    config.selfplay.num_games = 5
    config.selfplay.simulations_per_move = 100
    config.training.num_epochs = 3
    config.training.batch_size = 128
    config.arena.num_games = 3
    config.arena.simulations_per_move = 200
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parents[2]

    config.base_dir = BASE_DIR
    config.selfplay_binary = BASE_DIR / "bin" / "datagen"
    config.arena_binary = BASE_DIR / "bin" / "arena"
    config.models_dir = BASE_DIR / "models" / "test"
    config.data_dir = BASE_DIR / "data" / "test"
    
    # Save config
    config.save(config.logs_dir / "quick_test_config.json")
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


def run_standard_training():
    """Run standard training (recommended settings)"""
    config = PipelineConfig()
    
    # Standard settings
    config.num_iterations = 50
    config.selfplay.num_games = 400
    config.selfplay.simulations_per_move = 200
    config.training.num_epochs = 10
    config.training.batch_size = 256
    config.arena.num_games = 100
    config.arena.simulations_per_move = 800
    
    # Model settings
    config.training.model_size = 'small'  # or 'medium', 'large'
    
    # Data management
    config.data_window_size = 5  # Use last 5 iterations
    
    # Promotion criteria
    config.promotion_threshold = 0.55  # Need 55% win rate
    config.require_significance = True  # Require statistical significance
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parents[2]

    config.base_dir = BASE_DIR
    config.selfplay_binary = BASE_DIR / "bin" / "datagen"
    config.arena_binary = BASE_DIR / "bin" / "arena"
    config.models_dir = BASE_DIR / "models" / "standard"
    config.data_dir = BASE_DIR / "data" / "standard"
    
    # Save config
    config.save(config.logs_dir / "standard_training_config.json")
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


def run_intensive_training():
    """Run intensive training (for serious training runs)"""
    config = PipelineConfig()
    
    # Intensive settings
    config.num_iterations = 100
    config.selfplay.num_games = 2000
    config.selfplay.simulations_per_move = 1600
    config.selfplay.batch_size = 64
    config.training.num_epochs = 15
    config.training.batch_size = 512
    config.training.learning_rate = 0.001
    config.arena.num_games = 200
    config.arena.simulations_per_move = 1600
    
    # Larger model
    config.training.model_size = 'medium'
    
    # More data history
    config.data_window_size = 10
    
    # Stricter promotion
    config.promotion_threshold = 0.55
    config.require_significance = True
    
    # Paths
    config.base_dir = Path("../../")
    config.selfplay_binary = "./build/selfplay"
    config.arena_binary = "./build/arena"
    
    # Save config
    config.save(config.logs_dir / "intensive_training_config.json")
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


def resume_from_iteration(iteration: int):
    """Resume training from a specific iteration"""
    # Load existing config
    config_path = Path("../../logs/pipeline_config.json")
    
    if config_path.exists():
        config = PipelineConfig.load(config_path)
    else:
        print("No existing config found, using defaults")
        config = PipelineConfig()
    
    # Set starting iteration
    config.start_iteration = iteration
    
    print(f"Resuming from iteration {iteration}")
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


def custom_training():
    """Custom training configuration"""
    config = PipelineConfig()
    
    # Customize each component
    
    # Self-play settings
    config.selfplay.num_games = 1500
    config.selfplay.simulations_per_move = 1000
    config.selfplay.temperature_threshold_move = 30
    config.selfplay.exploration_temperature = 1.0
    config.selfplay.exploitation_temperature = 0.1
    config.selfplay.c_puct = 1.5
    config.selfplay.use_dirichlet = True
    config.selfplay.chunk_save = 100  # Save every 100 games
    
    # Training settings
    config.training.model_size = 'small'
    config.training.num_epochs = 12
    config.training.batch_size = 256
    config.training.learning_rate = 0.001
    config.training.weight_decay = 1e-4
    config.training.use_augmentation = True
    config.training.use_mix_z_q = True
    
    # Arena settings
    config.arena.num_games = 150
    config.arena.simulations_per_move = 1000
    config.arena.temperature = 0.1  # Low temp = strong play
    config.arena.alternate_colors = True
    
    # Pipeline settings
    config.num_iterations = 75
    config.data_window_size = 7
    config.promotion_threshold = 0.55
    config.require_significance = True
    
    # Paths (adjust to your setup)
    config.base_dir = Path("../../")
    config.models_dir = Path("../../models")
    config.data_dir = Path("../../data")
    config.logs_dir = Path("../../logs")
    config.selfplay_binary = "./build/selfplay"
    config.arena_binary = "./build/arena"
    
    # Optional: start from a pre-trained model
    # config.initial_model_path = "path/to/pretrained_model.pt"
    
    # Save config
    config.save(config.logs_dir / "custom_config.json")
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            print("Running quick test...")
            run_quick_test()
        
        elif mode == "standard":
            print("Running standard training...")
            run_standard_training()
        
        elif mode == "intensive":
            print("Running intensive training...")
            run_intensive_training()
        
        elif mode == "resume":
            if len(sys.argv) > 2:
                iteration = int(sys.argv[2])
                resume_from_iteration(iteration)
            else:
                print("Usage: python run_pipeline_example.py resume <iteration>")
        
        elif mode == "custom":
            print("Running custom training...")
            custom_training()
        
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: test, standard, intensive, resume, custom")
    
    else:
        # Default: run standard training
        print("Running standard training (use 'test', 'standard', 'intensive', 'resume', or 'custom' as argument)")
        run_standard_training()