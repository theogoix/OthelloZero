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
    config.start_iteration = 48
    config.num_iterations = 60
    config.selfplay.num_games = 400
    config.selfplay.simulations_per_move = 800
    config.training.num_epochs = 20
    config.training.batch_size = 256
    config.arena.num_games = 100
    config.arena.simulations_per_move = 800
    
    # Model settings
    config.training.model_size = 'small'
    
    # Data management
    config.data_window_size = 10  # Use last 5 iterations
    
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