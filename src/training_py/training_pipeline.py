"""
Main training pipeline orchestrator
Implements AlphaZero-style iterative improvement loop
"""
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path to import train module
sys.path.append(str(Path(__file__).parent))

from config import PipelineConfig
from runners.selfplay_runner import SelfPlayRunner
from runners.arena_runner import ArenaRunner
from managers.model_manager import ModelManager
from managers.data_manager import DataManager
from utils.logger import TrainingLogger

# Import training function from train.py
from train import train, TrainingConfig as TrainConfig


class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Create directories
        config.create_directories()
        
        # Initialize managers
        self.model_manager = ModelManager(
            config.models_dir,
            verbose=config.verbose
        )
        
        self.data_manager = DataManager(
            config.data_dir,
            window_size=config.data_window_size,
            verbose=config.verbose
        )
        
        self.logger = TrainingLogger(
            config.logs_dir,
            verbose=config.verbose
        )
        
        # Initialize runners
        self.selfplay_runner = SelfPlayRunner(
            config.selfplay_binary,
            verbose=config.verbose
        )
        
        self.arena_runner = ArenaRunner(
            config.arena_binary,
            verbose=config.verbose
        )
        
        # Save pipeline config
        self.logger.set_metadata('config', config.to_dict())
    
    def run(self):
        """Run the complete training pipeline"""
        if self.config.verbose:
            print("\n" + "="*70)
            print("OTHELLO TRAINING PIPELINE")
            print("="*70)
            print(f"Iterations: {self.config.start_iteration} -> {self.config.num_iterations}")
            print(f"Self-play games per iteration: {self.config.selfplay.num_games}")
            print(f"Training epochs per iteration: {self.config.training.num_epochs}")
            print(f"Arena games per iteration: {self.config.arena.num_games}")
            print("="*70 + "\n")
        
        # Run iterations
        for iteration in range(self.config.start_iteration, self.config.num_iterations):
            try:
                self._run_iteration(iteration)
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user!")
                self.logger.print_summary()
                break
            except Exception as e:
                print(f"\n\nERROR in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Print final summary
        if self.config.verbose:
            self.logger.print_summary()
            self.logger.export_csv(self.config.logs_dir / "training_log.csv")
    
    def _run_iteration(self, iteration: int):
        """Run a single training iteration"""
        iteration_start = time.time()
        self.logger.start_iteration(iteration)
        
        # Phase 1: Generate self-play data
        selfplay_result = self._generate_selfplay_data(iteration)
        if not selfplay_result.success:
            raise RuntimeError(f"Self-play generation failed: {selfplay_result.error_message}")
        
        self.logger.log_selfplay_stats(
            selfplay_result.num_games,
            selfplay_result.num_positions,
            selfplay_result.time_seconds
        )
        
        # Phase 2: Train model
        training_result = self._train_model(iteration)
        
        self.logger.log_training_stats(
            epochs=self.config.training.num_epochs,
            train_loss=training_result['train_loss'],
            train_policy_loss=training_result['train_policy_loss'],
            train_value_loss=training_result['train_value_loss'],
            val_loss=training_result['val_loss'],
            val_policy_loss=training_result['val_policy_loss'],
            val_value_loss=training_result['val_value_loss'],
            time_seconds=training_result['time']
        )
        
        # Phase 3: Evaluate model (skip for iteration 0)
        if iteration == 0:
            # First model - automatically promote
            self._promote_model(iteration, training_result['checkpoint_path'])
            self.logger.log_model_promotion(True, training_result['checkpoint_path'])
        else:
            # Evaluate against current best
            arena_result = self._evaluate_model(iteration, training_result['checkpoint_path'])
            
            self.logger.log_arena_stats(
                arena_result.total_games,
                arena_result.player1_wins,
                arena_result.player2_wins,
                arena_result.draws,
                arena_result.player1_win_rate,
                arena_result.is_significant,
                arena_result.time_seconds
            )
            
            # Decide whether to promote
            should_promote = self._should_promote_model(arena_result)
            
            if should_promote:
                self._promote_model(iteration, training_result['checkpoint_path'])
                self.logger.log_model_promotion(True, training_result['checkpoint_path'])
            else:
                # Archive rejected model
                self.model_manager.archive_model(
                    training_result['checkpoint_path'],
                    iteration,
                    label="rejected"
                )
                self.logger.log_model_promotion(False)
        
        # Phase 4: Cleanup old data if needed
        if iteration >= self.config.data_window_size:
            self.data_manager.cleanup_old_data(iteration)
        
        # Finish iteration
        iteration_time = time.time() - iteration_start
        self.logger.finish_iteration(iteration_time)
    
    def _generate_selfplay_data(self, iteration: int):
        """Generate self-play data for this iteration"""
        if self.config.verbose:
            print(f"\n--- Phase 1: Self-play Data Generation ---")
        
        # Get model to use for self-play
        if iteration == 0 and self.config.initial_model_path:
            model_path = Path(self.config.initial_model_path)
        else:
            model_path = self.model_manager.get_best_model()
            if model_path is None:
                raise RuntimeError("No model available for self-play!")
        

        if self.config.verbose:
            print(f"Using model: {model_path}")
        
        # Output path
        output_path = self.config.get_data_path(iteration)
        
        # Run self-play
        result = self.selfplay_runner.run(
            model_path=model_path,
            output_path=output_path,
            num_games=self.config.selfplay.num_games,
            simulations_per_move=self.config.selfplay.simulations_per_move,
            batch_size=self.config.selfplay.batch_size,
            temperature_threshold=self.config.selfplay.temperature_threshold_move,
            exploration_temp=self.config.selfplay.exploration_temperature,
            exploitation_temp=self.config.selfplay.exploitation_temperature,
            c_puct=self.config.selfplay.c_puct,
            use_dirichlet=self.config.selfplay.use_dirichlet,
            dirichlet_alpha=self.config.selfplay.dirichlet_alpha,
            dirichlet_epsilon=self.config.selfplay.dirichlet_epsilon,
            use_async=self.config.selfplay.use_async,
            chunk_save=self.config.selfplay.chunk_save
        )
        
        if result.success:
            # Register data with manager
            self.data_manager.add_iteration_data(result.output_path, iteration)
        
        return result
    
    def _train_model(self, iteration: int):
        """Train a new model"""
        if self.config.verbose:
            print(f"\n--- Phase 2: Model Training ---")
        
        # Get training data (current + recent iterations)
        training_data = self.data_manager.get_training_data(iteration)
        
        if not training_data:
            raise RuntimeError("No training data available!")
        
        # Setup training config
        train_config = TrainConfig()
        train_config.data_paths = [str(p) for p in training_data]
        train_config.batch_size = self.config.training.batch_size
        train_config.num_epochs = self.config.training.num_epochs
        train_config.learning_rate = self.config.training.learning_rate
        train_config.weight_decay = self.config.training.weight_decay
        train_config.train_split = self.config.training.train_split
        train_config.use_augmentation = self.config.training.use_augmentation
        train_config.use_mix_z_q = self.config.training.use_mix_z_q
        train_config.device = self.config.training.device
        train_config.num_workers = self.config.training.num_workers
        train_config.model_size = self.config.training.model_size
        train_config.checkpoint_dir = str(self.config.checkpoints_dir)
        train_config.verbose = self.config.verbose
        
        # Use previous best model as starting point (fine-tuning)
        if iteration > 0:
            best_model = self.model_manager.get_best_model()
            # Use the regular checkpoint, not the scripted version
            best_checkpoint = self.config.models_dir / "best_model.pt"
            if best_checkpoint.exists():
                train_config.pretrained_path = str(best_checkpoint)
                train_config.reset_optimizer = False
        
        # Run training
        start_time = time.time()
        train(train_config)
        training_time = time.time() - start_time
        
        # Get training results
        best_checkpoint = Path(train_config.checkpoint_dir) / "best_model.pt"
        
        import torch
        checkpoint = torch.load(best_checkpoint)
        train_metrics = checkpoint['metrics']['train']
        val_metrics = checkpoint['metrics']['val']
        
        return {
            'checkpoint_path': best_checkpoint,
            'train_loss': train_metrics['loss'],
            'train_policy_loss': train_metrics['policy_loss'],
            'train_value_loss': train_metrics['value_loss'],
            'val_loss': val_metrics['loss'],
            'val_policy_loss': val_metrics['policy_loss'],
            'val_value_loss': val_metrics['value_loss'],
            'time': training_time
        }
    
    def _evaluate_model(self, iteration: int, challenger_path: Path):
        """Evaluate new model against current best"""
        if self.config.verbose:
            print(f"\n--- Phase 3: Model Evaluation ---")
        
        base_model = self.model_manager.get_best_model()
        if base_model is None:
            raise RuntimeError("No base model to evaluate against!")
        
        # Prepare challenger model (export to TorchScript if needed)
        challenger_scripted = self.config.checkpoints_dir / "best_model_scripted.pt"
        if not challenger_scripted.exists():
            if self.config.verbose:
                print("TorchScript model not found, using checkpoint...")
            # In this case, we'd need to ensure the arena can load PyTorch checkpoints
            # or we export it here
        
        if self.config.verbose:
            print(f"Base model: {base_model}")
            print(f"Challenger: {challenger_scripted}")
        
        # Run arena
        output_csv = self.config.get_arena_results_path(iteration)
        
        result = self.arena_runner.run(
            base_model_path=challenger_scripted,  # Player 1 = challenger
            challenger_model_path=base_model,      # Player 2 = base
            num_games=self.config.arena.num_games,
            simulations_per_move=self.config.arena.simulations_per_move,
            batch_size=self.config.arena.batch_size,
            temperature=self.config.arena.temperature,
            alternate_colors=self.config.arena.alternate_colors,
            c_puct=self.config.arena.c_puct,
            use_dirichlet=self.config.arena.use_dirichlet,
            dirichlet_alpha=self.config.arena.dirichlet_alpha,
            dirichlet_epsilon=self.config.arena.dirichlet_epsilon,
            use_async=self.config.arena.use_async,
            output_csv=output_csv
        )
        
        if self.config.verbose:
            print(result)
        
        return result
    
    def _should_promote_model(self, arena_result) -> bool:
        """Decide whether to promote the challenger model"""
        # Check win rate threshold
        meets_threshold = arena_result.player1_win_rate >= self.config.promotion_threshold
        
        # Check statistical significance if required
        is_significant = arena_result.is_significant
        
        if self.config.require_significance:
            should_promote = meets_threshold and is_significant
        else:
            should_promote = meets_threshold
        
        if self.config.verbose:
            print(f"\nPromotion Decision:")
            print(f"  Win rate: {arena_result.player1_win_rate:.1%} (threshold: {self.config.promotion_threshold:.1%})")
            print(f"  Meets threshold: {meets_threshold}")
            print(f"  Statistically significant: {is_significant}")
            print(f"  Decision: {'PROMOTE' if should_promote else 'REJECT'}")
        
        return should_promote
    
    def _promote_model(self, iteration: int, checkpoint_path: Path):
        """Promote a model to become the new best"""
        self.model_manager.promote_model(
            checkpoint_path,
            iteration,
            archive_old=True
        )


def main():
    """Main entry point"""
    # Create default config
    config = PipelineConfig()
    
    # You can customize config here or load from file
    # config = PipelineConfig.load(Path("pipeline_config.json"))
    
    # Example customization
    config.num_iterations = 50
    config.selfplay.num_games = 500
    config.training.num_epochs = 10
    config.arena.num_games = 100
    config.promotion_threshold = 0.55
    
    # Save config for reference
    config.save(config.logs_dir / "pipeline_config.json")
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()