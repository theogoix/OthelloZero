"""
Logging utilities for the training pipeline
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class IterationLog:
    """Log entry for a single iteration"""
    iteration: int
    timestamp: str
    
    # Self-play stats
    selfplay_games: int = 0
    selfplay_positions: int = 0
    selfplay_time: float = 0.0
    
    # Training stats
    training_epochs: int = 0
    train_loss: float = 0.0
    train_policy_loss: float = 0.0
    train_value_loss: float = 0.0
    val_loss: float = 0.0
    val_policy_loss: float = 0.0
    val_value_loss: float = 0.0
    training_time: float = 0.0
    
    # Arena stats
    arena_games: int = 0
    arena_player1_wins: int = 0
    arena_player2_wins: int = 0
    arena_draws: int = 0
    arena_win_rate: float = 0.0
    arena_significant: bool = False
    arena_time: float = 0.0
    
    # Model promotion
    model_promoted: bool = False
    promoted_model_path: Optional[str] = None
    
    # Total iteration time
    total_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class TrainingLogger:
    """Logger for the entire training pipeline"""
    
    def __init__(self, logs_dir: Path, verbose: bool = True):
        self.logs_dir = Path(logs_dir)
        self.verbose = verbose
        
        # Create log directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "selfplay").mkdir(exist_ok=True)
        (self.logs_dir / "training").mkdir(exist_ok=True)
        (self.logs_dir / "arena").mkdir(exist_ok=True)
        
        # Main log file
        self.log_file = self.logs_dir / "pipeline_log.json"
        
        # Load existing logs or create new
        self.iterations: List[IterationLog] = []
        self.metadata: Dict[str, Any] = {}
        self._load_logs()
        
        # Current iteration being tracked
        self.current_iteration: Optional[IterationLog] = None
    
    def _load_logs(self):
        """Load existing logs from file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', {})
                    
                    # Reconstruct iteration logs
                    for iter_dict in data.get('iterations', []):
                        self.iterations.append(IterationLog(**iter_dict))
                
                if self.verbose:
                    print(f"Loaded {len(self.iterations)} iteration logs")
            
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load logs: {e}")
                self.iterations = []
                self.metadata = {}
    
    def _save_logs(self):
        """Save logs to file"""
        data = {
            'metadata': self.metadata,
            'iterations': [iter_log.to_dict() for iter_log in self.iterations]
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_metadata(self, key: str, value: Any):
        """Set a metadata field"""
        self.metadata[key] = value
        self._save_logs()
    
    def start_iteration(self, iteration: int):
        """Start logging a new iteration"""
        self.current_iteration = IterationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat()
        )
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Iteration {iteration}")
            print(f"{'='*70}")
    
    def log_selfplay_stats(
        self,
        games: int,
        positions: int,
        time_seconds: float
    ):
        """Log self-play generation statistics"""
        if self.current_iteration is None:
            return
        
        self.current_iteration.selfplay_games = games
        self.current_iteration.selfplay_positions = positions
        self.current_iteration.selfplay_time = time_seconds
        
        if self.verbose:
            print(f"\nSelf-play Statistics:")
            print(f"  Games: {games}")
            print(f"  Positions: {positions}")
            print(f"  Time: {time_seconds:.1f}s")
            print(f"  Games/second: {games/time_seconds:.2f}")
    
    def log_training_stats(
        self,
        epochs: int,
        train_loss: float,
        train_policy_loss: float,
        train_value_loss: float,
        val_loss: float,
        val_policy_loss: float,
        val_value_loss: float,
        time_seconds: float
    ):
        """Log training statistics"""
        if self.current_iteration is None:
            return
        
        self.current_iteration.training_epochs = epochs
        self.current_iteration.train_loss = train_loss
        self.current_iteration.train_policy_loss = train_policy_loss
        self.current_iteration.train_value_loss = train_value_loss
        self.current_iteration.val_loss = val_loss
        self.current_iteration.val_policy_loss = val_policy_loss
        self.current_iteration.val_value_loss = val_value_loss
        self.current_iteration.training_time = time_seconds
        
        if self.verbose:
            print(f"\nTraining Statistics:")
            print(f"  Epochs: {epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f})")
            print(f"  Time: {time_seconds:.1f}s")
    
    def log_arena_stats(
        self,
        games: int,
        player1_wins: int,
        player2_wins: int,
        draws: int,
        win_rate: float,
        significant: bool,
        time_seconds: float
    ):
        """Log arena evaluation statistics"""
        if self.current_iteration is None:
            return
        
        self.current_iteration.arena_games = games
        self.current_iteration.arena_player1_wins = player1_wins
        self.current_iteration.arena_player2_wins = player2_wins
        self.current_iteration.arena_draws = draws
        self.current_iteration.arena_win_rate = win_rate
        self.current_iteration.arena_significant = significant
        self.current_iteration.arena_time = time_seconds
        
        if self.verbose:
            print(f"\nArena Statistics:")
            print(f"  Games: {games}")
            print(f"  Challenger Wins: {player1_wins} ({win_rate:.1%})")
            print(f"  Base Wins: {player2_wins} ({(1-win_rate-draws/games):.1%})")
            print(f"  Draws: {draws} ({draws/games:.1%})")
            print(f"  Statistically Significant: {significant}")
            print(f"  Time: {time_seconds:.1f}s")
    
    def log_model_promotion(self, promoted: bool, model_path: Optional[str] = None):
        """Log whether model was promoted"""
        if self.current_iteration is None:
            return
        
        self.current_iteration.model_promoted = promoted
        self.current_iteration.promoted_model_path = str(model_path) if model_path else None
        
        if self.verbose:
            if promoted:
                print(f"\n✓ Model PROMOTED")
            else:
                print(f"\n✗ Model REJECTED")
    
    def finish_iteration(self, total_time: float):
        """Finish logging current iteration"""
        if self.current_iteration is None:
            return
        
        self.current_iteration.total_time = total_time
        
        # Add to iterations list
        self.iterations.append(self.current_iteration)
        
        # Save to file
        self._save_logs()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Iteration {self.current_iteration.iteration} Complete")
            print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"{'='*70}\n")
        
        self.current_iteration = None
    
    def get_iteration_summary(self, iteration: int) -> Optional[IterationLog]:
        """Get summary for a specific iteration"""
        for iter_log in self.iterations:
            if iter_log.iteration == iteration:
                return iter_log
        return None
    
    def get_all_iterations(self) -> List[IterationLog]:
        """Get all iteration logs"""
        return self.iterations
    
    def print_summary(self):
        """Print summary of all iterations"""
        if not self.iterations:
            print("No iterations logged yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"Training Pipeline Summary")
        print(f"{'='*70}")
        print(f"Total Iterations: {len(self.iterations)}")
        
        # Count promotions
        promotions = sum(1 for it in self.iterations if it.model_promoted)
        print(f"Models Promoted: {promotions}")
        
        # Total games/positions
        total_games = sum(it.selfplay_games for it in self.iterations)
        total_positions = sum(it.selfplay_positions for it in self.iterations)
        print(f"Total Self-play Games: {total_games:,}")
        print(f"Total Training Positions: {total_positions:,}")
        
        # Total time
        total_time = sum(it.total_time for it in self.iterations)
        print(f"Total Time: {total_time/3600:.2f} hours")
        
        # Latest iteration stats
        if self.iterations:
            latest = self.iterations[-1]
            print(f"\nLatest Iteration ({latest.iteration}):")
            print(f"  Val Loss: {latest.val_loss:.4f}")
            if latest.arena_games > 0:
                print(f"  Arena Win Rate: {latest.arena_win_rate:.1%}")
                print(f"  Promoted: {latest.model_promoted}")
        
        print(f"{'='*70}\n")
    
    def export_csv(self, output_path: Path):
        """Export logs to CSV for analysis"""
        import csv
        
        if not self.iterations:
            return
        
        with open(output_path, 'w', newline='') as f:
            # Get all field names
            fieldnames = list(self.iterations[0].to_dict().keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for iter_log in self.iterations:
                writer.writerow(iter_log.to_dict())
        
        if self.verbose:
            print(f"Exported logs to: {output_path}")