"""
Wrapper for running C++ arena binary
"""
import subprocess
import time
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ArenaResults:
    """Results from arena evaluation"""
    total_games: int
    player1_wins: int
    player2_wins: int
    draws: int
    
    player1_win_rate: float
    player2_win_rate: float
    draw_rate: float
    
    average_game_length: float
    time_seconds: float
    
    success: bool
    error_message: Optional[str] = None
    
    @property
    def is_player1_better(self) -> bool:
        """Check if player1 is significantly better"""
        return self.player1_win_rate > 0.5
    
    @property
    def win_rate_difference(self) -> float:
        """Difference in win rates (positive = player1 better)"""
        return self.player1_win_rate - self.player2_win_rate
    
    @property
    def is_significant(self) -> bool:
        """
        Check if result is statistically significant
        Uses normal approximation for binomial proportion
        """
        if self.total_games == 0:
            return False
        
        confidence: float = 0.95
        # Standard error of win rate difference
        p1 = self.player1_win_rate
        se = math.sqrt(p1 * (1 - p1) / self.total_games)
        
        # Z-score for the difference
        z_score = abs(self.win_rate_difference) / (se + 1e-10)
        
        # Critical values for different confidence levels
        critical_values = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        critical = critical_values.get(confidence, 1.96)
        return z_score > critical
    
    def __str__(self) -> str:
        return (
            f"Arena Results:\n"
            f"  Total games: {self.total_games}\n"
            f"  Player 1: {self.player1_wins} wins ({self.player1_win_rate:.1%})\n"
            f"  Player 2: {self.player2_wins} wins ({self.player2_win_rate:.1%})\n"
            f"  Draws: {self.draws} ({self.draw_rate:.1%})\n"
            f"  Win rate diff: {self.win_rate_difference:+.1%}\n"
            f"  Significant (95%): {self.is_significant}\n"
            f"  Time: {self.time_seconds:.1f}s"
        )


class ArenaRunner:
    """Runner for C++ arena binary"""
    
    def __init__(self, binary_path: str, verbose: bool = True):
        self.binary_path = Path(binary_path)
        self.verbose = verbose
        
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Arena binary not found: {self.binary_path}")
    
    def run(
        self,
        base_model_path: Path,
        challenger_model_path: Path,
        num_games: int,
        simulations_per_move: int,
        batch_size: int = 32,
        temperature: float = 0.1,
        alternate_colors: bool = True,
        c_puct: float = 1.5,
        use_dirichlet: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_async: bool = True,
        output_csv: Optional[Path] = None,
        timeout: Optional[int] = None
    ) -> ArenaResults:
        """
        Run arena evaluation between two models
        
        Args:
            base_model_path: Path to base model (player 1)
            challenger_model_path: Path to challenger model (player 2)
            num_games: Number of games to play
            simulations_per_move: MCTS simulations per move
            batch_size: Batch size for neural network evaluation
            temperature: Temperature for move selection
            alternate_colors: Whether to alternate who plays first
            c_puct: PUCT exploration constant
            use_dirichlet: Whether to add Dirichlet noise (usually False for eval)
            dirichlet_alpha: Dirichlet alpha parameter
            dirichlet_epsilon: Dirichlet epsilon
            use_async: Use async MCTS search
            output_csv: Optional path to save detailed results
            timeout: Maximum time in seconds (None = no limit)
        
        Returns:
            ArenaResults with statistics
        """
        # Create temporary output file if not specified
        if output_csv is None:
            output_csv = Path("temp_arena_results.csv")
        
        # Build command
        cmd = [
            str(self.binary_path),
            str(base_model_path),
            str(challenger_model_path),
            "--games", str(num_games),
            "--simulations", str(simulations_per_move),
            "--batch-size", str(batch_size),
            "--temp", str(temperature),
            "--output-csv", str(output_csv)
        ]
        
        if not use_async:
            cmd.append("--no-async")

        if self.verbose:
            print(f"Running arena: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(result.stdout)
            
            # Parse results from output
            stats = self._parse_output(result.stdout, num_games)
            stats['time_seconds'] = elapsed
            stats['success'] = True
            
            # Try to read CSV for more detailed stats if available
            if output_csv.exists():
                csv_stats = self._parse_csv(output_csv)
                stats.update(csv_stats)
            
            return ArenaResults(**stats)
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            error_msg = f"Arena evaluation timed out after {elapsed:.1f}s"
            if self.verbose:
                print(f"ERROR: {error_msg}")
            
            return ArenaResults(
                total_games=0,
                player1_wins=0,
                player2_wins=0,
                draws=0,
                player1_win_rate=0.0,
                player2_win_rate=0.0,
                draw_rate=0.0,
                average_game_length=0.0,
                time_seconds=elapsed,
                success=False,
                error_message=error_msg
            )
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            error_msg = f"Arena failed with exit code {e.returncode}"
            if self.verbose:
                print(f"ERROR: {error_msg}")
                print(f"STDERR: {e.stderr}")
            
            return ArenaResults(
                total_games=0,
                player1_wins=0,
                player2_wins=0,
                draws=0,
                player1_win_rate=0.0,
                player2_win_rate=0.0,
                draw_rate=0.0,
                average_game_length=0.0,
                time_seconds=elapsed,
                success=False,
                error_message=f"{error_msg}\n{e.stderr}"
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            if self.verbose:
                print(f"ERROR: {error_msg}")
            
            return ArenaResults(
                total_games=0,
                player1_wins=0,
                player2_wins=0,
                draws=0,
                player1_win_rate=0.0,
                player2_win_rate=0.0,
                draw_rate=0.0,
                average_game_length=0.0,
                time_seconds=elapsed,
                success=False,
                error_message=error_msg
            )
    
    def _parse_output(self, output: str, expected_games: int) -> dict:
        """Parse arena output to extract statistics"""
        stats = {
            'total_games': expected_games,
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'player1_win_rate': 0.0,
            'player2_win_rate': 0.0,
            'draw_rate': 0.0,
            'average_game_length': 0.0
        }
        
        for line in output.split('\n'):
            line = line.strip()
            
            # Look for patterns in C++ output
            if 'Total games:' in line:
                try:
                    stats['total_games'] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            
            elif 'Player 1 wins:' in line or 'Player 1:' in line:
                try:
                    # Extract number before parenthesis
                    parts = line.split(':')[1].strip().split('(')
                    stats['player1_wins'] = int(parts[0].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            elif 'Player 2 wins:' in line or 'Player 2:' in line:
                try:
                    parts = line.split(':')[1].strip().split('(')
                    stats['player2_wins'] = int(parts[0].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            elif 'Draws:' in line:
                try:
                    parts = line.split(':')[1].strip().split('(')
                    stats['draws'] = int(parts[0].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            elif 'Avg game length:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    stats['average_game_length'] = float(parts[0])
                except (ValueError, IndexError):
                    pass
        
        # Calculate rates
        if stats['total_games'] > 0:
            stats['player1_win_rate'] = stats['player1_wins'] / stats['total_games']
            stats['player2_win_rate'] = stats['player2_wins'] / stats['total_games']
            stats['draw_rate'] = stats['draws'] / stats['total_games']
        
        return stats
    
    def _parse_csv(self, csv_path: Path) -> dict:
        """Parse arena CSV output for additional statistics"""
        stats = {}
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    # Count outcomes
                    p1_wins = sum(1 for r in rows if int(r.get('outcome', 0)) > 0)
                    p2_wins = sum(1 for r in rows if int(r.get('outcome', 0)) < 0)
                    draws = sum(1 for r in rows if int(r.get('outcome', 0)) == 0)
                    
                    # Average game length
                    game_lengths = [int(r.get('num_moves', 0)) for r in rows]
                    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
                    
                    stats['player1_wins'] = p1_wins
                    stats['player2_wins'] = p2_wins
                    stats['draws'] = draws
                    stats['total_games'] = len(rows)
                    stats['average_game_length'] = avg_length
                    
                    if len(rows) > 0:
                        stats['player1_win_rate'] = p1_wins / len(rows)
                        stats['player2_win_rate'] = p2_wins / len(rows)
                        stats['draw_rate'] = draws / len(rows)
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not parse CSV: {e}")
        
        return stats