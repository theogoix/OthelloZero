"""
Wrapper for running C++ self-play binary
"""
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SelfPlayResults:
    """Results from self-play generation"""
    num_games: int
    num_positions: int
    time_seconds: float
    output_path: Path
    success: bool
    error_message: Optional[str] = None
    
    @property
    def games_per_second(self) -> float:
        return self.num_games / self.time_seconds if self.time_seconds > 0 else 0
    
    @property
    def positions_per_game(self) -> float:
        return self.num_positions / self.num_games if self.num_games > 0 else 0


class SelfPlayRunner:
    """Runner for C++ self-play binary"""
    
    def __init__(self, binary_path: str, verbose: bool = True):
        self.binary_path = Path(binary_path)
        self.verbose = verbose
        
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Self-play binary not found: {self.binary_path}")
    
    def run(
        self,
        model_path: Path,
        output_path: Path,
        num_games: int,
        simulations_per_move: int,
        batch_size: int = 32,
        temperature_threshold: int = 30,
        exploration_temp: float = 1.0,
        exploitation_temp: float = 0.1,
        c_puct: float = 1.5,
        use_dirichlet: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_async: bool = True,
        chunk_save: int = 50,
        timeout: Optional[int] = None
    ) -> SelfPlayResults:
        """
        Run self-play data generation
        
        Args:
            model_path: Path to TorchScript model
            output_path: Where to save generated data
            num_games: Number of games to play
            simulations_per_move: MCTS simulations per move
            batch_size: Batch size for neural network evaluation
            temperature_threshold: Move number to switch temperature
            exploration_temp: Temperature for early moves
            exploitation_temp: Temperature for late moves
            c_puct: PUCT exploration constant
            use_dirichlet: Whether to add Dirichlet noise to root
            dirichlet_alpha: Dirichlet alpha parameter
            dirichlet_epsilon: Dirichlet epsilon (mixing weight)
            use_async: Use async MCTS search
            chunk_save: Save data every N games
            timeout: Maximum time in seconds (None = no limit)
        
        Returns:
            SelfPlayResults with statistics
        """
        # Build command
        cmd = [
            str(self.binary_path),
            str(model_path),
            str(output_path),
            "--games", str(num_games),
            "--simulations", str(simulations_per_move),
            "--temp-threshold", str(temperature_threshold),
        ]
        
        if not use_async:
            cmd.append("--no-async")


        if self.verbose:
            print(f"Running self-play: {' '.join(cmd)}")
        
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
            
            # Parse output to extract statistics
            # Assuming C++ prints lines like:
            # "Total games: 1000"
            # "Total positions: 64000"
            num_games_actual = num_games
            num_positions = 0
            
            for line in result.stdout.split('\n'):
                if 'Total games:' in line:
                    try:
                        num_games_actual = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Total positions:' in line:
                    try:
                        num_positions = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
            
            if self.verbose:
                print(result.stdout)
            
            return SelfPlayResults(
                num_games=num_games_actual,
                num_positions=num_positions,
                time_seconds=elapsed,
                output_path=output_path,
                success=True
            )
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            error_msg = f"Self-play timed out after {elapsed:.1f}s"
            if self.verbose:
                print(f"ERROR: {error_msg}")
            
            return SelfPlayResults(
                num_games=0,
                num_positions=0,
                time_seconds=elapsed,
                output_path=output_path,
                success=False,
                error_message=error_msg
            )
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            error_msg = f"Self-play failed with exit code {e.returncode}"
            if self.verbose:
                print(f"ERROR: {error_msg}")
                print(f"STDERR: {e.stderr}")
            
            return SelfPlayResults(
                num_games=0,
                num_positions=0,
                time_seconds=elapsed,
                output_path=output_path,
                success=False,
                error_message=f"{error_msg}\n{e.stderr}"
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            if self.verbose:
                print(f"ERROR: {error_msg}")
            
            return SelfPlayResults(
                num_games=0,
                num_positions=0,
                time_seconds=elapsed,
                output_path=output_path,
                success=False,
                error_message=error_msg
            )