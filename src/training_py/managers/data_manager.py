"""
Data management for the training pipeline
"""
from pathlib import Path
from typing import List, Optional
import os


class DataManager:
    """Manages self-play data across iterations"""
    
    def __init__(self, data_dir: Path, window_size: int = 5, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.verbose = verbose
        
        # Create directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track data files
        self.data_files: List[Path] = []
        self._scan_existing_data()
    
    def _scan_existing_data(self):
        """Scan directory for existing data files"""
        if not self.data_dir.exists():
            return
        
        # Look for iteration_*.bin files
        self.data_files = sorted(self.data_dir.glob("iteration_*.bin"))
        
        if self.verbose and self.data_files:
            print(f"Found {len(self.data_files)} existing data files")
    
    def add_iteration_data(self, data_path: Path, iteration: int):
        """
        Register a new data file for an iteration
        
        Args:
            data_path: Path to the data file
            iteration: Iteration number
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Expected path for the iteration
        expected_path = self.data_dir / f"iteration_{iteration}.bin"
        
        # If it's not already at the expected location, move/copy it
        if data_path.resolve() != expected_path.resolve():
            data_path.rename(expected_path)
            data_path = expected_path
        
        # Add to tracking
        if data_path not in self.data_files:
            self.data_files.append(data_path)
            self.data_files.sort()
        
        if self.verbose:
            file_size = data_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  Registered data file: {data_path.name} ({file_size:.1f} MB)")
    
    def get_training_data(self, current_iteration: int) -> List[Path]:
        """
        Get list of data files to use for training
        Uses a sliding window of recent iterations
        
        Args:
            current_iteration: Current iteration number
        
        Returns:
            List of data file paths
        """
        # Get data files within the window
        start_iteration = max(0, current_iteration - self.window_size + 1)
        
        training_files = []
        for i in range(start_iteration, current_iteration + 1):
            data_path = self.data_dir / f"iteration_{i}.bin"
            if data_path.exists():
                training_files.append(data_path)
        
        if self.verbose:
            print(f"  Using {len(training_files)} data files for training:")
            for f in training_files:
                print(f"    - {f.name}")
        
        return training_files
    
    def cleanup_old_data(self, current_iteration: int, keep_last_n: Optional[int] = None):
        """
        Remove old data files outside the window
        
        Args:
            current_iteration: Current iteration number
            keep_last_n: Override window_size, keep last N iterations
        """
        keep_n = keep_last_n if keep_last_n is not None else self.window_size
        
        # Determine cutoff iteration
        cutoff_iteration = max(0, current_iteration - keep_n)
        
        deleted_count = 0
        deleted_size = 0
        
        for data_file in list(self.data_files):
            # Extract iteration number from filename
            try:
                # Format: iteration_N.bin
                iter_num = int(data_file.stem.split('_')[1])
                
                if iter_num < cutoff_iteration:
                    file_size = data_file.stat().st_size
                    data_file.unlink()
                    self.data_files.remove(data_file)
                    deleted_count += 1
                    deleted_size += file_size
                    
            except (ValueError, IndexError):
                # Couldn't parse iteration number, skip
                continue
        
        if self.verbose and deleted_count > 0:
            size_mb = deleted_size / (1024 * 1024)
            print(f"  Cleaned up {deleted_count} old data files ({size_mb:.1f} MB)")
    
    def get_data_statistics(self) -> dict:
        """Get statistics about stored data"""
        total_size = 0
        file_count = len(self.data_files)
        
        for data_file in self.data_files:
            if data_file.exists():
                total_size += data_file.stat().st_size
        
        return {
            'num_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'average_size_mb': (total_size / file_count / (1024 * 1024)) if file_count > 0 else 0,
            'files': [f.name for f in self.data_files]
        }
    
    def verify_data_integrity(self) -> List[Path]:
        """
        Verify all tracked data files exist
        Returns list of missing files
        """
        missing_files = []
        
        for data_file in self.data_files:
            if not data_file.exists():
                missing_files.append(data_file)
        
        if missing_files and self.verbose:
            print(f"Warning: {len(missing_files)} data files are missing:")
            for f in missing_files:
                print(f"  - {f}")
        
        return missing_files
    
    def get_latest_iteration(self) -> int:
        """Get the latest iteration number from existing data files"""
        if not self.data_files:
            return -1
        
        max_iter = -1
        for data_file in self.data_files:
            try:
                iter_num = int(data_file.stem.split('_')[1])
                max_iter = max(max_iter, iter_num)
            except (ValueError, IndexError):
                continue
        
        return max_iter


from typing import Optional