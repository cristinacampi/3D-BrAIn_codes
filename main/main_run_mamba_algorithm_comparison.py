"""Entry point for the complete grouped Mamba/classical benchmark.

Usage
-----
python main/main_run_mamba_algorithm_comparison.py dataset.npz output_directory
"""

import sys
from pathlib import Path


repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(repository_root / "src"))

from tools.run_mamba_comparison import main


if __name__ == "__main__":
    main()
