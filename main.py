"""Main entry point for PV Circularity Simulator.

Run with: streamlit run main.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the module design UI
from ui.module_design import main

if __name__ == "__main__":
    main()
