from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(EXAMPLE_ROOT))
from tshape_map_runner import main
if __name__ == "__main__":
    main(HERE)
