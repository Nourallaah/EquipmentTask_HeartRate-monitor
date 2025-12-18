# main.py
import sys
import os
from src.gui.main_window import main

if __name__ == "__main__":
    # Ensure the application runs from the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the GUI application
    main()