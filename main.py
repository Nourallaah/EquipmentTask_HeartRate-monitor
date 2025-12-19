import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import HeartMonitorGUI

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = HeartMonitorGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()