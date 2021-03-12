"""main.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Description**: Bachelorthesis - Prototype for FER

**Institution**: Johannes Kepler University Linz - Institute of Computational Perception

This file handles starts the FER application by creating an instance of the GUI class and call the run() method.
"""

from Application.UserInterface import GUI

if __name__ == '__main__':
    app = GUI()
    app.run()
