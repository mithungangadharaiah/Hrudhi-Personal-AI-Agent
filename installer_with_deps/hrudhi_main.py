
import sys
import os

# Add current directory to path so we can import hrudhi
sys.path.insert(0, os.path.dirname(__file__))

from hrudhi.hrudhi import HrudhiApp
import tkinter as tk

def run():
    root = tk.Tk()
    app = HrudhiApp(root)
    root.mainloop()

if __name__ == "__main__":
    run()
