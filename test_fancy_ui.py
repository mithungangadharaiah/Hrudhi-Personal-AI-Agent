#!/usr/bin/env python3
"""
Test script for the fancy Hrudhi UI
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hrudhi'))

from hrudhi import HrudhiApp
import tkinter as tk

def main():
    print("🚀 Starting Hrudhi with Fancy UI...")
    
    root = tk.Tk()
    app = HrudhiApp(root)
    
    print("✨ Fancy UI loaded! Robot face should be visible.")
    print("🤖 Features:")
    print("   • Animated robotic face with moods")
    print("   • Dark theme with glowing elements") 
    print("   • Smooth animations and transitions")
    print("   • Creative visual effects")
    print("   • Real-time status updates")
    
    root.mainloop()

if __name__ == "__main__":
    main()