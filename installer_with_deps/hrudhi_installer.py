#!/usr/bin/env python3
"""
Hrudhi Bootstrap Installer
Downloads and installs dependencies on first run
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import json
import datetime
from pathlib import Path

class HrudhiInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hrudhi AI Agent - First Time Setup")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # App data directory
        self.app_dir = Path.home() / "AppData" / "Local" / "Hrudhi"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_ui()
        self.check_installation()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Hrudhi AI Agent", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Status
        self.status_label = tk.Label(self.root, text="Checking installation...")
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=50, fill=tk.X)
        
        # Install button
        self.install_btn = tk.Button(self.root, text="Install Dependencies", 
                                   command=self.start_installation, state=tk.DISABLED)
        self.install_btn.pack(pady=10)
        
        # Launch button
        self.launch_btn = tk.Button(self.root, text="Launch Hrudhi", 
                                  command=self.launch_hrudhi, state=tk.DISABLED)
        self.launch_btn.pack(pady=5)
    
    def check_installation(self):
        config_file = self.app_dir / "config.json"
        if config_file.exists():
            self.status_label.config(text="Hrudhi is ready to use!")
            self.launch_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="First time setup required")
            self.install_btn.config(state=tk.NORMAL)
    
    def start_installation(self):
        self.install_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Installing dependencies...")
        
        # Run installation in separate thread
        thread = threading.Thread(target=self.install_dependencies)
        thread.daemon = True
        thread.start()
    
    def install_dependencies(self):
        try:
            # Install required packages
            packages = [
                "sentence-transformers",
                "scikit-learn", 
                "numpy",
                "torch",
                "transformers"
            ]
            
            for i, package in enumerate(packages):
                self.root.after(0, lambda p=package: 
                    self.status_label.config(text=f"Installing {p}..."))
                
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
            
            # Create config file to mark installation complete
            config = {
                "installed": True,
                "version": "1.0.0",
                "install_date": str(datetime.datetime.now())
            }
            
            with open(self.app_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            # Update UI on main thread
            self.root.after(0, self.installation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.installation_failed(str(e)))
    
    def installation_complete(self):
        self.progress.stop()
        self.status_label.config(text="Installation completed successfully!")
        self.launch_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Hrudhi is ready to use!")
    
    def installation_failed(self, error):
        self.progress.stop()
        self.status_label.config(text="Installation failed")
        self.install_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", f"Installation failed: {error}")
    
    def launch_hrudhi(self):
        try:
            # Import and run main Hrudhi application
            self.root.destroy()
            import hrudhi_main
            hrudhi_main.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Hrudhi: {e}")

if __name__ == "__main__":
    app = HrudhiInstaller()
    app.root.mainloop()
