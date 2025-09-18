#!/usr/bin/env python3
"""
Hrudhi Compressed Installer - FIXED VERSION
Downloads and installs dependencies, then runs the application
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
        self.root.title("Hrudhi AI Agent - Setup")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # App data directory
        self.app_dir = Path.home() / "AppData" / "Local" / "Hrudhi"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_installing = False
        self.setup_ui()
        
        # Check installation status after UI loads
        self.root.after(500, self.check_installation)
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#6366F1", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🤖 Hrudhi AI Agent", 
                              font=("Arial", 18, "bold"), fg="white", bg="#6366F1")
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="Your Personal AI Note Assistant", 
                                 font=("Arial", 10), fg="#E0E7FF", bg="#6366F1")
        subtitle_label.pack()
        
        # Main content
        content_frame = tk.Frame(self.root, padx=40, pady=30)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = tk.Label(content_frame, text="Initializing...", 
                                   font=("Arial", 11), wraplength=400)
        self.status_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(content_frame, mode='indeterminate')
        self.progress.pack(pady=(0, 20), fill=tk.X)
        
        # Buttons frame
        button_frame = tk.Frame(content_frame)
        button_frame.pack(pady=10)
        
        # Install button - FIXED with proper event binding
        self.install_btn = tk.Button(button_frame, text="📦 Install Dependencies", 
                                   font=("Arial", 10, "bold"),
                                   bg="#10B981", fg="white", 
                                   padx=20, pady=8,
                                   cursor="hand2",
                                   relief="raised",
                                   borderwidth=2)
        self.install_btn.bind("<Button-1>", self.on_install_click)
        self.install_btn.bind("<Return>", self.on_install_click)
        self.install_btn.pack(pady=5)
        
        # Launch button
        self.launch_btn = tk.Button(button_frame, text="🚀 Launch Hrudhi", 
                                  font=("Arial", 10, "bold"),
                                  bg="#6366F1", fg="white",
                                  padx=20, pady=8,
                                  cursor="hand2",
                                  relief="raised",
                                  borderwidth=2,
                                  state=tk.DISABLED)
        self.launch_btn.bind("<Button-1>", self.on_launch_click)
        self.launch_btn.bind("<Return>", self.on_launch_click)
        self.launch_btn.pack(pady=5)
        
        # Info label
        info_label = tk.Label(content_frame, 
                             text="First run requires internet connection\nfor AI model downloads (~500MB)",
                             font=("Arial", 9), fg="#6B7280", justify=tk.CENTER)
        info_label.pack(pady=(20, 0))
        
        # Make install button focusable for keyboard navigation
        self.install_btn.focus_set()
    
    def check_installation(self):
        """Check if dependencies are already installed"""
        config_file = self.app_dir / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if config.get('installed', False):
                    self.status_label.config(text="✅ Hrudhi is ready to use!", fg="#10B981")
                    self.install_btn.config(state=tk.DISABLED, text="✅ Already Installed")
                    self.launch_btn.config(state=tk.NORMAL)
                    return
            except:
                pass
        
        self.status_label.config(text="⚙️ First time setup required", fg="#F59E0B")
        self.install_btn.config(state=tk.NORMAL)
    
    def on_install_click(self, event=None):
        """Handle install button click - FIXED VERSION"""
        if self.is_installing:
            messagebox.showinfo("Already Installing", "Installation is already in progress...")
            return
        
        # Immediate feedback to show button was clicked
        self.install_btn.config(relief="sunken")
        self.root.update()
        
        # Show confirmation
        result = messagebox.askyesno(
            "Install Dependencies", 
            "This will install Python packages needed for AI functionality.\n\n"
            "Requirements:\n"
            "• Internet connection\n"
            "• ~500MB download\n"
            "• 3-5 minutes\n\n"
            "Continue?"
        )
        
        if result:
            self.start_installation()
        else:
            self.install_btn.config(relief="raised")
    
    def start_installation(self):
        """Start installation in background thread"""
        self.is_installing = True
        self.install_btn.config(state=tk.DISABLED, text="Installing...", relief="sunken")
        self.progress.start()
        self.status_label.config(text="� Starting installation...", fg="#2563EB")
        
        # Use threading to prevent UI freeze
        install_thread = threading.Thread(target=self.install_worker, daemon=True)
        install_thread.start()
    
    def install_worker(self):
        """Background installation worker"""
        packages = [
            "sentence-transformers",
            "transformers", 
            "torch",
            "sumy",
            "nltk",
            "beautifulsoup4",
            "requests",
            "scikit-learn",
            "numpy"
        ]
        
        try:
            for i, package in enumerate(packages, 1):
                # Update UI from main thread
                self.root.after(0, lambda p=package, n=i, t=len(packages): 
                    self.status_label.config(text=f"📦 Installing {p} ({n}/{t})"))
                
                # Install with better error handling
                cmd = [sys.executable, "-m", "pip", "install", package, "--user", "--quiet", "--no-warn-script-location"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    # Try without --user flag
                    cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        raise Exception(f"Failed to install {package}: {result.stderr}")
            
            # Save config
            config = {
                "installed": True,
                "version": "1.0.0",
                "install_date": str(datetime.datetime.now()),
                "installer_type": "compressed"
            }
            
            with open(self.app_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Success - update UI on main thread
            self.root.after(0, self.installation_success)
            
        except Exception as e:
            # Error - update UI on main thread  
            self.root.after(0, lambda err=str(e): self.installation_error(err))
    
    def installation_success(self):
        """Handle successful installation"""
        self.is_installing = False
        self.progress.stop()
        self.status_label.config(text="🎉 Installation completed successfully!", fg="#10B981")
        self.install_btn.config(text="✅ Installed", state=tk.DISABLED, relief="raised")
        self.launch_btn.config(state=tk.NORMAL)
        
        messagebox.showinfo(
            "Success", 
            "Hrudhi is ready to use!\n\n"
            "Your AI companion can now help with:\n"
            "• Smart note-taking\n"
            "• Intelligent conversations\n" 
            "• Text summarization\n"
            "• Web content learning"
        )
    
    def installation_error(self, error):
        """Handle installation error"""
        self.is_installing = False  
        self.progress.stop()
        self.status_label.config(text="❌ Installation failed", fg="#EF4444")
        self.install_btn.config(state=tk.NORMAL, text="🔄 Retry Installation", relief="raised")
        
        messagebox.showerror(
            "Installation Failed",
            f"Installation failed: {error}\n\n"
            "Try:\n"
            "• Check internet connection\n"
            "• Run as Administrator\n"
            "• Temporarily disable antivirus"
        )
    
    def on_launch_click(self, event=None):
        """Handle launch button click"""
        self.launch_btn.config(relief="sunken")
        self.root.update()
        
        try:
            # Look for main script
            if os.path.exists("hrudhi_main.py"):
                os.system(f'start "" "{sys.executable}" "hrudhi_main.py"')
            elif os.path.exists("hrudhi/hrudhi.py"):
                os.system(f'start "" "{sys.executable}" "hrudhi/hrudhi.py"')
            else:
                raise FileNotFoundError("Could not find main Hrudhi application")
            
            # Close installer
            self.root.after(1000, self.root.destroy)
            
        except Exception as e:
            self.launch_btn.config(relief="raised")
            messagebox.showerror("Launch Error", f"Failed to launch Hrudhi:\n{e}")

def main():
    """Main entry point with error handling"""
    try:
        print("Starting Hrudhi Installer...")
        app = HrudhiInstaller()
        app.root.mainloop()
    except Exception as e:
        print(f"Installer error: {e}")
        try:
            messagebox.showerror("Installer Error", f"Failed to start installer:\n{e}")
        except:
            print("Could not show error dialog")

if __name__ == "__main__":
    main()
