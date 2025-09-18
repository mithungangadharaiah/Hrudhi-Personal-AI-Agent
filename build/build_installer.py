#!/usr/bin/env python3
"""
Build script for Hrudhi Compressed Installer
Creates GitHub-friendly compressed installers under 25MB
Downloads dependencies on first run - much more flexible
"""

import os
import sys
import subprocess
import shutil
import zipfile
import datetime
import json
from pathlib import Path

def create_compressed_installer():
    """Create a compressed installer that's under GitHub's 100MB limit"""
    
    print("🚀 Building Compressed Hrudhi Installer...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create temporary installer directory
    temp_installer_dir = project_root / "temp_installer"
    if temp_installer_dir.exists():
        shutil.rmtree(temp_installer_dir)
    temp_installer_dir.mkdir(exist_ok=True)
    
    # Create the lightweight bootstrap installer
    bootstrap_script = '''#!/usr/bin/env python3
"""
Hrudhi Compressed Installer
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
import tempfile
import zipfile
from pathlib import Path

class HrudhiCompressedInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hrudhi AI Agent - Setup")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # App data directory
        self.app_dir = Path.home() / "AppData" / "Local" / "Hrudhi"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_ui()
        self.check_installation()
    
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
        self.status_label = tk.Label(content_frame, text="Checking installation...", 
                                   font=("Arial", 11))
        self.status_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(content_frame, mode='indeterminate', 
                                      style="TProgressbar")
        self.progress.pack(pady=(0, 20), fill=tk.X)
        
        # Buttons frame
        button_frame = tk.Frame(content_frame)
        button_frame.pack(pady=10)
        
        # Install button
        self.install_btn = tk.Button(button_frame, text="📦 Install Dependencies", 
                                   command=self.start_installation, 
                                   font=("Arial", 10, "bold"),
                                   bg="#10B981", fg="white", 
                                   padx=20, pady=8,
                                   state=tk.DISABLED)
        self.install_btn.pack(pady=5)
        
        # Launch button
        self.launch_btn = tk.Button(button_frame, text="🚀 Launch Hrudhi", 
                                  command=self.launch_hrudhi,
                                  font=("Arial", 10, "bold"),
                                  bg="#6366F1", fg="white",
                                  padx=20, pady=8,
                                  state=tk.DISABLED)
        self.launch_btn.pack(pady=5)
        
        # Info label
        info_label = tk.Label(content_frame, 
                             text="First run requires internet connection\\nfor AI model downloads (~500MB)",
                             font=("Arial", 9), fg="#6B7280", justify=tk.CENTER)
        info_label.pack(pady=(20, 0))
    
    def check_installation(self):
        config_file = self.app_dir / "config.json"
        if config_file.exists():
            self.status_label.config(text="✅ Hrudhi is ready to use!", fg="#10B981")
            self.launch_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="⚙️ First time setup required", fg="#F59E0B")
            self.install_btn.config(state=tk.NORMAL)
    
    def start_installation(self):
        self.install_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # Run installation in separate thread
        thread = threading.Thread(target=self.install_dependencies)
        thread.daemon = True
        thread.start()
    
    def install_dependencies(self):
        try:
            # Update status for each step
            steps = [
                ("🔍 Checking Python environment", None),
                ("📦 Installing sentence-transformers", "sentence-transformers"),
                ("🧮 Installing scikit-learn", "scikit-learn"),
                ("🔢 Installing numpy", "numpy"),
                ("🔥 Installing torch", "torch"),
                ("🤖 Installing transformers", "transformers"),
                ("✨ Finalizing setup", None)
            ]
            
            for i, (message, package) in enumerate(steps):
                self.root.after(0, lambda m=message: 
                    self.status_label.config(text=m))
                
                if package:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
            
            # Create config file
            config = {
                "installed": True,
                "version": "1.0.0",
                "install_date": str(datetime.datetime.now()),
                "compressed_installer": True
            }
            
            with open(self.app_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Update UI on main thread
            self.root.after(0, self.installation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.installation_failed(str(e)))
    
    def installation_complete(self):
        self.progress.stop()
        self.status_label.config(text="🎉 Installation completed successfully!", fg="#10B981")
        self.launch_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Hrudhi is ready to use!\\n\\nYour AI companion is now ready to help you with note-taking!")
    
    def installation_failed(self, error):
        self.progress.stop()
        self.status_label.config(text="❌ Installation failed", fg="#EF4444")
        self.install_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", f"Installation failed: {error}\\n\\nPlease check your internet connection and try again.")
    
    def launch_hrudhi(self):
        try:
            # Import and run main Hrudhi application
            self.root.destroy()
            import hrudhi_main
            hrudhi_main.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Hrudhi: {e}")

if __name__ == "__main__":
    app = HrudhiCompressedInstaller()
    app.root.mainloop()
'''
    
    # Create main app launcher
    main_app_code = '''
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from hrudhi.hrudhi import HrudhiApp, initialize
import tkinter as tk

def run():
    """Run the main Hrudhi application"""
    try:
        print("🧠 Initializing AI models...")
        initialize()
        root = tk.Tk()
        app = HrudhiApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"Error launching Hrudhi: {e}")
        print(traceback.format_exc())
        input("Press Enter to exit...")

if __name__ == "__main__":
    run()
'''
    
    # Write files to temp directory
    with open(temp_installer_dir / "hrudhi_installer.py", "w", encoding="utf-8") as f:
        f.write(bootstrap_script)
    
    with open(temp_installer_dir / "hrudhi_main.py", "w", encoding="utf-8") as f:
        f.write(main_app_code)
    
    # Copy hrudhi package
    hrudhi_src = project_root / "hrudhi"
    hrudhi_dst = temp_installer_dir / "hrudhi"
    if hrudhi_dst.exists():
        shutil.rmtree(hrudhi_dst)
    shutil.copytree(hrudhi_src, hrudhi_dst)
    
    # Create launcher scripts
    windows_launcher = '''@echo off
title Hrudhi AI Agent Installer
echo.
echo 🤖 Starting Hrudhi AI Agent Setup...
echo.
python hrudhi_installer.py
if errorlevel 1 (
    echo.
    echo ❌ Failed to start installer. Make sure Python 3.8+ is installed.
    echo.
    pause
)
'''
    
    linux_launcher = '''#!/bin/bash
echo "🤖 Starting Hrudhi AI Agent Setup..."
python3 hrudhi_installer.py || {
    echo "❌ Failed to start installer. Make sure Python 3.8+ is installed."
    read -p "Press Enter to exit..."
}
'''
    
    with open(temp_installer_dir / "Start_Hrudhi.bat", "w", encoding="utf-8") as f:
        f.write(windows_launcher)
    
    with open(temp_installer_dir / "start_hrudhi.sh", "w", encoding="utf-8") as f:
        f.write(linux_launcher)
    
    # Make Linux script executable
    try:
        os.chmod(temp_installer_dir / "start_hrudhi.sh", 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    # Create comprehensive README
    readme_content = '''# 🤖 Hrudhi AI Agent - Compressed Installer

## Quick Start
1. **Windows**: Double-click `Start_Hrudhi.bat`
2. **Linux/Mac**: Run `./start_hrudhi.sh` or `python3 hrudhi_installer.py`
3. Click "Install Dependencies" on first run
4. Click "Launch Hrudhi" to start your AI assistant

## What is Hrudhi?
Hrudhi is your personal AI-powered note-taking companion that:
- 🧠 **Remembers everything** you tell it with smart search
- 🔍 **Finds notes by context** not just keywords
- 💡 **Learns from your notes** to get better over time
- 🎨 **Beautiful modern interface** with a cute robot companion
- 🔒 **100% private** - all data stays on your computer

## System Requirements
- **Python 3.8+** (Download from python.org)
- **Internet connection** (first run only, ~500MB download)
- **2GB free disk space** (for AI models)
- **Windows 10+, Linux, or macOS**

## File Sizes
- **This installer**: ~5-15MB (compressed)
- **After setup**: ~2GB (includes AI models)
- **Your notes**: Minimal (just text files)

## Features
- ✅ Smart contextual note search using AI
- ✅ Clean, modern interface with robot companion
- ✅ Automatic categorization and organization
- ✅ Learning AI that improves with use
- ✅ Offline operation after initial setup
- ✅ Cross-platform compatibility
- ✅ No cloud, no tracking, no ads

## Troubleshooting

**"Python not found"**
- Download Python from python.org
- Make sure "Add to PATH" is checked during installation

**"Installation failed"**
- Check internet connection
- Try running as administrator (Windows)
- Ensure antivirus isn't blocking the installation

**"Module not found"**
- Try: `python -m pip install --upgrade pip`
- Then rerun the installer

## Privacy & Security
- All notes stored locally in `~/Desktop/HrudhiNotes/`
- No data sent to external servers
- Open source code available for review
- AI models run entirely on your computer

## Support
- GitHub Issues: Report bugs and feature requests
- Documentation: Full user guide in repository
- Community: Join discussions and share tips

Enjoy your new AI companion! 🤖✨
'''
    
    with open(temp_installer_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create the compressed ZIP file
    zip_path = project_root / "Hrudhi-AI-Agent-Installer.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"📦 Creating compressed installer: {zip_path.name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(temp_installer_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_installer_dir)
                zipf.write(file_path, arcname)
    
    # Clean up temp directory
    shutil.rmtree(temp_installer_dir)
    
    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ Compressed installer created: {zip_path.name}")
    print(f"📏 File size: {size_mb:.1f} MB")
    
    if size_mb < 25:
        print("🎉 Perfect! File is well under GitHub's 100MB limit")
        return True
    elif size_mb < 100:
        print("✅ Good! File is under GitHub's 100MB limit")
        return True
    else:
        print("⚠️  Warning: File might be too large for GitHub")
        return False

def create_installer_with_deps():
    """Create installer that downloads dependencies on first run"""
    return create_compressed_installer()
    bootstrap_script = '''#!/usr/bin/env python3
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
'''
    
    # Write bootstrap script
    with open(installer_dir / "hrudhi_installer.py", "w", encoding="utf-8") as f:
        f.write(bootstrap_script)
    
    # Copy main application
    main_app_code = '''
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
'''
    
    with open(installer_dir / "hrudhi_main.py", "w", encoding="utf-8") as f:
        f.write(main_app_code)
    
    # Copy hrudhi package
    hrudhi_src = project_root / "hrudhi"
    hrudhi_dst = installer_dir / "hrudhi"
    if hrudhi_dst.exists():
        shutil.rmtree(hrudhi_dst)
    shutil.copytree(hrudhi_src, hrudhi_dst)
    
    # Create batch file for easy launching
    batch_content = '''@echo off
echo Starting Hrudhi AI Agent...
python hrudhi_installer.py
pause
'''
    
    with open(installer_dir / "Start_Hrudhi.bat", "w", encoding="utf-8") as f:
        f.write(batch_content)
    
    # Create installer README
    readme_content = '''# Hrudhi Installer with Dependencies

## Installation
1. Ensure Python 3.8+ is installed on your system
2. Double-click `Start_Hrudhi.bat` or run `python hrudhi_installer.py`
3. Click "Install Dependencies" on first run
4. Once installed, click "Launch Hrudhi"

## Features
- ✅ Smaller initial download (~50MB)
- ✅ Downloads dependencies as needed
- ✅ Internet required for first setup
- ✅ Automatic updates possible
- ✅ AI-powered note search
- ✅ Learning from your notes

## Requirements
- Python 3.8 or higher
- Internet connection (first run only)
- ~2GB disk space for AI models

## File Size
Initial installer: ~50MB
After setup: ~2GB (includes AI models)
'''
    
    with open(installer_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"📂 Installer with dependencies created in: {installer_dir}")
    return True

if __name__ == "__main__":
    success = create_installer_with_deps()
    if success:
        print("\n🎉 Installer with dependencies completed successfully!")
        print("Find your installer in the 'installer_with_deps' folder.")
    else:
        print("\n💥 Build failed. Check the error messages above.")
        sys.exit(1)