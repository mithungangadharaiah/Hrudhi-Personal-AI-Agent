#!/usr/bin/env python3
"""
Build script for Hrudhi Standalone Executable
Creates a single executable with all dependencies bundled (~500MB-1GB)
No internet required after download - best user experience
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_standalone():
    """Build standalone executable with all dependencies"""
    
    print("🚀 Building Hrudhi Standalone Executable...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    
    if not main_script.exists():
        print(f"❌ Error: Main script not found at {main_script}")
        return False
    
    # Clean previous builds
    dist_dir = project_root / "dist"
    build_dir_pyinstaller = project_root / "build" / "pyinstaller_temp"
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir_pyinstaller.exists():
        shutil.rmtree(build_dir_pyinstaller)
    
    # Find PyInstaller in virtual environment
    venv_pyinstaller = project_root / ".venv" / "Scripts" / "pyinstaller.exe"
    if venv_pyinstaller.exists():
        pyinstaller_cmd = str(venv_pyinstaller)
    else:
        pyinstaller_cmd = "pyinstaller"
    
    # PyInstaller command for standalone executable
    cmd = [
        pyinstaller_cmd,
        "--onefile",                    # Single executable
        "--windowed",                   # No console window (GUI app)
        "--name=Hrudhi",               # Output name
        "--workpath=" + str(build_dir_pyinstaller),  # Temp build directory
        "--distpath=" + str(dist_dir), # Output directory
        "--hidden-import=sentence_transformers",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=sklearn",
        "--hidden-import=numpy",
        "--collect-all=sentence_transformers",
        "--collect-all=transformers",
        "--collect-all=torch",
        "--collect-all=tokenizers",
        str(main_script)
    ]
    
    print(f"📦 Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    print("⏱️ This may take 10-15 minutes due to large AI models...")
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True, 
                              capture_output=True, text=True)
        print("✅ Build completed successfully!")
        
        # Check output
        exe_path = dist_dir / "Hrudhi.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"📁 Executable created: {exe_path}")
            print(f"📊 Size: {size_mb:.1f} MB")
            
            # Create installer folder
            installer_dir = project_root / "installer_standalone"
            installer_dir.mkdir(exist_ok=True)
            
            # Copy executable to installer folder
            shutil.copy2(exe_path, installer_dir / "Hrudhi.exe")
            
            # Create installation instructions
            readme_content = f"""# Hrudhi Standalone Installation

## Installation
1. Copy Hrudhi.exe to your desired location (e.g., C:\\Program Files\\Hrudhi\\)
2. Double-click Hrudhi.exe to run
3. Your notes will be stored in ~/Desktop/HrudhiNotes

## Features
- ✅ No internet required
- ✅ All dependencies included
- ✅ Simple installation
- ✅ AI-powered note search
- ✅ Learning from your notes

## File Size
This is a standalone executable (~{size_mb:.1f} MB) with all AI models and dependencies included.

## First Run
The first time you run Hrudhi.exe, it may take a few moments to initialize the AI models.
"""
            
            with open(installer_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            print(f"📂 Installer files created in: {installer_dir}")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = build_standalone()
    if success:
        print("\n🎉 Standalone build completed successfully!")
        print("Find your executable in the 'installer_standalone' folder.")
    else:
        print("\n💥 Build failed. Check the error messages above.")
        sys.exit(1)