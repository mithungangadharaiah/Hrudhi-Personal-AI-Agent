#!/usr/bin/env python3
"""
Master build script for Hrudhi AI Agent
Provides options to build both distribution approaches
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build Hrudhi AI Agent")
    parser.add_argument("--approach", choices=["standalone", "installer", "both"], 
                       default="both", help="Distribution approach")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"
    
    print("🤖 Hrudhi AI Agent Build System")
    print("=" * 40)
    
    success = True
    
    if args.approach in ["standalone", "both"]:
        print("\n🚀 Building Standalone Executable...")
        try:
            sys.path.insert(0, str(build_dir))
            from build_standalone import build_standalone
            success &= build_standalone()
        except Exception as e:
            print(f"❌ Standalone build failed: {e}")
            success = False
    
    if args.approach in ["installer", "both"]:
        print("\n🚀 Building Installer with Dependencies...")
        try:
            sys.path.insert(0, str(build_dir))
            from build_installer import create_installer_with_deps
            success &= create_installer_with_deps()
        except Exception as e:
            print(f"❌ Installer build failed: {e}")
            success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Build completed successfully!")
        print("\nDistribution Options:")
        print("📁 Standalone: ./installer_standalone/Hrudhi.exe (~500MB-1GB)")
        print("📁 Installer: ./installer_with_deps/Start_Hrudhi.bat (~50MB)")
    else:
        print("💥 Build failed. Check error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())