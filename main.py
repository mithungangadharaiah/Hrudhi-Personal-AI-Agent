#!/usr/bin/env python3
"""
Hrudhi Personal AI Agent
Entry point for the application
"""

import sys
import os

# Add the hrudhi package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hrudhi'))

# Import and run the main application  
from hrudhi import main as hrudhi_main

def main():
    """Main entry point for Hrudhi AI Agent with Creative Interface"""
    try:
        print("🚀 Starting Hrudhi with creative interface...")
        print("🤖 Loading your adorable AI companion...")
        
        hrudhi_main()  # Call the main function from hrudhi package
        
    except KeyboardInterrupt:
        print("\n🤖 Hrudhi has closed. Press any key to exit...")
        input()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting Hrudhi: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install sentence-transformers scikit-learn")
        input("Press any key to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()