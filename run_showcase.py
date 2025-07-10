#!/usr/bin/env python3
"""
Quick launcher for qaLLMApp capabilities showcase
简易启动器 / 簡易ランチャー
"""

import sys
import subprocess
import os

def main():
    """Launch the capabilities showcase"""
    print("🚀 Launching qaLLMApp Capabilities Showcase...")
    print("🚀 qaLLMApp 機能紹介を起動中...")
    print()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    showcase_path = os.path.join(script_dir, "capabilities_showcase.py")
    
    try:
        # Run the capabilities showcase
        subprocess.run([sys.executable, showcase_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running showcase: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye! / さようなら！")
        sys.exit(0)
    except FileNotFoundError:
        print(f"❌ Could not find capabilities_showcase.py at {showcase_path}")
        print("Please make sure you're running this from the qaLLMApp directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()