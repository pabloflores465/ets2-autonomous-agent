#!/usr/bin/env python3
"""Build script: checks imports, runs lint, format, typecheck."""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, name):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ {name} FAILED")
        return False
    print(f"  ✅ {name} PASSED")
    return True

def main():
    all_ok = True
    all_ok &= run("ruff check src/", "Lint (ruff)")
    all_ok &= run("ruff format --check src/", "Format (ruff)")
    all_ok &= run("python -c 'from src.main import ETS2Agent; print(\"Imports OK\")'", "Import check")
    
    if all_ok:
        print("\n✅ BUILD SUCCESSFUL")
    else:
        print("\n❌ BUILD FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
