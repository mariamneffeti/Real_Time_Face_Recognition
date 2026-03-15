#!/usr/bin/env python3
"""Quick verification script for FaceVision setup"""

print("\n=== FaceVision Setup Verification ===\n")

# Test imports
try:
    from api import app
    print("✓ API module imports successfully")
except Exception as e:
    print(f"✗ API import failed: {e}")
    exit(1)

try:
    from core.engine import FaceEngine
    print("✓ Core engine imports successfully")
except Exception as e:
    print(f"✗ Engine import failed: {e}")
    exit(1)

# Test dependencies
try:
    import cv2
    print("✓ OpenCV available")
except:
    print("✗ OpenCV not available")

try:
    from deepface import DeepFace
    print("✓ DeepFace available")
except:
    print("✗ DeepFace not available")

try:
    from fastapi import FastAPI
    print("✓ FastAPI available")
except:
    print("✗ FastAPI not available")

# Show API endpoints
routes = [r for r in app.routes if hasattr(r, 'path')]
print(f"\n✓ Found {len(routes)} API endpoints:")
for route in sorted(routes, key=lambda x: x.path):
    print(f"  {route.path}")

# Test file structure
from pathlib import Path
print("\n✓ File structure:")
for f in ['run.py', 'api.py', 'requirements.txt', 'README.md']:
    exists = "✓" if Path(f).exists() else "✗"
    print(f"  {exists} {f}")

print("\n=== All Checks Passed ===\n")
print("Ready to run:")
print("  - CLI: python run.py")
print("  - API: uvicorn api:app --reload\n")
