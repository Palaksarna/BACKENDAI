#!/usr/bin/env python3
"""Test script to identify import errors"""

try:
    print("Testing imports...")
    
    print("1. Importing ai_memory_system.app.main...")
    from ai_memory_system.app.main import app
    print("   ✓ Success!")
    
    print("2. Testing FastAPI app object...")
    print(f"   App type: {type(app)}")
    print(f"   Routes: {len(app.routes)}")
    print("   ✓ Success!")
    
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
