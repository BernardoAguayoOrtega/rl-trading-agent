#!/usr/bin/env python3
"""
Test Runner for AI Trading Agent
===============================

Runs all tests and provides comprehensive reporting.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file, description):
    """Run a test file and return results"""
    print(f"\n{'='*50}")
    print(f"🧪 Running {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file], 
            capture_output=False, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED")
            return False
            
    except Exception as e:
        print(f"❌ {description} ERROR: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 AI Trading Agent - Test Suite")
    print("=" * 50)
    
    tests = [
        ("tests/test_phase1.py", "Phase 1 Environment Tests"),
        ("tests/test_ai_live.py", "AI Integration Tests")
    ]
    
    results = []
    
    for test_file, description in tests:
        test_path = Path(__file__).parent.parent / test_file
        if test_path.exists():
            success = run_test(test_file, description)
            results.append((description, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for Phase 2!")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
