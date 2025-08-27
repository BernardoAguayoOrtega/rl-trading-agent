#!/usr/bin/env python3
"""
AI Trading Agent - Quick Setup Script
===================================

This script helps you quickly set up and verify your AI Trading Agent.
"""

import sys
import os
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_prerequisites():
    """Check if prerequisites are installed"""
    print("🔍 Checking Prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        print(f"✅ Python {python_version.major}.{python_version.minor}")
    else:
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    # Check uv
    if run_command("uv --version", "UV package manager check"):
        return True
    else:
        print("💡 Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def setup_environment():
    """Set up the development environment"""
    print("\n📦 Setting Up Environment...")
    
    # Sync dependencies
    if not run_command("uv sync", "Installing dependencies"):
        return False
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found")
        print("💡 Copy .env.example to .env and add your OpenAI API key")
        run_command("cp .env.example .env", "Creating .env from template")
    else:
        print("✅ .env file exists")
    
    return True


def run_tests():
    """Run the test suite"""
    print("\n🧪 Running Tests...")
    
    # Run Phase 1 tests
    if not run_command("uv run python tests/test_phase1.py", "Phase 1 tests"):
        return False
    
    # Check if OpenAI key is configured for AI tests
    with open(".env", "r") as f:
        env_content = f.read()
        if "your-openai-api-key-here" not in env_content and "OPENAI_API_KEY=" in env_content:
            run_command("uv run python tests/test_ai_live.py", "AI integration tests")
        else:
            print("⚠️  OpenAI API key not configured - skipping AI tests")
    
    return True


def show_next_steps():
    """Show what to do next"""
    print("\n🎯 Next Steps:")
    print("=" * 40)
    print("1. Configure your OpenAI API key in .env file")
    print("2. Run tests: uv run python tests/test_phase1.py")
    print("3. Test AI: uv run python tests/test_ai_live.py")
    print("4. Start development: uv run jupyter lab")
    print("5. Open planning notebook: notebooks/ai_trading_agent_plan.ipynb")
    
    print("\n📚 Useful Commands:")
    print("• Main app: uv run ai-trading-agent")
    print("• Jupyter: uv run jupyter lab")
    print("• Tests: uv run python tests/test_phase1.py")
    print("• AI test: uv run python tests/test_ai_live.py")


def main():
    """Main setup function"""
    print("🚀 AI Trading Agent - Quick Setup")
    print("=" * 40)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 Working in: {project_root}")
    
    success = True
    
    # Check prerequisites
    if not check_prerequisites():
        success = False
    
    # Set up environment
    if success and not setup_environment():
        success = False
    
    # Run tests
    if success and not run_tests():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        show_next_steps()
    else:
        print("❌ Setup failed. Please fix the issues above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
