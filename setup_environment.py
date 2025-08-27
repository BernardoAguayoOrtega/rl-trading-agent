#!/usr/bin/env python3
"""
AI Trading Agent Setup Script
============================

This script helps set up the environment for the AI Trading Agent.
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """Set up the trading environment"""
    print("🚀 Setting up AI Trading Agent Environment")
    print("="*50)
    
    # Check if in correct directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OpenAI API Key Setup Required")
        print("Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        
        # Create .env template
        env_template = """# AI Trading Agent Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Key (Required)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Override default trading symbol
# DEFAULT_SYMBOL=SPY

# Optional: Override initial capital
# INITIAL_CAPITAL=100000
"""
        
        with open(".env.template", "w") as f:
            f.write(env_template)
        
        print(f"\n💡 Created .env.template file for your convenience")
    else:
        print("✅ OpenAI API key found")
    
    # Check uv installation
    try:
        import subprocess
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv package manager: {result.stdout.strip()}")
        else:
            print("❌ uv package manager not found")
            return False
    except FileNotFoundError:
        print("❌ uv package manager not found")
        print("💡 Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    print("\n📦 Dependencies Status:")
    
    # List installed packages
    try:
        result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True, cwd=".")
        if "openai" in result.stdout:
            print("✅ OpenAI SDK")
        if "yfinance" in result.stdout:
            print("✅ Yahoo Finance")
        if "pandas" in result.stdout:
            print("✅ Pandas")
        if "numpy" in result.stdout:
            print("✅ NumPy")
        if "scikit-learn" in result.stdout:
            print("✅ Scikit-learn")
        if "ta-lib" in result.stdout:
            print("✅ TA-Lib")
            
    except Exception as e:
        print(f"⚠️  Could not check dependencies: {e}")
    
    print("\n🎯 Quick Start Commands:")
    print("="*30)
    print("# Test Phase 1 setup:")
    print("python test_phase1.py")
    print("\n# Start Jupyter Lab:")
    print("uv run jupyter lab")
    print("\n# Run the main application:")
    print("uv run ai-trading-agent")
    
    print("\n📚 Next Steps:")
    print("1. Set your OpenAI API key (if not done)")
    print("2. Run Phase 1 tests: python test_phase1.py")
    print("3. Open the planning notebook: jupyter lab ai_trading_agent_plan.ipynb")
    print("4. Start implementing Phase 2: Data Management")
    
    return True


def create_gitignore():
    """Create .gitignore file for the project"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.pkl
*.h5
data/
logs/

# API Keys and Secrets
.env
*.key
secrets/

# MacOS
.DS_Store

# Trading specific
backtest_results/
trading_logs/
performance_reports/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")


def main():
    """Main setup function"""
    if setup_environment():
        create_gitignore()
        print("\n🎉 Setup completed successfully!")
        print("Ready to revolutionize trading with AI! 🚀")
        return True
    else:
        print("\n❌ Setup failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
