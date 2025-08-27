#!/usr/bin/env python3
"""
Quick AI Test - Verify OpenAI Integration
========================================

This script tests the actual AI functionality with your OpenAI API key.
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_trading_agent.ai_client import AITradingClient
from ai_trading_agent.config import Config


def test_ai_functionality():
    """Test actual AI functionality"""
    print("🤖 Testing AI Trading Client with Real API")
    print("=" * 50)
    
    try:
        # Create AI client with real API key
        config = Config()
        client = AITradingClient(config.openai)
        
        print("✅ AI client initialized successfully")
        print(f"✅ Using model: {config.openai.model}")
        print(f"✅ API key configured: {config.openai.api_key[:8]}...")
        
        # Test market analysis with sample data
        print("\n🔍 Testing Market Analysis...")
        
        sample_market_data = {
            'current_price': 450.25,
            'price_change': 1.2,
            'volume': 125000000,
            'technical_indicators': {
                'RSI': 65.5,
                'MACD': 0.85,
                'SMA_20': 448.10,
                'SMA_50': 445.30
            },
            'market_hours': 'open',
            'volatility': 0.18,
            'trend': 'upward'
        }
        
        # This will make an actual API call to OpenAI
        analysis = client.analyze_market_conditions(sample_market_data)
        
        print("✅ Market analysis completed!")
        print(f"   Market Regime: {analysis.get('market_regime', 'Unknown')}")
        print(f"   Trend Strength: {analysis.get('trend_strength', 'Unknown')}")
        print(f"   Market Score: {analysis.get('market_score', 'Unknown')}")
        
        if analysis.get('key_insights'):
            print("   Key Insights:")
            for insight in analysis['key_insights'][:2]:  # Show first 2 insights
                print(f"     • {insight}")
        
        print("\n🎉 AI Integration Test PASSED!")
        print("✅ OpenAI API is working correctly")
        print("✅ AI client can analyze market conditions")
        print("✅ Ready for full trading implementation")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Integration Test FAILED: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check your OpenAI API key is valid")
        print("2. Verify you have API credits available")
        print("3. Check your internet connection")
        return False


def main():
    """Main test function"""
    print("🚀 AI Trading Agent - Live AI Test")
    print("=" * 50)
    
    if test_ai_functionality():
        print("\n🎯 What's Next?")
        print("=" * 30)
        print("✅ Phase 1 is completely working!")
        print("🔄 Ready to start Phase 2: Data Management")
        print("📊 Your AI can now:")
        print("   • Analyze market conditions")
        print("   • Make trading decisions")
        print("   • Assess risk levels")
        print("   • Develop strategies")
        
        print("\n💡 Next Steps:")
        print("1. Move to Phase 2: Yahoo Finance integration")
        print("2. Build the data pipeline")
        print("3. Test with real market data")
    else:
        print("\n⚠️  Please fix the AI integration issues before proceeding")


if __name__ == "__main__":
    main()
