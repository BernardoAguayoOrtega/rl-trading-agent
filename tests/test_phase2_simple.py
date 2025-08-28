#!/usr/bin/env python3
"""
Phase 2 Testing Script - Data Pipeline & Technical Indicators
============================================================

This script tests the newly implemented Phase 2 components.
"""

import os
import sys
from pathlib import Path
import warnings

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all Phase 2 components can be imported"""
    print("📦 Testing Imports...")
    
    try:
        from ai_trading_agent.data_manager import IntelligentDataManager, DataValidationError
        print("  ✅ Data Manager imported successfully")
        
        from ai_trading_agent.indicators import TechnicalIndicatorFactory, MarketDataProcessor
        print("  ✅ Technical Indicators imported successfully")
        
        from ai_trading_agent.config import DataConfig
        print("  ✅ Data Config imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_data_manager():
    """Test the intelligent data manager"""
    print("\n📊 Testing Data Manager...")
    
    try:
        from ai_trading_agent.data_manager import IntelligentDataManager
        
        # Initialize data manager
        data_manager = IntelligentDataManager()
        print("  ✅ Data manager initialized")
        
        # Test basic data fetching
        data = data_manager.get_market_data("SPY", period="5d", interval="1d")
        print(f"  ✅ Successfully fetched {len(data)} records for SPY")
        
        # Verify data structure
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if not missing_columns:
            print("  ✅ Data structure validation passed")
        else:
            print(f"  ⚠️  Missing columns: {missing_columns}")
        
        # Test AI data preparation
        ai_data = data_manager.prepare_data_for_ai(data, "SPY")
        print(f"  ✅ AI data prepared - Price: ${ai_data['current_price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data Manager test failed: {e}")
        return False


def test_technical_indicators():
    """Test the technical indicators"""
    print("\n📈 Testing Technical Indicators...")
    
    try:
        from ai_trading_agent.indicators import TechnicalIndicatorFactory
        from ai_trading_agent.data_manager import IntelligentDataManager
        
        # Get some data to work with
        data_manager = IntelligentDataManager()
        data = data_manager.get_market_data("SPY", period="3mo", interval="1d")
        
        # Initialize indicator factory
        indicator_factory = TechnicalIndicatorFactory()
        print("  ✅ Indicator factory initialized")
        
        # Test individual indicators
        indicators_to_test = ['sma', 'rsi', 'macd']
        
        for indicator_name in indicators_to_test:
            try:
                result = indicator_factory.calculate_indicator(data, indicator_name)
                print(f"  ✅ {indicator_name.upper()} calculated successfully")
            except Exception as e:
                print(f"  ❌ {indicator_name.upper()} failed: {e}")
                return False
        
        # Test AI-ready indicators
        ai_indicators = indicator_factory.get_ai_ready_indicators(data, "SPY")
        print(f"  ✅ AI indicators ready - RSI: {ai_indicators.get('rsi_14', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Technical Indicators test failed: {e}")
        return False


def test_market_data_processor():
    """Test the market data processor"""
    print("\n🔄 Testing Market Data Processor...")
    
    try:
        from ai_trading_agent.indicators import MarketDataProcessor
        
        # Initialize processor
        processor = MarketDataProcessor()
        print("  ✅ Market data processor initialized")
        
        # Test complete market analysis
        analysis = processor.get_complete_market_analysis("SPY", period="5d")
        
        # Validate analysis structure
        required_sections = ['symbol', 'current_price', 'technical_indicators']
        has_all_sections = all(section in analysis for section in required_sections)
        
        if has_all_sections:
            print("  ✅ Complete market analysis generated")
            print(f"    Symbol: {analysis['symbol']}")
            print(f"    Current Price: ${analysis['current_price']:.2f}")
            print(f"    Technical Indicators: {len(analysis['technical_indicators'])} indicators")
        else:
            missing = [section for section in required_sections if section not in analysis]
            print(f"  ⚠️  Missing analysis sections: {missing}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Market Data Processor test failed: {e}")
        return False


def test_ai_integration():
    """Test integration with AI components"""
    print("\n🤖 Testing AI Integration...")
    
    try:
        from ai_trading_agent.indicators import MarketDataProcessor
        
        # Get market analysis in AI-ready format
        processor = MarketDataProcessor()
        analysis = processor.get_complete_market_analysis("SPY")
        
        # Validate AI input format
        required_ai_fields = ['current_price', 'technical_indicators', 'trend']
        has_required = all(field in analysis for field in required_ai_fields)
        
        if has_required:
            print("  ✅ AI integration format validated")
            print(f"    Market Trend: {analysis['trend']}")
            print(f"    Volatility: {analysis['volatility']:.2f}%")
            print(f"    RSI Signal: {analysis['technical_indicators'].get('rsi_signal', 'unknown')}")
        else:
            missing = [field for field in required_ai_fields if field not in analysis]
            print(f"  ❌ Missing AI fields: {missing}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI Integration test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests"""
    print("🚀 AI Trading Agent - Phase 2 Testing")
    print("=" * 50)
    
    # Track test results
    tests = [
        ("Imports", test_imports),
        ("Data Manager", test_data_manager),
        ("Technical Indicators", test_technical_indicators),
        ("Market Data Processor", test_market_data_processor),
        ("AI Integration", test_ai_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Tests...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test suite failed: {e}")
            results.append((test_name, False))
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("\n📋 Phase 2 Implementation Complete!")
        print("✅ Data pipeline functional")
        print("✅ Technical indicators working")
        print("✅ AI integration ready")
        print("\n🎯 Ready for advanced trading strategies!")
    else:
        print(f"⚠️  {total - passed} test suite(s) failed")
        print("📋 Please review and fix issues before proceeding")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
