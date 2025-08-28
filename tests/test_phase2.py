#!/usr/bin/env python3
"""
Phase 2 Testing Script - Data Pipeline & Technical Indicators
============================================================

This script tests the intelligent data management and technical indicators
implementation for the AI Trading Agent.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

from ai_trading_agent.data_manager import IntelligentDataManager, DataValidationError
from ai_trading_agent.indicators import TechnicalIndicatorFactory, MarketDataProcessor
from ai_trading_agent.config import DataConfig


def test_data_manager():
    """Test the IntelligentDataManager"""
    print("📊 Testing Data Manager...")
    
    try:
        # Initialize data manager
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        print("  ✅ Data manager initialized")
        
        # Test single symbol data fetch
        symbol = "SPY"
        data = data_manager.get_market_data(
            symbol=symbol,
            period="6mo",
            interval="1d",
            validate=True
        )
        
        print(f"  ✅ Fetched {len(data)} records for {symbol}")
        print(f"  ✅ Date range: {data.index.min()} to {data.index.max()}")
        
        # Validate data structure
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in data.columns for col in required_columns), "Missing required columns"
        print("  ✅ Data structure validation passed")
        
        # Test data for AI preparation
        ai_data = data_manager.prepare_data_for_ai(data, symbol)
        assert 'current_price' in ai_data, "Missing current_price in AI data"
        assert 'price_change_pct' in ai_data, "Missing price_change_pct in AI data"
        print("  ✅ AI data preparation successful")
        
        print(f"     Current price: ${ai_data['current_price']:.2f}")
        print(f"     Price change: {ai_data['price_change_pct']:.2f}%")
        print(f"     Trend: {ai_data['trend']}")
        
        return True, data
        
    except Exception as e:
        print(f"  ❌ Data manager test failed: {e}")
        return False, None


def test_multiple_symbols():
    """Test multiple symbol data acquisition"""
    print("\n📈 Testing Multiple Symbols...")
    
    try:
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        
        symbols = ["SPY", "QQQ", "IWM", "TLT"]
        all_data = data_manager.get_multiple_symbols(
            symbols=symbols,
            period="3mo",
            interval="1d",
            validate=True
        )
        
        print(f"  ✅ Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
        
        for symbol, data in all_data.items():
            print(f"     {symbol}: {len(data)} records")
        
        return True, all_data
        
    except Exception as e:
        print(f"  ❌ Multiple symbols test failed: {e}")
        return False, None


def test_technical_indicators():
    """Test technical indicators integration"""
    print("\n🔧 Testing Technical Indicators...")
    
    try:
        # Initialize indicator factory
        indicator_factory = TechnicalIndicatorFactory()
        print("  ✅ Indicator factory initialized")
        
        # Get test data
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        data = data_manager.get_market_data("SPY", period="1y", interval="1d")
        
        # Test individual indicators
        print("  🧮 Testing individual indicators...")
        
        # Test SMA
        data_sma = indicator_factory.calculate_indicator(data, 'sma', periodo=20)
        assert 's20' in data_sma.columns, "SMA calculation failed"
        print("     ✅ SMA calculation successful")
        
        # Test RSI
        data_rsi = indicator_factory.calculate_indicator(data, 'rsi', periodo=14)
        assert 'rsi14' in data_rsi.columns, "RSI calculation failed"
        print("     ✅ RSI calculation successful")
        
        # Test MACD
        data_macd = indicator_factory.calculate_indicator(data, 'macd')
        assert all(col in data_macd.columns for col in ['macdL', 'macdS', 'macdH']), "MACD calculation failed"
        print("     ✅ MACD calculation successful")
        
        # Test indicator suite
        data_with_indicators, calculated = indicator_factory.calculate_indicator_suite(data)
        print(f"  ✅ Indicator suite: {len(calculated)} indicators calculated")
        
        # Test indicators for AI
        ai_indicators = indicator_factory.prepare_indicators_for_ai(data_with_indicators, "SPY")
        print(f"  ✅ AI indicators prepared: {len(ai_indicators)} indicators")
        
        # Display some key indicators
        if 'RSI' in ai_indicators:
            print(f"     RSI: {ai_indicators['RSI']:.2f}")
        if 'MACD' in ai_indicators:
            print(f"     MACD: {ai_indicators['MACD']:.4f}")
        if 'SMA_20' in ai_indicators:
            print(f"     SMA 20: ${ai_indicators['SMA_20']:.2f}")
        
        return True, ai_indicators
        
    except Exception as e:
        print(f"  ❌ Technical indicators test failed: {e}")
        return False, None


def test_market_data_processor():
    """Test the MarketDataProcessor"""
    print("\n🔍 Testing Market Data Processor...")
    
    try:
        # Initialize processor
        processor = MarketDataProcessor()
        print("  ✅ Market data processor initialized")
        
        # Get test data
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        data = data_manager.get_market_data("SPY", period="6mo", interval="1d")
        
        # Process data for AI
        ai_analysis = processor.process_for_ai_analysis(data, "SPY")
        print("  ✅ Data processed for AI analysis")
        
        # Validate AI analysis structure
        required_sections = ['price_data', 'volume_data', 'volatility_data', 'technical_indicators', 'market_structure']
        for section in required_sections:
            assert section in ai_analysis, f"Missing {section} in AI analysis"
        
        print("  ✅ AI analysis structure validation passed")
        
        # Display analysis summary
        price_data = ai_analysis['price_data']
        market_structure = ai_analysis['market_structure']
        
        print(f"     Current Price: ${price_data['current_price']:.2f}")
        print(f"     Price Change: {price_data['price_change_pct']:.2f}%")
        print(f"     Trend Strength: {market_structure['trend_strength']:.2f}")
        print(f"     Momentum: {market_structure['momentum']}")
        print(f"     Indicators: {len(ai_analysis['technical_indicators'])}")
        
        return True, ai_analysis
        
    except Exception as e:
        print(f"  ❌ Market data processor test failed: {e}")
        return False, None


def test_data_validation():
    """Test data validation capabilities"""
    print("\n🔍 Testing Data Validation...")
    
    try:
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        
        # Test with invalid symbol (should handle gracefully)
        try:
            data_manager.get_market_data("INVALID_SYMBOL_123", period="1mo")
            print("  ⚠️  Invalid symbol test: Expected error but got data")
        except DataValidationError:
            print("  ✅ Invalid symbol properly rejected")
        
        # Test data quality report
        # First get some valid data with longer period to ensure sufficient data
        data_manager.get_market_data("SPY", period="3mo")  # Use 3 months instead of 1
        data_manager.get_market_data("QQQ", period="3mo")
        
        quality_report = data_manager.get_data_quality_report()
        print(f"  ✅ Data quality report generated for {quality_report['total_symbols_tracked']} symbols")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data validation test failed: {e}")
        return False


def test_real_time_capabilities():
    """Test real-time data capabilities"""
    print("\n⚡ Testing Real-time Capabilities...")
    
    try:
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        
        # Test real-time data (will get recent intraday data)
        symbol = "SPY"
        real_time_data = data_manager.get_real_time_data(symbol, max_age_minutes=60)
        
        print(f"  ✅ Real-time data fetched: {len(real_time_data)} records")
        if not real_time_data.empty:
            latest_time = real_time_data.index[-1]
            print(f"     Latest data point: {latest_time}")
            print(f"     Latest price: ${real_time_data['Close'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real-time capabilities test failed: {e}")
        return False


def test_ai_integration():
    """Test integration with existing AI client"""
    print("\n🤖 Testing AI Integration...")
    
    try:
        from ai_trading_agent.ai_client import AITradingClient
        from ai_trading_agent.config import Config
        
        # Initialize components
        config = Config()
        data_manager = IntelligentDataManager(config.data)
        processor = MarketDataProcessor()
        ai_client = AITradingClient(config.openai)
        
        # Get and process market data
        symbol = "SPY"
        data = data_manager.get_market_data(symbol, period="3mo", interval="1d")
        ai_analysis = processor.process_for_ai_analysis(data, symbol)
        
        # Prepare data for AI analysis
        market_data = {
            'current_price': ai_analysis['price_data']['current_price'],
            'price_change': ai_analysis['price_data']['price_change_pct'],
            'volume': ai_analysis['volume_data']['current_volume'],
            'technical_indicators': ai_analysis['technical_indicators'],
            'market_hours': 'open',  # Simplified
            'volatility': ai_analysis['volatility_data']['volatility_20'],
            'trend': 'upward' if ai_analysis['market_structure']['trend_strength'] > 0.5 else 'downward'
        }
        
        # Test AI market analysis with real data
        ai_response = ai_client.analyze_market_conditions(market_data)
        
        print("  ✅ AI analysis with real market data successful")
        print(f"     Market Regime: {ai_response.get('market_regime', 'Unknown')}")
        print(f"     Trend Strength: {ai_response.get('trend_strength', 'Unknown')}")
        print(f"     Market Score: {ai_response.get('market_score', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI integration test failed: {e}")
        print("     Note: This requires a valid OpenAI API key")
        return False


def main():
    """Main test function for Phase 2"""
    print("🚀 AI Trading Agent - Phase 2 Testing")
    print("=" * 50)
    print("Testing Intelligent Data Management System")
    print("=" * 50)
    
    test_results = []
    
    # Test data manager
    result, test_data = test_data_manager()
    test_results.append(("Data Manager", result))
    
    # Test multiple symbols
    result, _ = test_multiple_symbols()
    test_results.append(("Multiple Symbols", result))
    
    # Test technical indicators
    result, _ = test_technical_indicators()
    test_results.append(("Technical Indicators", result))
    
    # Test market data processor
    result, _ = test_market_data_processor()
    test_results.append(("Market Data Processor", result))
    
    # Test data validation
    result = test_data_validation()
    test_results.append(("Data Validation", result))
    
    # Test real-time capabilities
    result = test_real_time_capabilities()
    test_results.append(("Real-time Capabilities", result))
    
    # Test AI integration
    result = test_ai_integration()
    test_results.append(("AI Integration", result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ Intelligent Data Management is working correctly")
        print("✅ Technical indicators integration successful")
        print("✅ AI integration with real market data confirmed")
        print("\n🎯 Phase 2 Complete! Ready for Phase 3: Framework Integration")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
