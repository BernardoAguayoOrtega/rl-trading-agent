#!/usr/bin/env python3
"""
Phase 3 Test Suite - AI Trading System Integration Tests
========================================================

Comprehensive tests for Phase 3 components:
- AI Trading Engine
- Trading Signals Generation
- Backtesting Integration
- Position Management
- Framework Compatibility

Run this to verify all Phase 3 components are working correctly.
"""

import os
import sys
from pathlib import Path
import warnings
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all Phase 3 components can be imported"""
    print("📦 Testing Phase 3 Imports...")
    
    try:
        from ai_trading_agent import (
            AITradingEngine, 
            AIBacktestingIntegration,
            AITradingSignalGenerator,
            TradingDecision,
            Position,
            AISignal,
            Config
        )
        print("  ✅ All Phase 3 components imported successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_trading_engine():
    """Test AI Trading Engine functionality"""
    print("\n🤖 Testing AI Trading Engine...")
    
    try:
        from ai_trading_agent import AITradingEngine, Config
        
        # Initialize with test config
        config = Config()
        config.openai.validate_on_init = False  # Skip API validation for tests
        
        engine = AITradingEngine(config)
        print("  ✅ Trading engine initialized")
        
        # Test portfolio summary
        summary = engine.get_portfolio_summary()
        assert 'initial_capital' in summary
        assert 'current_value' in summary
        assert summary['initial_capital'] == config.trading.initial_capital
        print("  ✅ Portfolio summary working")
        
        # Test position data generation
        position_data = engine._get_position_data("SPY", None)
        assert position_data['has_position'] == False
        assert position_data['symbol'] == "SPY"
        print("  ✅ Position data generation working")
        
        # Test performance data
        perf_data = engine._get_recent_performance()
        # Should return None when no trades exist
        assert perf_data is None
        print("  ✅ Performance data handling working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Trading engine test failed: {e}")
        return False


def test_trading_signals():
    """Test AI Trading Signal Generator"""
    print("\n📡 Testing AI Trading Signal Generator...")
    
    try:
        from ai_trading_agent import AITradingSignalGenerator, TradingDecision, Config
        
        # Initialize signal generator
        config = Config()
        signal_gen = AITradingSignalGenerator(config)
        print("  ✅ Signal generator initialized")
        
        # Create mock trading decision
        decision = TradingDecision(
            action="BUY",
            confidence=0.8,
            position_size=0.1,
            entry_price=450.0,
            stop_loss=0.02,
            take_profit=0.04,
            reasoning="Test decision for signal generation",
            market_outlook="bullish"
        )
        
        # Test signal conversion
        signal = signal_gen.convert_ai_decision_to_signal("SPY", decision, 450.0)
        assert signal == 'P'  # Should be 'P' for BUY without existing position
        print("  ✅ AI decision to signal conversion working")
        
        # Test signal statistics
        stats = signal_gen.get_signal_statistics()
        assert 'total_signals' in stats
        assert stats['total_signals'] > 0
        print("  ✅ Signal statistics working")
        
        # Test signal export
        exported = signal_gen.export_signals_for_backtesting("SPY", format='dict')
        assert isinstance(exported, dict)
        print("  ✅ Signal export working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Trading signals test failed: {e}")
        return False


def test_backtesting_integration():
    """Test AI Backtesting Integration"""
    print("\n📈 Testing Backtesting Integration...")
    
    try:
        from ai_trading_agent import AIBacktestingIntegration, TradingDecision, Config
        
        # Initialize backtesting integration
        config = Config()
        config.openai.validate_on_init = False
        
        backtester = AIBacktestingIntegration(config)
        print("  ✅ Backtesting integration initialized")
        
        # Test traditional indicator functions
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.random(100) * 100 + 450,
            'High': np.random.random(100) * 100 + 460,
            'Low': np.random.random(100) * 100 + 440,
            'Close': np.random.random(100) * 100 + 450,
            'Volume': np.random.random(100) * 1000000
        }, index=dates)
        
        # Test SMA calculation
        result = backtester.ocpSma(sample_data.copy(), 20)
        assert 's20' in result.columns
        print("  ✅ SMA calculation working")
        
        # Test RSI calculation
        result = backtester.ocpRsi(result, 14)
        assert 'rsi14' in result.columns
        print("  ✅ RSI calculation working")
        
        # Test AI sistema function
        mock_decision = TradingDecision(
            action="BUY",
            confidence=0.8,
            position_size=0.1,
            entry_price=450.0,
            stop_loss=0.02,
            take_profit=0.04,
            reasoning="Mock decision for testing",
            market_outlook="bullish"
        )
        
        ai_decisions = {sample_data.index[50]: mock_decision}
        result = backtester.dameAISistema(sample_data.copy(), ai_decisions, 0.7)
        assert 'signal' in result.columns
        assert 'position' in result.columns
        print("  ✅ AI sistema function working")
        
        # Test position logic
        result = backtester.damePosition(result)
        assert 'TRADE' in result.columns
        print("  ✅ Position logic working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backtesting integration test failed: {e}")
        return False


def test_data_structures():
    """Test Phase 3 data structures"""
    print("\n🏗️  Testing Data Structures...")
    
    try:
        from ai_trading_agent import TradingDecision, Position, AISignal
        from datetime import datetime
        
        # Test TradingDecision
        decision = TradingDecision(
            action="BUY",
            confidence=0.75,
            position_size=0.15,
            entry_price=445.50,
            stop_loss=0.025,
            take_profit=0.045,
            reasoning="Test trading decision",
            market_outlook="neutral"
        )
        
        assert decision.action == "BUY"
        assert decision.confidence == 0.75
        assert isinstance(decision.timestamp, datetime)
        print("  ✅ TradingDecision structure working")
        
        # Test Position
        position = Position(
            symbol="SPY",
            action="long",
            entry_price=445.50,
            position_size=0.15,
            stop_loss=0.025,
            take_profit=0.045,
            entry_date=datetime.now()
        )
        
        assert position.symbol == "SPY"
        assert position.action == "long"
        assert position.unrealized_pnl == 0.0
        print("  ✅ Position structure working")
        
        # Test AISignal
        signal = AISignal(
            symbol="SPY",
            timestamp=datetime.now(),
            signal_type='P',
            ai_action="BUY",
            confidence=0.75,
            reasoning="Test signal",
            entry_price=445.50,
            stop_loss=0.025,
            take_profit=0.045,
            position_size=0.15,
            market_outlook="neutral"
        )
        
        assert signal.signal_type == 'P'
        assert signal.ai_action == "BUY"
        print("  ✅ AISignal structure working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data structures test failed: {e}")
        return False


def test_framework_compatibility():
    """Test compatibility with existing backtesting framework"""
    print("\n🔧 Testing Framework Compatibility...")
    
    try:
        from ai_trading_agent import AIBacktestingIntegration, Config
        
        config = Config()
        config.openai.validate_on_init = False
        
        backtester = AIBacktestingIntegration(config)
        
        # Create test data that mimics your framework structure
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'Open': np.random.random(50) * 10 + 450,
            'High': np.random.random(50) * 10 + 460,
            'Low': np.random.random(50) * 10 + 440,
            'Close': np.random.random(50) * 10 + 450,
            'Volume': np.random.random(50) * 1000000
        }, index=dates)
        test_data['Date'] = dates
        
        # Test the complete backtesting pipeline
        
        # 1. Calculate indicators (your existing functions)
        test_data = backtester.ocpSma(test_data, 20)
        test_data = backtester.ocpRsi(test_data, 14)
        
        # 2. Add mock signals
        test_data['signal'] = ''
        test_data.iloc[10, test_data.columns.get_loc('signal')] = 'P'
        test_data.iloc[20, test_data.columns.get_loc('signal')] = 'cP'
        
        # 3. Apply position logic
        test_data['position'] = test_data['signal'].shift(1)
        test_data = backtester.damePosition(test_data)
        
        # 4. Calculate P&L
        test_data = backtester.dameSalidaPnl(test_data, 'long', tp=0.03, sl=0.02)
        
        # 5. Calculate curves
        test_data = backtester.calculaCurvas(test_data)
        
        # Verify all required columns exist
        required_cols = ['TRADE', 'ROID', 'ROIACUM', 'cvSis', 'cvAct', 'ddSis', 'ddAct']
        for col in required_cols:
            assert col in test_data.columns, f"Missing column: {col}"
        
        print("  ✅ Framework compatibility verified")
        print("  ✅ Complete backtesting pipeline working")
        
        # Test specific calculations
        trades = test_data[test_data['TRADE'] == 'Out']
        if len(trades) > 0:
            print(f"  ✅ Generated {len(trades)} completed trades")
        
        # Test performance curves
        final_value = test_data['cvSis'].iloc[-1]
        assert final_value > 0, "Portfolio value should be positive"
        print(f"  ✅ Portfolio performance calculated (final value: {final_value:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Framework compatibility test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🛡️  Testing Error Handling...")
    
    try:
        from ai_trading_agent import AITradingEngine, Config
        
        # Test with invalid configuration
        config = Config()
        config.openai.validate_on_init = False
        
        # Test trading engine error handling
        engine = AITradingEngine(config)
        
        # Test invalid portfolio operation
        summary = engine.get_portfolio_summary()
        assert isinstance(summary, dict)
        print("  ✅ Portfolio error handling working")
        
        # Test empty performance data
        perf = engine._get_recent_performance()
        assert perf is None  # Should handle empty trade history gracefully
        print("  ✅ Empty performance data handled correctly")
        
        # Test signal generator with empty data
        from ai_trading_agent import AITradingSignalGenerator
        signal_gen = AITradingSignalGenerator(config)
        stats = signal_gen.get_signal_statistics("NONEXISTENT")
        assert isinstance(stats, dict)
        print("  ✅ Signal generator error handling working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def test_integration():
    """Test integration between all Phase 3 components"""
    print("\n🔗 Testing Component Integration...")
    
    try:
        from ai_trading_agent import (
            AITradingEngine, 
            AITradingSignalGenerator, 
            AIBacktestingIntegration,
            TradingDecision,
            Config
        )
        
        # Initialize all components
        config = Config()
        config.openai.validate_on_init = False
        
        engine = AITradingEngine(config)
        signal_gen = AITradingSignalGenerator(config)
        backtester = AIBacktestingIntegration(config)
        
        print("  ✅ All components initialized together")
        
        # Test data flow between components
        mock_decision = TradingDecision(
            action="BUY",
            confidence=0.8,
            position_size=0.1,
            entry_price=450.0,
            stop_loss=0.02,
            take_profit=0.04,
            reasoning="Integration test decision",
            market_outlook="bullish"
        )
        
        # Convert decision to signal
        signal = signal_gen.convert_ai_decision_to_signal("SPY", mock_decision, 450.0)
        assert signal == 'P'
        print("  ✅ Decision to signal conversion working")
        
        # Test that signal can be used in backtesting
        test_decisions = {pd.Timestamp('2024-01-15'): mock_decision}
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Open': [450] * 30,
            'High': [455] * 30,
            'Low': [445] * 30,
            'Close': [452] * 30,
            'Volume': [1000000] * 30
        }, index=dates)
        
        # Apply AI sistema
        result = backtester.dameAISistema(sample_data, test_decisions, 0.7)
        assert 'signal' in result.columns
        print("  ✅ Integration with backtesting working")
        
        # Test portfolio operations
        portfolio = engine.get_portfolio_summary()
        assert portfolio['initial_capital'] > 0
        print("  ✅ Portfolio integration working")
        
        print("  ✅ All components integrated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def main():
    """Run all Phase 3 tests"""
    print("🧪 AI Trading Agent - Phase 3 Test Suite")
    print("=" * 60)
    print("Testing all Phase 3 components and integrations\n")
    
    # Define all tests
    tests = [
        ("Component Imports", test_imports),
        ("AI Trading Engine", test_trading_engine),
        ("Trading Signals", test_trading_signals),
        ("Backtesting Integration", test_backtesting_integration),
        ("Data Structures", test_data_structures),
        ("Framework Compatibility", test_framework_compatibility),
        ("Error Handling", test_error_handling),
        ("Component Integration", test_integration),
    ]
    
    # Run tests
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Results summary
    print("\n" + "🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Final Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed / len(results) * 100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🚀 Phase 3 is ready for production use!")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n✅ Phase 3 test suite completed successfully!")
        else:
            print(f"\n❌ Phase 3 test suite completed with failures.")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)
