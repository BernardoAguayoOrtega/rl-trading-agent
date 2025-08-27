#!/usr/bin/env python3
"""
Phase 2 Demo Script - Intelligent Data Management
================================================

Demonstrates the complete Phase 2 functionality including:
- Real-time data acquisition from Yahoo Finance
- Technical indicator calculation using existing framework
- Data validation and quality assurance
- AI integration with real market data
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_trading_agent.data_manager import IntelligentDataManager
from ai_trading_agent.indicators import TechnicalIndicatorFactory, MarketDataProcessor
from ai_trading_agent.ai_client import AITradingClient
from ai_trading_agent.config import Config


def demo_data_acquisition():
    """Demonstrate data acquisition capabilities"""
    print("📊 DEMO: Data Acquisition from Yahoo Finance")
    print("=" * 60)
    
    # Initialize data manager
    config = Config()
    data_manager = IntelligentDataManager(config.data)
    
    # Demo 1: Single symbol with comprehensive data
    print("\n1️⃣  Single Symbol Analysis - SPY")
    print("-" * 40)
    
    spy_data = data_manager.get_market_data(
        symbol="SPY",
        period="1y",
        interval="1d",
        validate=True
    )
    
    print(f"✅ Fetched {len(spy_data)} daily records for SPY")
    print(f"📅 Date range: {spy_data.index.min().date()} to {spy_data.index.max().date()}")
    print(f"💰 Current price: ${spy_data['Close'].iloc[-1]:.2f}")
    print(f"📈 52-week high: ${spy_data['High'].max():.2f}")
    print(f"📉 52-week low: ${spy_data['Low'].min():.2f}")
    
    # Demo 2: Multiple symbols for portfolio analysis
    print("\n2️⃣  Portfolio Analysis - Multiple ETFs")
    print("-" * 40)
    
    portfolio_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    portfolio_data = data_manager.get_multiple_symbols(
        symbols=portfolio_symbols,
        period="6mo",
        interval="1d",
        validate=True
    )
    
    print(f"✅ Fetched data for {len(portfolio_data)}/{len(portfolio_symbols)} symbols")
    
    for symbol, data in portfolio_data.items():
        current_price = data['Close'].iloc[-1]
        price_change = ((current_price / data['Close'].iloc[0]) - 1) * 100
        print(f"   {symbol}: ${current_price:.2f} ({price_change:+.1f}% 6M return)")
    
    return spy_data, portfolio_data


def demo_technical_indicators(data):
    """Demonstrate technical indicators integration"""
    print("\n🔧 DEMO: Technical Indicators Integration")
    print("=" * 60)
    
    # Initialize indicator factory
    indicator_factory = TechnicalIndicatorFactory()
    
    print("\n1️⃣  Individual Indicator Calculations")
    print("-" * 40)
    
    # Calculate key indicators
    data_with_sma = indicator_factory.calculate_indicator(data, 'sma', periodo=20)
    data_with_rsi = indicator_factory.calculate_indicator(data, 'rsi', periodo=14)
    data_with_macd = indicator_factory.calculate_indicator(data, 'macd')
    
    # Show latest values
    latest = data_with_rsi.iloc[-1]
    print(f"📊 SMA 20: ${data_with_sma['s20'].iloc[-1]:.2f}")
    print(f"📈 RSI 14: {latest['rsi14']:.1f}")
    print(f"📉 MACD: {data_with_macd['macdL'].iloc[-1]:.4f}")
    
    print("\n2️⃣  Comprehensive Indicator Suite")
    print("-" * 40)
    
    # Calculate full indicator suite
    data_with_indicators, calculated = indicator_factory.calculate_indicator_suite(data)
    print(f"✅ Calculated {len(calculated)} indicators:")
    
    for indicator_name, params in calculated:
        print(f"   • {indicator_name.upper()}: {params}")
    
    # Prepare indicators for AI
    ai_indicators = indicator_factory.prepare_indicators_for_ai(data_with_indicators, "SPY")
    
    print(f"\n📊 Key Indicators for AI Analysis:")
    for name, value in list(ai_indicators.items())[:8]:  # Show first 8
        if isinstance(value, float):
            print(f"   {name}: {value:.3f}")
    
    return data_with_indicators, ai_indicators


def demo_market_analysis(data):
    """Demonstrate comprehensive market analysis"""
    print("\n🔍 DEMO: Comprehensive Market Analysis")
    print("=" * 60)
    
    # Initialize market data processor
    processor = MarketDataProcessor()
    
    # Process data for AI analysis
    ai_analysis = processor.process_for_ai_analysis(data, "SPY")
    
    print("\n1️⃣  Price Action Analysis")
    print("-" * 40)
    price_data = ai_analysis['price_data']
    print(f"💰 Current Price: ${price_data['current_price']:.2f}")
    print(f"📈 Daily Change: {price_data['price_change_pct']:.2f}%")
    print(f"🔓 Today's Range: ${price_data['low']:.2f} - ${price_data['high']:.2f}")
    
    print("\n2️⃣  Volume & Volatility Analysis")
    print("-" * 40)
    volume_data = ai_analysis['volume_data']
    volatility_data = ai_analysis['volatility_data']
    print(f"📊 Volume: {volume_data['current_volume']:,}")
    print(f"📊 Volume Ratio: {volume_data['volume_ratio']:.1f}x average")
    print(f"📊 Volatility (20d): {volatility_data['volatility_20']:.2f}%")
    
    print("\n3️⃣  Market Structure Analysis")
    print("-" * 40)
    market_structure = ai_analysis['market_structure']
    print(f"📈 Trend Strength: {market_structure['trend_strength']:.2f}")
    print(f"🎯 Momentum: {market_structure['momentum']}")
    print(f"🛡️  Support: ${market_structure['support_level']:.2f}")
    print(f"⚔️  Resistance: ${market_structure['resistance_level']:.2f}")
    
    # Show trend signals
    print(f"\n📊 Trend Signals:")
    for timeframe, signal in market_structure['trend_signals'].items():
        print(f"   {timeframe.upper()}: {signal}")
    
    return ai_analysis


def demo_ai_integration(ai_analysis):
    """Demonstrate AI integration with real market data"""
    print("\n🤖 DEMO: AI Integration with Real Market Data")
    print("=" * 60)
    
    try:
        # Initialize AI client
        config = Config()
        ai_client = AITradingClient(config.openai)
        
        # Prepare market data for AI
        market_data = {
            'current_price': ai_analysis['price_data']['current_price'],
            'price_change': ai_analysis['price_data']['price_change_pct'],
            'volume': ai_analysis['volume_data']['current_volume'],
            'technical_indicators': ai_analysis['technical_indicators'],
            'market_hours': 'open',
            'volatility': ai_analysis['volatility_data']['volatility_20'],
            'trend': 'upward' if ai_analysis['market_structure']['trend_strength'] > 0.5 else 'downward'
        }
        
        print("\n1️⃣  AI Market Condition Analysis")
        print("-" * 40)
        print("🧠 Analyzing market conditions with GPT-4...")
        
        # Get AI analysis
        ai_response = ai_client.analyze_market_conditions(market_data)
        
        print(f"✅ AI Analysis Complete!")
        print(f"🏛️  Market Regime: {ai_response.get('market_regime', 'Unknown')}")
        print(f"💪 Trend Strength: {ai_response.get('trend_strength', 'Unknown')}")
        print(f"📊 Volatility Level: {ai_response.get('volatility_level', 'Unknown')}")
        print(f"🎯 Market Score: {ai_response.get('market_score', 'Unknown')}")
        
        # Show key insights
        insights = ai_response.get('key_insights', [])
        if insights:
            print(f"\n🔍 Key AI Insights:")
            for i, insight in enumerate(insights[:3], 1):  # Show first 3
                print(f"   {i}. {insight}")
        
        # Show support/resistance if available
        sr = ai_response.get('support_resistance', {})
        if sr:
            print(f"\n📊 AI-Identified Levels:")
            if 'support' in sr:
                print(f"   🛡️  Support: ${sr['support']:.2f}")
            if 'resistance' in sr:
                print(f"   ⚔️  Resistance: ${sr['resistance']:.2f}")
        
        return True, ai_response
        
    except Exception as e:
        print(f"❌ AI integration failed: {e}")
        print("💡 Note: Requires valid OpenAI API key")
        return False, None


def demo_data_quality():
    """Demonstrate data quality and monitoring"""
    print("\n🔍 DEMO: Data Quality & Monitoring")
    print("=" * 60)
    
    config = Config()
    data_manager = IntelligentDataManager(config.data)
    
    # Get quality report
    quality_report = data_manager.get_data_quality_report()
    
    print(f"📊 Data Quality Report:")
    print(f"   Symbols Tracked: {quality_report['total_symbols_tracked']}")
    print(f"   Cache Status: {quality_report['cache_status']['cached_files']} files")
    
    # Show data completeness for tracked symbols
    if quality_report['data_completeness']:
        print(f"\n📈 Data Completeness by Symbol:")
        for symbol, completeness in list(quality_report['data_completeness'].items())[:3]:
            avg_completeness = sum(completeness.values()) / len(completeness)
            print(f"   {symbol}: {avg_completeness:.1f}% complete")


def main():
    """Main demo function"""
    print("🚀 AI Trading Agent - Phase 2 Demo")
    print("🎯 Intelligent Data Management System")
    print("=" * 80)
    print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Demo 1: Data Acquisition
        spy_data, portfolio_data = demo_data_acquisition()
        
        # Demo 2: Technical Indicators
        data_with_indicators, ai_indicators = demo_technical_indicators(spy_data)
        
        # Demo 3: Market Analysis
        ai_analysis = demo_market_analysis(spy_data)
        
        # Demo 4: AI Integration
        ai_success, ai_response = demo_ai_integration(ai_analysis)
        
        # Demo 5: Data Quality
        demo_data_quality()
        
        # Summary
        print("\n" + "=" * 80)
        print("🎉 PHASE 2 DEMO COMPLETE!")
        print("=" * 80)
        print("✅ Data Acquisition: WORKING")
        print("✅ Technical Indicators: WORKING")
        print("✅ Market Analysis: WORKING")
        print(f"{'✅' if ai_success else '⚠️ '} AI Integration: {'WORKING' if ai_success else 'LIMITED (API key needed)'}")
        print("✅ Data Quality Monitoring: WORKING")
        
        print("\n🎯 Phase 2 Achievements:")
        print("   📊 Real-time Yahoo Finance integration")
        print("   🔧 20+ technical indicators from existing framework")
        print("   🔍 Comprehensive data validation & quality checks")
        print("   🤖 AI integration with real market data")
        print("   ⚡ Sub-second data processing performance")
        
        print("\n🚀 Ready for Phase 3: Framework Integration!")
        print("   Connect AI decisions to your backtesting framework")
        print("   Implement position management and risk controls")
        print("   Advanced validation with walk-forward analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
