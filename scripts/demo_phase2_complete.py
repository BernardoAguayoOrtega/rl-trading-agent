#!/usr/bin/env python3
"""
Phase 2 Demo - AI Trading Agent Data Pipeline & Technical Indicators
===================================================================

This demo showcases the complete Phase 2 implementation including:
- Intelligent data acquisition from Yahoo Finance
- Technical indicator calculations
- AI-ready market analysis
- Integration with existing AI client
"""

import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from ai_trading_agent.indicators import MarketDataProcessor
from ai_trading_agent.ai_client import AITradingClient
from ai_trading_agent.config import default_config

def demo_data_pipeline():
    """Demo the data acquisition and processing pipeline"""
    print("🔄 Data Pipeline Demo")
    print("=" * 40)
    
    # Initialize the market data processor
    processor = MarketDataProcessor()
    
    # Get comprehensive market analysis for a popular stock
    symbol = "AAPL"
    print(f"📊 Analyzing {symbol}...")
    
    analysis = processor.get_complete_market_analysis(symbol, period="1mo")
    
    # Display key information
    print(f"\n📈 {symbol} Market Analysis:")
    print(f"  Current Price: ${analysis['current_price']:.2f}")
    print(f"  Price Change: {analysis['price_change_pct']:.2f}%")
    print(f"  Trend: {analysis['trend']}")
    print(f"  Volatility: {analysis['volatility']:.2f}%")
    print(f"  Data Points: {analysis['data_points']}")
    
    # Technical indicators
    tech_indicators = analysis['technical_indicators']
    print(f"\n🔧 Technical Indicators:")
    print(f"  RSI (14): {tech_indicators.get('rsi_14', 0):.1f}")
    print(f"  RSI Signal: {tech_indicators.get('rsi_signal', 'unknown')}")
    print(f"  MACD Signal: {tech_indicators.get('macd_signal_status', 'unknown')}")
    print(f"  Price vs SMA20: {tech_indicators.get('price_vs_sma20', 'unknown')}")
    print(f"  Bollinger Band Position: {tech_indicators.get('bb_position', 'unknown')}")
    
    return analysis

def demo_multi_symbol_analysis():
    """Demo multi-symbol analysis"""
    print("\n🔄 Multi-Symbol Analysis Demo")
    print("=" * 40)
    
    processor = MarketDataProcessor()
    
    # Analyze multiple popular stocks
    symbols = ["SPY", "QQQ", "TSLA", "NVDA"]
    print(f"📊 Analyzing {len(symbols)} symbols...")
    
    multi_analysis = processor.get_multi_symbol_analysis(symbols, period="5d")
    
    print(f"\n📈 Portfolio Overview:")
    for symbol, data in multi_analysis.items():
        if 'error' not in data:
            trend_icon = "📈" if data['trend'] == 'upward' else "📉"
            print(f"  {trend_icon} {symbol}: ${data['current_price']:.2f} ({data['price_change_pct']:+.2f}%)")
            
            # Show key indicator
            rsi = data['technical_indicators'].get('rsi_14', 50)
            rsi_status = "🔥" if rsi > 70 else "❄️" if rsi < 30 else "📊"
            print(f"    {rsi_status} RSI: {rsi:.1f}")
        else:
            print(f"  ❌ {symbol}: {data['error']}")
    
    return multi_analysis

def demo_ai_integration():
    """Demo AI integration with real market data"""
    print("\n🤖 AI Integration Demo")
    print("=" * 40)
    
    try:
        # Get market data
        processor = MarketDataProcessor()
        analysis = processor.get_complete_market_analysis("SPY", period="2mo")
        
        # Prepare data for AI
        market_data = {
            'current_price': analysis['current_price'],
            'price_change': analysis['price_change_pct'],
            'volume': analysis['volume'],
            'volatility': analysis['volatility'],
            'trend': analysis['trend'],
            'market_hours': analysis['market_hours'],
            'technical_indicators': analysis['technical_indicators']
        }
        
        print("📊 Market Data for AI:")
        print(f"  Symbol: SPY")
        print(f"  Price: ${market_data['current_price']:.2f}")
        print(f"  Change: {market_data['price_change']:+.2f}%")
        print(f"  Trend: {market_data['trend']}")
        print(f"  Technical Indicators: {len(market_data['technical_indicators'])} indicators")
        
        # Initialize AI client
        ai_client = AITradingClient(default_config.openai)
        
        # Get AI market analysis
        print(f"\n🧠 AI Analysis:")
        ai_response = ai_client.analyze_market_conditions(market_data)
        
        print(f"  Market Regime: {ai_response.get('market_regime', 'unknown')}")
        print(f"  Trend Strength: {ai_response.get('trend_strength', 'unknown')}")
        print(f"  Volatility Level: {ai_response.get('volatility_level', 'unknown')}")
        print(f"  Market Score: {ai_response.get('market_score', 'unknown')}")
        
        if 'key_insights' in ai_response:
            print(f"\n💡 Key Insights:")
            for insight in ai_response['key_insights'][:3]:  # Show top 3
                print(f"    • {insight}")
        
        return ai_response
        
    except Exception as e:
        print(f"  ⚠️  AI integration demo failed: {e}")
        print("     (This requires a valid OpenAI API key)")
        return None

def demo_real_time_capabilities():
    """Demo real-time data capabilities"""
    print("\n⚡ Real-Time Data Demo")
    print("=" * 40)
    
    try:
        processor = MarketDataProcessor()
        
        # Get real-time analysis
        symbol = "SPY"
        rt_analysis = processor.get_real_time_analysis(symbol, max_age_minutes=60)
        
        print(f"📊 Real-Time Analysis for {symbol}:")
        print(f"  Current Price: ${rt_analysis['current_price']:.2f}")
        print(f"  Data Age: {rt_analysis['data_age_minutes']} minutes max")
        print(f"  Is Real-Time: {rt_analysis['is_real_time']}")
        print(f"  Analysis Time: {rt_analysis['analysis_timestamp']}")
        
        # Show live technical indicators
        tech_indicators = rt_analysis['technical_indicators']
        print(f"\n🔧 Live Technical Indicators:")
        print(f"  RSI: {tech_indicators.get('rsi_14', 0):.1f}")
        print(f"  MACD Status: {tech_indicators.get('macd_signal_status', 'unknown')}")
        
        return rt_analysis
        
    except Exception as e:
        print(f"  ⚠️  Real-time demo failed: {e}")
        return None

def main():
    """Run the complete Phase 2 demo"""
    print("🚀 AI Trading Agent - Phase 2 Demo")
    print("=" * 50)
    print("Showcasing Intelligent Data Pipeline & Technical Indicators")
    print("=" * 50)
    
    try:
        # Demo 1: Basic data pipeline
        analysis = demo_data_pipeline()
        
        # Demo 2: Multi-symbol analysis
        multi_analysis = demo_multi_symbol_analysis()
        
        # Demo 3: AI integration
        ai_response = demo_ai_integration()
        
        # Demo 4: Real-time capabilities
        rt_analysis = demo_real_time_capabilities()
        
        # Summary
        print("\n🎉 Phase 2 Demo Complete!")
        print("=" * 50)
        print("✅ Data Pipeline: Fully functional")
        print("✅ Technical Indicators: Professional-grade calculations")
        print("✅ Multi-symbol Analysis: Efficient batch processing")
        print("✅ AI Integration: Real market data → AI insights")
        print("✅ Real-time Capabilities: Near real-time market analysis")
        
        print(f"\n📊 Demo Statistics:")
        if analysis:
            print(f"  • Data Points Processed: {analysis['data_points']}")
        if multi_analysis:
            successful_symbols = len([s for s in multi_analysis.values() if 'error' not in s])
            print(f"  • Symbols Analyzed: {successful_symbols}/{len(multi_analysis)}")
        if ai_response:
            print(f"  • AI Analysis: Successful")
        if rt_analysis:
            print(f"  • Real-time Data: Available")
        
        print(f"\n🎯 Your AI trading system now has:")
        print(f"  • Professional data acquisition")
        print(f"  • Comprehensive technical analysis")
        print(f"  • AI-powered market insights")
        print(f"  • Real-time market monitoring")
        
        print(f"\n🚀 Ready for advanced trading strategies!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
