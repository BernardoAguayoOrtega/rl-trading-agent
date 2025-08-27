"""
OpenAI Client Integration for Trading Decisions
==============================================

This module provides a clean interface to OpenAI's API for trading-related
decision making, market analysis, and strategy development.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from .config import OpenAIConfig, default_config

logger = logging.getLogger(__name__)


class AITradingClient:
    """OpenAI client specifically designed for trading applications"""
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize the AI trading client"""
        self.config = config or default_config.openai
        self.client = OpenAI(api_key=self.config.api_key)
        
        # System prompts for different types of analysis
        self.system_prompts = {
            'market_analysis': """You are an expert quantitative analyst with deep knowledge of financial markets, 
            technical analysis, and market microstructure. Provide objective, data-driven analysis 
            focusing on actionable insights for algorithmic trading.""",
            
            'strategy_development': """You are a professional algorithmic trading strategist. 
            Your role is to develop and optimize trading strategies based on market conditions, 
            risk parameters, and performance metrics. Focus on practical, implementable strategies.""",
            
            'risk_assessment': """You are a risk management expert specializing in algorithmic trading. 
            Evaluate potential risks, calculate risk metrics, and provide risk-adjusted recommendations 
            for trading decisions.""",
            
            'decision_making': """You are a professional trading AI making real-time trading decisions. 
            Combine technical analysis, market conditions, and risk assessment to provide clear, 
            actionable trading recommendations with confidence levels."""
        }
    
    def analyze_market_conditions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions and provide insights"""
        prompt = self._create_market_analysis_prompt(data)
        
        response = self._make_request(
            prompt=prompt,
            system_prompt=self.system_prompts['market_analysis'],
            temperature=0.1
        )
        
        return self._parse_market_analysis_response(response)
    
    def develop_strategy(self, market_analysis: Dict[str, Any], 
                        performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Develop or adapt trading strategy based on market conditions"""
        prompt = self._create_strategy_development_prompt(market_analysis, performance_data)
        
        response = self._make_request(
            prompt=prompt,
            system_prompt=self.system_prompts['strategy_development'],
            temperature=0.2
        )
        
        return self._parse_strategy_response(response)
    
    def assess_risk(self, position_data: Dict[str, Any], 
                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for potential trading decisions"""
        prompt = self._create_risk_assessment_prompt(position_data, market_conditions)
        
        response = self._make_request(
            prompt=prompt,
            system_prompt=self.system_prompts['risk_assessment'],
            temperature=0.1
        )
        
        return self._parse_risk_assessment_response(response)
    
    def make_trading_decision(self, market_data: Dict[str, Any], 
                            strategy_params: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Make final trading decision based on all available information"""
        prompt = self._create_trading_decision_prompt(market_data, strategy_params, risk_assessment)
        
        response = self._make_request(
            prompt=prompt,
            system_prompt=self.system_prompts['decision_making'],
            temperature=0.1
        )
        
        return self._parse_trading_decision_response(response)
    
    def _make_request(self, prompt: str, system_prompt: str, 
                     temperature: Optional[float] = None) -> str:
        """Make request to OpenAI API with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise AITradingClientError(f"Failed to get AI response: {e}")
    
    def _create_market_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for market analysis"""
        return f"""
        Analyze the current market conditions based on the following data:
        
        PRICE DATA:
        - Current Price: ${data.get('current_price', 'N/A')}
        - Price Change: {data.get('price_change', 'N/A')}%
        - Volume: {data.get('volume', 'N/A')}
        
        TECHNICAL INDICATORS:
        {self._format_technical_indicators(data.get('technical_indicators', {}))}
        
        MARKET CONTEXT:
        - Market Hours: {data.get('market_hours', 'N/A')}
        - Volatility: {data.get('volatility', 'N/A')}
        - Trend: {data.get('trend', 'N/A')}
        
        Please provide a comprehensive market analysis in JSON format:
        {{
            "market_regime": "trending|ranging|volatile",
            "trend_strength": 0.0-1.0,
            "volatility_level": "low|medium|high",
            "support_resistance": {{"support": price, "resistance": price}},
            "momentum": "bullish|bearish|neutral",
            "key_insights": ["insight1", "insight2", "insight3"],
            "market_score": 0.0-1.0
        }}
        """
    
    def _create_strategy_development_prompt(self, market_analysis: Dict[str, Any], 
                                          performance_data: Optional[Dict[str, Any]]) -> str:
        """Create prompt for strategy development"""
        performance_section = ""
        if performance_data:
            performance_section = f"""
            RECENT PERFORMANCE:
            - Win Rate: {performance_data.get('win_rate', 'N/A')}%
            - Avg Return: {performance_data.get('avg_return', 'N/A')}%
            - Sharpe Ratio: {performance_data.get('sharpe_ratio', 'N/A')}
            - Max Drawdown: {performance_data.get('max_drawdown', 'N/A')}%
            """
        
        return f"""
        Based on the following market analysis, develop an optimal trading strategy:
        
        MARKET ANALYSIS:
        {json.dumps(market_analysis, indent=2)}
        
        {performance_section}
        
        Please recommend a trading strategy in JSON format:
        {{
            "strategy_name": "strategy_name",
            "entry_conditions": ["condition1", "condition2"],
            "exit_conditions": ["condition1", "condition2"],
            "position_sizing": {{
                "base_size": 0.0-1.0,
                "confidence_multiplier": 0.0-2.0,
                "risk_adjustment": 0.0-1.0
            }},
            "stop_loss": 0.0-0.1,
            "take_profit": 0.0-0.2,
            "timeframe": "1min|5min|15min|1h|1d",
            "confidence": 0.0-1.0,
            "reasoning": "detailed explanation"
        }}
        """
    
    def _create_risk_assessment_prompt(self, position_data: Dict[str, Any], 
                                     market_conditions: Dict[str, Any]) -> str:
        """Create prompt for risk assessment"""
        return f"""
        Assess the risk for the following trading position:
        
        POSITION DATA:
        {json.dumps(position_data, indent=2)}
        
        MARKET CONDITIONS:
        {json.dumps(market_conditions, indent=2)}
        
        Please provide a comprehensive risk assessment in JSON format:
        {{
            "overall_risk_score": 0.0-1.0,
            "risk_factors": ["factor1", "factor2", "factor3"],
            "position_risk": 0.0-1.0,
            "market_risk": 0.0-1.0,
            "liquidity_risk": 0.0-1.0,
            "recommended_position_size": 0.0-1.0,
            "stop_loss_recommendation": 0.0-0.1,
            "risk_reward_ratio": 1.0-5.0,
            "max_acceptable_loss": 0.0-0.1,
            "risk_mitigation": ["action1", "action2"]
        }}
        """
    
    def _create_trading_decision_prompt(self, market_data: Dict[str, Any],
                                      strategy_params: Dict[str, Any],
                                      risk_assessment: Dict[str, Any]) -> str:
        """Create prompt for final trading decision"""
        return f"""
        Make a trading decision based on the comprehensive analysis:
        
        MARKET DATA:
        {json.dumps(market_data, indent=2)}
        
        STRATEGY PARAMETERS:
        {json.dumps(strategy_params, indent=2)}
        
        RISK ASSESSMENT:
        {json.dumps(risk_assessment, indent=2)}
        
        Please provide your trading decision in this EXACT JSON format:
        {{
            "action": "BUY|SELL|HOLD",
            "confidence": 0.0-1.0,
            "position_size": 0.0-1.0,
            "entry_price": price_level,
            "stop_loss": price_level,
            "take_profit": price_level,
            "holding_period": "1-10 days",
            "risk_reward_ratio": 1.0-5.0,
            "reasoning": "detailed explanation of decision",
            "key_factors": ["factor1", "factor2", "factor3"],
            "market_outlook": "bullish|bearish|neutral"
        }}
        """
    
    def _format_technical_indicators(self, indicators: Dict[str, Any]) -> str:
        """Format technical indicators for prompt"""
        if not indicators:
            return "No technical indicators provided"
        
        formatted = []
        for indicator, value in indicators.items():
            formatted.append(f"- {indicator}: {value}")
        
        return "\n".join(formatted)
    
    def _parse_market_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse market analysis response from AI"""
        return self._parse_json_response(response, "market analysis")
    
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse strategy development response from AI"""
        return self._parse_json_response(response, "strategy development")
    
    def _parse_risk_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse risk assessment response from AI"""
        return self._parse_json_response(response, "risk assessment")
    
    def _parse_trading_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse trading decision response from AI"""
        return self._parse_json_response(response, "trading decision")
    
    def _parse_json_response(self, response: str, response_type: str) -> Dict[str, Any]:
        """Parse JSON response with error handling"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse {response_type} response: {e}")
            logger.error(f"Raw response: {response}")
            
            # Return default response based on type
            return self._get_default_response(response_type)
    
    def _get_default_response(self, response_type: str) -> Dict[str, Any]:
        """Get default response when parsing fails"""
        defaults = {
            "market analysis": {
                "market_regime": "uncertain",
                "trend_strength": 0.5,
                "volatility_level": "medium",
                "support_resistance": {"support": 0, "resistance": 0},
                "momentum": "neutral",
                "key_insights": ["Unable to parse AI response"],
                "market_score": 0.5
            },
            "strategy development": {
                "strategy_name": "Conservative Hold",
                "entry_conditions": ["High confidence signal"],
                "exit_conditions": ["Stop loss or take profit"],
                "position_sizing": {
                    "base_size": 0.05,
                    "confidence_multiplier": 1.0,
                    "risk_adjustment": 0.8
                },
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "timeframe": "1d",
                "confidence": 0.3,
                "reasoning": "Default strategy due to parsing error"
            },
            "risk assessment": {
                "overall_risk_score": 0.8,
                "risk_factors": ["Unknown market conditions"],
                "position_risk": 0.8,
                "market_risk": 0.8,
                "liquidity_risk": 0.5,
                "recommended_position_size": 0.05,
                "stop_loss_recommendation": 0.02,
                "risk_reward_ratio": 2.0,
                "max_acceptable_loss": 0.02,
                "risk_mitigation": ["Reduce position size", "Increase monitoring"]
            },
            "trading decision": {
                "action": "HOLD",
                "confidence": 0.3,
                "position_size": 0.0,
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "holding_period": "0 days",
                "risk_reward_ratio": 0,
                "reasoning": "Default hold due to parsing error",
                "key_factors": ["Unable to parse AI response"],
                "market_outlook": "neutral"
            }
        }
        
        return defaults.get(response_type, {})


class AITradingClientError(Exception):
    """Custom exception for AI trading client errors"""
    pass
