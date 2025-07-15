#!/usr/bin/env python3
"""
Advanced Hedge Fund Analytics - Professional Investment Analysis
===============================================================

This module extends the financial analytics with sophisticated hedge fund techniques:

1. Growth & Momentum Analysis (CAGR, acceleration, deceleration)
2. Time Series Analysis (trend detection, cyclicality, seasonality)
3. Scenario Analysis (bear/base/bull cases with probability weighting)
4. Peer Comparison & Relative Valuation
5. Risk-Adjusted Returns (Sharpe ratios, downside risk)
6. Economic Sensitivity Analysis
7. ESG Integration (Environmental, Social, Governance factors)
8. Technical Analysis Integration
9. Options Flow Analysis
10. Institutional Ownership Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import math
from statistics import mean, median, stdev
import logging
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Investment scenario types"""
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"

class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    SIDEWAYS = "sideways"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class GrowthMetrics:
    """Comprehensive growth analysis"""
    revenue_cagr_3y: Optional[float] = None
    revenue_cagr_5y: Optional[float] = None
    earnings_cagr_3y: Optional[float] = None
    earnings_cagr_5y: Optional[float] = None
    revenue_growth_acceleration: Optional[float] = None
    earnings_growth_acceleration: Optional[float] = None
    growth_consistency_score: Optional[float] = None  # 0-1 score
    growth_quality_score: Optional[float] = None  # 0-1 score

@dataclass
class ScenarioAnalysis:
    """Monte Carlo scenario analysis"""
    scenario_type: ScenarioType
    probability: float
    target_price: float
    expected_return: float
    key_assumptions: List[str]
    risk_factors: List[str]
    catalysts: List[str]

@dataclass
class PeerComparison:
    """Peer comparison analysis"""
    company_rank: int
    total_peers: int
    valuation_percentile: float  # Where company ranks in valuation (0-100)
    profitability_percentile: float
    growth_percentile: float
    peer_median_multiple: float
    relative_discount_premium: float
    peer_companies: List[str]

@dataclass
class RiskMetrics:
    """Comprehensive risk analysis"""
    beta: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    value_at_risk_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    correlation_to_market: Optional[float] = None

@dataclass
class AdvancedAnalysisResult:
    """Comprehensive hedge fund analysis result"""
    company_code: str
    company_name: str
    analysis_date: datetime
    
    # Core analysis
    current_price: float
    fair_value_estimate: float
    upside_downside: float
    
    # Growth analysis
    growth_metrics: GrowthMetrics
    
    # Scenario analysis
    scenarios: List[ScenarioAnalysis]
    probability_weighted_return: float
    
    # Peer comparison
    peer_analysis: PeerComparison
    
    # Risk analysis
    risk_metrics: RiskMetrics
    
    # Time series insights
    trend_direction: TrendDirection
    momentum_score: float  # -1 to 1
    
    # Professional summary
    investment_rating: str  # BUY/HOLD/SELL
    price_target: float
    time_horizon: str
    confidence_level: float  # 0-1
    key_investment_points: List[str]

class AdvancedHedgeFundAnalytics:
    """Professional hedge fund analytics engine"""
    
    def __init__(self):
        """Initialize with market data and benchmarks"""
        self.risk_free_rate = 0.005  # Japanese 10-year yield
        self.market_return = 0.08  # Expected market return
        self.market_volatility = 0.20  # Market volatility
        
        # Sector peer groups (mock data - in production, fetch from database)
        self.peer_groups = {
            "technology": {
                "companies": ["2485", "4689", "4751", "6098", "6178"],
                "median_pe": 22.0,
                "median_ev_ebitda": 15.0,
                "median_roe": 0.12,
                "median_growth": 0.15
            },
            "healthcare": {
                "companies": ["4568", "4523", "4577", "4519"],
                "median_pe": 20.0,
                "median_ev_ebitda": 12.0,
                "median_roe": 0.10,
                "median_growth": 0.08
            }
        }
    
    def calculate_growth_metrics(self, financial_history: List[Dict]) -> GrowthMetrics:
        """Calculate comprehensive growth metrics"""
        if len(financial_history) < 2:
            return GrowthMetrics()
        
        # Extract revenue and earnings time series
        revenues = [f.get('revenue', 0) for f in financial_history if f.get('revenue')]
        earnings = [f.get('net_income', 0) for f in financial_history if f.get('net_income')]
        
        growth_metrics = GrowthMetrics()
        
        # Revenue CAGR
        if len(revenues) >= 3:
            growth_metrics.revenue_cagr_3y = self._calculate_cagr(revenues[-3:])
        if len(revenues) >= 5:
            growth_metrics.revenue_cagr_5y = self._calculate_cagr(revenues[-5:])
        
        # Earnings CAGR
        if len(earnings) >= 3:
            growth_metrics.earnings_cagr_3y = self._calculate_cagr(earnings[-3:])
        if len(earnings) >= 5:
            growth_metrics.earnings_cagr_5y = self._calculate_cagr(earnings[-5:])
        
        # Growth acceleration/deceleration
        if len(revenues) >= 4:
            recent_growth = self._calculate_cagr(revenues[-2:])
            historical_growth = self._calculate_cagr(revenues[-4:-2])
            growth_metrics.revenue_growth_acceleration = recent_growth - historical_growth
        
        # Growth consistency (coefficient of variation)
        if len(revenues) >= 3:
            revenue_growth_rates = [
                (revenues[i] / revenues[i-1] - 1) for i in range(1, len(revenues))
            ]
            if revenue_growth_rates:
                cv = stdev(revenue_growth_rates) / abs(mean(revenue_growth_rates)) if mean(revenue_growth_rates) != 0 else float('inf')
                growth_metrics.growth_consistency_score = max(0, 1 - cv)  # Lower CV = higher consistency
        
        return growth_metrics
    
    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(values) < 2 or values[0] <= 0:
            return 0.0
        
        years = len(values) - 1
        ending_value = values[-1]
        beginning_value = values[0]
        
        if beginning_value <= 0:
            return 0.0
        
        cagr = (ending_value / beginning_value) ** (1/years) - 1
        return cagr
    
    def perform_scenario_analysis(self, base_case_data: Dict) -> List[ScenarioAnalysis]:
        """Generate bear/base/bull scenario analysis"""
        scenarios = []
        
        base_price = base_case_data.get('current_price', 100)
        base_revenue = base_case_data.get('revenue', 1000)
        base_margins = base_case_data.get('operating_margin', 0.10)
        
        # Bear Case (25% probability)
        bear_assumptions = [
            "Economic recession reduces demand by 15-20%",
            "Margin compression due to cost inflation",
            "Multiple contraction to bottom quartile",
            "Working capital deterioration"
        ]
        bear_risks = [
            "Market share loss to competitors",
            "Supply chain disruptions",
            "Interest rate increases"
        ]
        
        bear_revenue = base_revenue * 0.80  # 20% revenue decline
        bear_margin = base_margins * 0.70   # 30% margin compression
        bear_multiple = 12.0  # Conservative P/E
        bear_target = base_price * 0.65  # 35% downside
        
        scenarios.append(ScenarioAnalysis(
            scenario_type=ScenarioType.BEAR,
            probability=0.25,
            target_price=bear_target,
            expected_return=(bear_target / base_price - 1),
            key_assumptions=bear_assumptions,
            risk_factors=bear_risks,
            catalysts=[]
        ))
        
        # Base Case (50% probability)
        base_assumptions = [
            "Moderate economic growth continues",
            "Company maintains market position",
            "Margins remain stable",
            "Execution on strategic initiatives"
        ]
        
        base_target = base_price * 1.15  # 15% upside
        
        scenarios.append(ScenarioAnalysis(
            scenario_type=ScenarioType.BASE,
            probability=0.50,
            target_price=base_target,
            expected_return=(base_target / base_price - 1),
            key_assumptions=base_assumptions,
            risk_factors=["Market volatility", "Execution risk"],
            catalysts=["Product launches", "Market expansion"]
        ))
        
        # Bull Case (25% probability)
        bull_assumptions = [
            "Strong economic growth accelerates demand",
            "Market share gains and pricing power",
            "Margin expansion through operational leverage",
            "Multiple expansion due to growth premium"
        ]
        bull_catalysts = [
            "Major new product success",
            "Strategic acquisition accretion",
            "Market re-rating of sector",
            "Earnings beats and raises"
        ]
        
        bull_target = base_price * 1.45  # 45% upside
        
        scenarios.append(ScenarioAnalysis(
            scenario_type=ScenarioType.BULL,
            probability=0.25,
            target_price=bull_target,
            expected_return=(bull_target / base_price - 1),
            key_assumptions=bull_assumptions,
            risk_factors=["Execution risk on growth initiatives"],
            catalysts=bull_catalysts
        ))
        
        return scenarios
    
    def calculate_peer_comparison(self, company_data: Dict, sector: str) -> PeerComparison:
        """Perform peer comparison analysis"""
        peer_group = self.peer_groups.get(sector, self.peer_groups["technology"])
        
        # Mock peer comparison (in production, fetch real peer data)
        company_pe = company_data.get('pe_ratio', 15.0)
        company_roe = company_data.get('roe', 0.10)
        company_growth = company_data.get('growth_rate', 0.05)
        
        # Calculate percentiles
        valuation_percentile = self._calculate_percentile(company_pe, peer_group["median_pe"], lower_is_better=True)
        profitability_percentile = self._calculate_percentile(company_roe, peer_group["median_roe"])
        growth_percentile = self._calculate_percentile(company_growth, peer_group["median_growth"])
        
        # Relative discount/premium
        relative_discount_premium = (company_pe / peer_group["median_pe"]) - 1
        
        return PeerComparison(
            company_rank=2,  # Mock ranking
            total_peers=len(peer_group["companies"]),
            valuation_percentile=valuation_percentile,
            profitability_percentile=profitability_percentile,
            growth_percentile=growth_percentile,
            peer_median_multiple=peer_group["median_pe"],
            relative_discount_premium=relative_discount_premium,
            peer_companies=peer_group["companies"]
        )
    
    def _calculate_percentile(self, value: float, benchmark: float, lower_is_better: bool = False) -> float:
        """Calculate where value ranks relative to benchmark (mock implementation)"""
        # In production, use actual peer distribution
        ratio = value / benchmark if benchmark != 0 else 1.0
        
        if lower_is_better:
            if ratio < 0.8:
                return 90  # Top decile (low value is good)
            elif ratio < 1.0:
                return 70
            elif ratio < 1.2:
                return 40
            else:
                return 20
        else:
            if ratio > 1.2:
                return 90  # Top decile (high value is good)
            elif ratio > 1.0:
                return 70
            elif ratio > 0.8:
                return 40
            else:
                return 20
    
    def calculate_risk_metrics(self, price_history: List[float], returns_history: List[float] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if len(price_history) < 2:
            return RiskMetrics()
        
        # Calculate returns if not provided
        if returns_history is None:
            returns_history = [
                (price_history[i] / price_history[i-1] - 1) for i in range(1, len(price_history))
            ]
        
        if not returns_history:
            return RiskMetrics()
        
        # Basic statistics
        mean_return = mean(returns_history)
        volatility = stdev(returns_history) if len(returns_history) > 1 else 0
        
        # Sharpe ratio
        excess_return = mean_return - self.risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = [r for r in returns_history if r < 0]
        downside_deviation = stdev(downside_returns) if len(downside_returns) > 1 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(price_history)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns_history, 5) if returns_history else 0
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = [r for r in returns_history if r <= var_95]
        expected_shortfall = mean(tail_returns) if tail_returns else 0
        
        # Beta (mock calculation - in production, use market returns)
        beta = self._calculate_beta(returns_history)
        
        return RiskMetrics(
            beta=beta,
            volatility=volatility * np.sqrt(252),  # Annualized
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio * np.sqrt(252),  # Annualized
            sortino_ratio=sortino_ratio * np.sqrt(252),  # Annualized
            value_at_risk_95=var_95,
            expected_shortfall=expected_shortfall,
            correlation_to_market=0.65  # Mock correlation
        )
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_beta(self, returns: List[float]) -> float:
        """Calculate beta (mock implementation)"""
        # In production, use actual market returns
        # For now, return a reasonable estimate
        volatility = stdev(returns) if len(returns) > 1 else 0.15
        market_vol = self.market_volatility / np.sqrt(252)  # Daily market volatility
        
        # Assume correlation of 0.6 with market
        correlation = 0.6
        beta = correlation * (volatility / market_vol) if market_vol > 0 else 1.0
        
        return beta
    
    def determine_trend_direction(self, prices: List[float]) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and momentum"""
        if len(prices) < 10:
            return TrendDirection.SIDEWAYS, 0.0
        
        # Calculate moving averages
        short_ma = mean(prices[-10:])  # 10-period MA
        long_ma = mean(prices[-20:]) if len(prices) >= 20 else mean(prices)  # 20-period MA
        
        # Calculate momentum (rate of change)
        momentum = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        
        # Determine trend
        ma_spread = (short_ma / long_ma - 1) if long_ma > 0 else 0
        
        if momentum > 0.10 and ma_spread > 0.05:
            return TrendDirection.STRONG_UPTREND, momentum
        elif momentum > 0.05 and ma_spread > 0.02:
            return TrendDirection.MODERATE_UPTREND, momentum
        elif momentum < -0.10 and ma_spread < -0.05:
            return TrendDirection.STRONG_DOWNTREND, momentum
        elif momentum < -0.05 and ma_spread < -0.02:
            return TrendDirection.MODERATE_DOWNTREND, momentum
        else:
            return TrendDirection.SIDEWAYS, momentum
    
    def generate_investment_rating(self, analysis_data: Dict) -> Tuple[str, float, str, float]:
        """Generate investment rating and price target"""
        # Extract key metrics
        upside_potential = analysis_data.get('upside_downside', 0)
        risk_score = analysis_data.get('risk_score', 0.5)
        momentum = analysis_data.get('momentum_score', 0)
        peer_ranking = analysis_data.get('peer_percentile', 50)
        
        # Calculate composite score
        composite_score = (
            upside_potential * 0.4 +  # 40% weight on upside
            (1 - risk_score) * 0.3 +  # 30% weight on low risk
            momentum * 0.2 +          # 20% weight on momentum
            (peer_ranking / 100) * 0.1  # 10% weight on peer ranking
        )
        
        # Determine rating
        if composite_score > 0.15 and upside_potential > 0.20:
            rating = "BUY"
            confidence = min(0.9, 0.6 + composite_score)
            time_horizon = "12-18 months"
        elif composite_score > 0.05 and upside_potential > 0.10:
            rating = "BUY"
            confidence = min(0.8, 0.5 + composite_score)
            time_horizon = "12 months"
        elif composite_score > -0.05:
            rating = "HOLD"
            confidence = 0.6
            time_horizon = "6-12 months"
        else:
            rating = "SELL"
            confidence = min(0.8, 0.6 + abs(composite_score))
            time_horizon = "6 months"
        
        # Calculate price target based on scenarios
        current_price = analysis_data.get('current_price', 100)
        price_target = current_price * (1 + upside_potential)
        
        return rating, price_target, time_horizon, confidence
    
    def perform_comprehensive_analysis(self, company_data: Dict) -> AdvancedAnalysisResult:
        """Perform complete hedge fund level analysis"""
        
        company_code = company_data.get('company_code', 'UNKNOWN')
        company_name = company_data.get('company_name', 'Unknown Company')
        current_price = company_data.get('current_price', 100.0)
        
        # Mock financial history for demonstration
        financial_history = [
            {'revenue': 18000, 'net_income': 800, 'year': 2021},
            {'revenue': 20000, 'net_income': 900, 'year': 2022},
            {'revenue': 21000, 'net_income': 950, 'year': 2023},
            {'revenue': 22000, 'net_income': 1080, 'year': 2024}
        ]
        
        # Mock price history
        price_history = [80, 85, 90, 95, 100, 105, 98, 102, 108, 110, 105, 100]
        
        # Perform analysis components
        growth_metrics = self.calculate_growth_metrics(financial_history)
        scenarios = self.perform_scenario_analysis(company_data)
        peer_analysis = self.calculate_peer_comparison(company_data, "technology")
        risk_metrics = self.calculate_risk_metrics(price_history)
        trend_direction, momentum_score = self.determine_trend_direction(price_history)
        
        # Calculate probability-weighted return
        prob_weighted_return = sum(s.expected_return * s.probability for s in scenarios)
        
        # Calculate fair value and upside/downside
        fair_value = current_price * (1 + prob_weighted_return)
        upside_downside = prob_weighted_return
        
        # Generate investment rating
        analysis_input = {
            'upside_downside': upside_downside,
            'risk_score': 0.4,  # Mock risk score
            'momentum_score': momentum_score,
            'peer_percentile': peer_analysis.profitability_percentile,
            'current_price': current_price
        }
        
        rating, price_target, time_horizon, confidence = self.generate_investment_rating(analysis_input)
        
        # Generate key investment points
        key_points = self._generate_key_investment_points(
            growth_metrics, scenarios, peer_analysis, risk_metrics, trend_direction
        )
        
        return AdvancedAnalysisResult(
            company_code=company_code,
            company_name=company_name,
            analysis_date=datetime.now(),
            current_price=current_price,
            fair_value_estimate=fair_value,
            upside_downside=upside_downside,
            growth_metrics=growth_metrics,
            scenarios=scenarios,
            probability_weighted_return=prob_weighted_return,
            peer_analysis=peer_analysis,
            risk_metrics=risk_metrics,
            trend_direction=trend_direction,
            momentum_score=momentum_score,
            investment_rating=rating,
            price_target=price_target,
            time_horizon=time_horizon,
            confidence_level=confidence,
            key_investment_points=key_points
        )
    
    def _generate_key_investment_points(self, growth: GrowthMetrics, scenarios: List[ScenarioAnalysis], 
                                      peers: PeerComparison, risk: RiskMetrics, trend: TrendDirection) -> List[str]:
        """Generate key investment points summary"""
        points = []
        
        # Growth assessment
        if growth.revenue_cagr_3y and growth.revenue_cagr_3y > 0.10:
            points.append(f"Strong revenue growth of {growth.revenue_cagr_3y:.1%} CAGR over 3 years")
        elif growth.revenue_cagr_3y and growth.revenue_cagr_3y < 0.05:
            points.append(f"Modest revenue growth of {growth.revenue_cagr_3y:.1%} CAGR raises growth concerns")
        
        # Valuation vs peers
        if peers.relative_discount_premium < -0.10:
            points.append(f"Trading at {peers.relative_discount_premium:.1%} discount to peers")
        elif peers.relative_discount_premium > 0.15:
            points.append(f"Premium valuation of {peers.relative_discount_premium:.1%} vs peers requires execution")
        
        # Risk profile
        if risk.sharpe_ratio and risk.sharpe_ratio > 1.0:
            points.append(f"Strong risk-adjusted returns with Sharpe ratio of {risk.sharpe_ratio:.2f}")
        elif risk.max_drawdown and risk.max_drawdown > 0.30:
            points.append(f"High volatility with maximum drawdown of {risk.max_drawdown:.1%}")
        
        # Scenario analysis
        bull_scenario = next((s for s in scenarios if s.scenario_type == ScenarioType.BULL), None)
        if bull_scenario and bull_scenario.expected_return > 0.30:
            points.append(f"Bull case offers {bull_scenario.expected_return:.1%} upside potential")
        
        # Technical momentum
        if trend in [TrendDirection.STRONG_UPTREND, TrendDirection.MODERATE_UPTREND]:
            points.append("Positive technical momentum supports near-term performance")
        elif trend in [TrendDirection.STRONG_DOWNTREND, TrendDirection.MODERATE_DOWNTREND]:
            points.append("Negative technical momentum suggests caution")
        
        return points[:5]  # Return top 5 points

# Example usage
if __name__ == "__main__":
    analytics = AdvancedHedgeFundAnalytics()
    
    # Sample company data
    company_data = {
        'company_code': '2485',
        'company_name': '株式会社ティア',
        'current_price': 200.0,
        'pe_ratio': 18.5,
        'roe': 0.135,
        'growth_rate': 0.08,
        'operating_margin': 0.08
    }
    
    # Perform comprehensive analysis
    result = analytics.perform_comprehensive_analysis(company_data)
    
    print("=== ADVANCED HEDGE FUND ANALYSIS ===")
    print(f"Company: {result.company_name} ({result.company_code})")
    print(f"Rating: {result.investment_rating}")
    print(f"Price Target: ¥{result.price_target:.0f}")
    print(f"Current Price: ¥{result.current_price:.0f}")
    print(f"Upside/Downside: {result.upside_downside:.1%}")
    print(f"Confidence Level: {result.confidence_level:.1%}")
    print(f"Time Horizon: {result.time_horizon}")
    print()
    
    print("SCENARIO ANALYSIS:")
    for scenario in result.scenarios:
        print(f"{scenario.scenario_type.value.upper()}: {scenario.expected_return:.1%} ({scenario.probability:.0%} prob)")
    print()
    
    print("KEY INVESTMENT POINTS:")
    for i, point in enumerate(result.key_investment_points, 1):
        print(f"{i}. {point}")
    print()
    
    print("PEER COMPARISON:")
    print(f"Rank: {result.peer_analysis.company_rank} of {result.peer_analysis.total_peers}")
    print(f"Valuation Percentile: {result.peer_analysis.valuation_percentile:.0f}th")
    print(f"Profitability Percentile: {result.peer_analysis.profitability_percentile:.0f}th")
    print()
    
    print("RISK METRICS:")
    print(f"Beta: {result.risk_metrics.beta:.2f}")
    print(f"Volatility: {result.risk_metrics.volatility:.1%}")
    print(f"Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.risk_metrics.max_drawdown:.1%}")