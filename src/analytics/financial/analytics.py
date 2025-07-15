#!/usr/bin/env python3
"""
Advanced Financial Analytics for Hedge Fund Level Analysis
========================================================

This module implements sophisticated financial metrics and analysis techniques
used by professional hedge fund analysts and institutional investors.

Categories:
1. Valuation Metrics (P/E, EV/EBITDA, P/B, PEG, etc.)
2. Profitability Metrics (ROE, ROA, ROIC, margins, etc.)
3. Leverage & Solvency Metrics (debt ratios, coverage ratios)
4. Growth & Momentum Metrics (CAGR, acceleration, etc.)
5. Quality Metrics (FCF, earnings quality, working capital)
6. Efficiency Metrics (turnover ratios, operational efficiency)
7. Risk Metrics (volatility, beta, downside risk)
8. Comparative Analysis (peer comparison, sector benchmarks)
9. Time Series Analysis (trends, cyclicality, forecasting)
10. Sector-Specific Metrics (industry-tailored analysis)
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

logger = logging.getLogger(__name__)

class MetricCategory(Enum):
    """Categories of financial metrics"""
    VALUATION = "valuation"
    PROFITABILITY = "profitability"
    LEVERAGE = "leverage"
    GROWTH = "growth"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    RISK = "risk"
    COMPARATIVE = "comparative"
    SECTOR_SPECIFIC = "sector_specific"

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class FinancialData:
    """Comprehensive financial data structure for analysis"""
    # Core Financial Statement Items
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    inventory: Optional[float] = None
    accounts_receivable: Optional[float] = None
    total_debt: Optional[float] = None
    current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    shareholders_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capex: Optional[float] = None
    working_capital: Optional[float] = None
    
    # Market Data
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    stock_price: Optional[float] = None
    book_value_per_share: Optional[float] = None
    
    # Derived Metrics (computed)
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    
    # Metadata
    period: Optional[str] = None
    currency: str = "JPY"
    fiscal_year_end: Optional[date] = None

@dataclass
class MetricResult:
    """Result of a financial metric calculation"""
    name: str
    value: Optional[float]
    category: MetricCategory
    interpretation: str
    benchmark: Optional[float] = None
    percentile: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    confidence: float = 1.0
    calculation_details: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """Comprehensive analysis result"""
    company_code: str
    company_name: str
    analysis_date: datetime
    metrics: List[MetricResult] = field(default_factory=list)
    overall_score: Optional[float] = None
    investment_thesis: str = ""
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    peer_comparison: Optional[Dict[str, Any]] = None

class AdvancedFinancialAnalytics:
    """Hedge fund level financial analysis system"""
    
    def __init__(self):
        self.sector_benchmarks = self._load_sector_benchmarks()
        self.risk_free_rate = 0.005  # Japanese 10-year bond yield ~0.5%
        
    def _load_sector_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load sector-specific benchmark values"""
        return {
            "technology": {
                "pe_ratio": 25.0,
                "ev_ebitda": 18.0,
                "gross_margin": 0.65,
                "operating_margin": 0.20,
                "roe": 0.15,
                "debt_to_equity": 0.30
            },
            "healthcare": {
                "pe_ratio": 22.0,
                "ev_ebitda": 15.0,
                "gross_margin": 0.75,
                "operating_margin": 0.18,
                "roe": 0.12,
                "debt_to_equity": 0.35
            },
            "financial": {
                "pe_ratio": 12.0,
                "ev_ebitda": None,  # Not applicable
                "roe": 0.10,
                "roa": 0.01,
                "tier1_capital_ratio": 0.12
            },
            "manufacturing": {
                "pe_ratio": 15.0,
                "ev_ebitda": 10.0,
                "gross_margin": 0.35,
                "operating_margin": 0.08,
                "roe": 0.10,
                "debt_to_equity": 0.50,
                "asset_turnover": 1.2
            },
            "retail": {
                "pe_ratio": 18.0,
                "ev_ebitda": 12.0,
                "gross_margin": 0.40,
                "operating_margin": 0.06,
                "roe": 0.12,
                "inventory_turnover": 8.0
            }
        }
    
    # 1. VALUATION METRICS
    def calculate_valuation_metrics(self, data: FinancialData) -> List[MetricResult]:
        """Calculate comprehensive valuation metrics"""
        metrics = []
        
        # P/E Ratio
        if data.market_cap and data.net_income and data.net_income > 0:
            pe_ratio = data.market_cap / data.net_income
            interpretation = self._interpret_pe_ratio(pe_ratio)
            metrics.append(MetricResult(
                name="P/E Ratio",
                value=pe_ratio,
                category=MetricCategory.VALUATION,
                interpretation=interpretation,
                calculation_details={"market_cap": data.market_cap, "net_income": data.net_income}
            ))
        
        # EV/EBITDA
        if data.enterprise_value and data.ebitda and data.ebitda > 0:
            ev_ebitda = data.enterprise_value / data.ebitda
            interpretation = self._interpret_ev_ebitda(ev_ebitda)
            metrics.append(MetricResult(
                name="EV/EBITDA",
                value=ev_ebitda,
                category=MetricCategory.VALUATION,
                interpretation=interpretation
            ))
        
        # P/B Ratio
        if data.market_cap and data.shareholders_equity and data.shareholders_equity > 0:
            pb_ratio = data.market_cap / data.shareholders_equity
            interpretation = self._interpret_pb_ratio(pb_ratio)
            metrics.append(MetricResult(
                name="P/B Ratio",
                value=pb_ratio,
                category=MetricCategory.VALUATION,
                interpretation=interpretation
            ))
        
        # P/S Ratio
        if data.market_cap and data.revenue and data.revenue > 0:
            ps_ratio = data.market_cap / data.revenue
            interpretation = self._interpret_ps_ratio(ps_ratio)
            metrics.append(MetricResult(
                name="P/S Ratio",
                value=ps_ratio,
                category=MetricCategory.VALUATION,
                interpretation=interpretation
            ))
        
        # EV/Sales
        if data.enterprise_value and data.revenue and data.revenue > 0:
            ev_sales = data.enterprise_value / data.revenue
            metrics.append(MetricResult(
                name="EV/Sales",
                value=ev_sales,
                category=MetricCategory.VALUATION,
                interpretation=f"Enterprise value is {ev_sales:.1f}x annual sales"
            ))
        
        return metrics
    
    # 2. PROFITABILITY METRICS
    def calculate_profitability_metrics(self, data: FinancialData) -> List[MetricResult]:
        """Calculate comprehensive profitability metrics"""
        metrics = []
        
        # ROE (Return on Equity)
        if data.net_income and data.shareholders_equity and data.shareholders_equity > 0:
            roe = data.net_income / data.shareholders_equity
            interpretation = self._interpret_roe(roe)
            risk_level = self._assess_roe_risk(roe)
            metrics.append(MetricResult(
                name="ROE",
                value=roe,
                category=MetricCategory.PROFITABILITY,
                interpretation=interpretation,
                risk_level=risk_level
            ))
        
        # ROA (Return on Assets)
        if data.net_income and data.total_assets and data.total_assets > 0:
            roa = data.net_income / data.total_assets
            interpretation = self._interpret_roa(roa)
            metrics.append(MetricResult(
                name="ROA",
                value=roa,
                category=MetricCategory.PROFITABILITY,
                interpretation=interpretation
            ))
        
        # ROIC (Return on Invested Capital)
        if data.operating_income and data.shareholders_equity and data.total_debt:
            invested_capital = data.shareholders_equity + data.total_debt
            if invested_capital > 0:
                # Approximate NOPAT (Net Operating Profit After Tax)
                tax_rate = 0.30  # Approximate Japanese corporate tax rate
                nopat = data.operating_income * (1 - tax_rate)
                roic = nopat / invested_capital
                interpretation = self._interpret_roic(roic)
                metrics.append(MetricResult(
                    name="ROIC",
                    value=roic,
                    category=MetricCategory.PROFITABILITY,
                    interpretation=interpretation,
                    calculation_details={"nopat": nopat, "invested_capital": invested_capital}
                ))
        
        # Margin Analysis
        if data.revenue and data.revenue > 0:
            # Gross Margin
            if data.gross_profit:
                gross_margin = data.gross_profit / data.revenue
                metrics.append(MetricResult(
                    name="Gross Margin",
                    value=gross_margin,
                    category=MetricCategory.PROFITABILITY,
                    interpretation=f"Gross margin of {gross_margin:.1%} indicates pricing power and cost efficiency"
                ))
            
            # Operating Margin
            if data.operating_income:
                operating_margin = data.operating_income / data.revenue
                metrics.append(MetricResult(
                    name="Operating Margin",
                    value=operating_margin,
                    category=MetricCategory.PROFITABILITY,
                    interpretation=f"Operating margin of {operating_margin:.1%} shows operational efficiency"
                ))
            
            # Net Margin
            if data.net_income:
                net_margin = data.net_income / data.revenue
                metrics.append(MetricResult(
                    name="Net Margin",
                    value=net_margin,
                    category=MetricCategory.PROFITABILITY,
                    interpretation=f"Net margin of {net_margin:.1%} reflects overall profitability"
                ))
        
        return metrics
    
    # 3. LEVERAGE & SOLVENCY METRICS
    def calculate_leverage_metrics(self, data: FinancialData) -> List[MetricResult]:
        """Calculate leverage and solvency metrics"""
        metrics = []
        
        # Debt-to-Equity Ratio
        if data.total_debt and data.shareholders_equity and data.shareholders_equity > 0:
            debt_to_equity = data.total_debt / data.shareholders_equity
            interpretation = self._interpret_debt_to_equity(debt_to_equity)
            risk_level = self._assess_leverage_risk(debt_to_equity)
            metrics.append(MetricResult(
                name="Debt-to-Equity",
                value=debt_to_equity,
                category=MetricCategory.LEVERAGE,
                interpretation=interpretation,
                risk_level=risk_level
            ))
        
        # Debt-to-Assets Ratio
        if data.total_debt and data.total_assets and data.total_assets > 0:
            debt_to_assets = data.total_debt / data.total_assets
            metrics.append(MetricResult(
                name="Debt-to-Assets",
                value=debt_to_assets,
                category=MetricCategory.LEVERAGE,
                interpretation=f"Debt represents {debt_to_assets:.1%} of total assets"
            ))
        
        # Interest Coverage Ratio (approximated)
        if data.operating_income and data.total_debt:
            # Estimate interest expense as 2% of total debt
            estimated_interest = data.total_debt * 0.02
            if estimated_interest > 0:
                interest_coverage = data.operating_income / estimated_interest
                interpretation = self._interpret_interest_coverage(interest_coverage)
                metrics.append(MetricResult(
                    name="Interest Coverage (Est.)",
                    value=interest_coverage,
                    category=MetricCategory.LEVERAGE,
                    interpretation=interpretation,
                    calculation_details={"estimated_interest": estimated_interest}
                ))
        
        # Current Ratio
        if data.current_assets and data.current_liabilities and data.current_liabilities > 0:
            current_ratio = data.current_assets / data.current_liabilities
            interpretation = self._interpret_current_ratio(current_ratio)
            metrics.append(MetricResult(
                name="Current Ratio",
                value=current_ratio,
                category=MetricCategory.LEVERAGE,
                interpretation=interpretation
            ))
        
        # Quick Ratio (approximated)
        if data.current_assets and data.inventory and data.current_liabilities and data.current_liabilities > 0:
            quick_assets = data.current_assets - data.inventory
            quick_ratio = quick_assets / data.current_liabilities
            metrics.append(MetricResult(
                name="Quick Ratio",
                value=quick_ratio,
                category=MetricCategory.LEVERAGE,
                interpretation=f"Quick ratio of {quick_ratio:.2f} measures immediate liquidity"
            ))
        
        return metrics
    
    # 4. QUALITY METRICS
    def calculate_quality_metrics(self, data: FinancialData) -> List[MetricResult]:
        """Calculate earnings and financial quality metrics"""
        metrics = []
        
        # Free Cash Flow Margin
        if data.free_cash_flow and data.revenue and data.revenue > 0:
            fcf_margin = data.free_cash_flow / data.revenue
            interpretation = self._interpret_fcf_margin(fcf_margin)
            metrics.append(MetricResult(
                name="FCF Margin",
                value=fcf_margin,
                category=MetricCategory.QUALITY,
                interpretation=interpretation
            ))
        
        # FCF to Net Income Ratio (Quality of Earnings)
        if data.free_cash_flow and data.net_income and data.net_income > 0:
            fcf_to_ni = data.free_cash_flow / data.net_income
            interpretation = self._interpret_fcf_to_ni(fcf_to_ni)
            metrics.append(MetricResult(
                name="FCF/Net Income",
                value=fcf_to_ni,
                category=MetricCategory.QUALITY,
                interpretation=interpretation
            ))
        
        # Working Capital Efficiency
        if data.working_capital and data.revenue and data.revenue > 0:
            wc_to_sales = data.working_capital / data.revenue
            metrics.append(MetricResult(
                name="Working Capital/Sales",
                value=wc_to_sales,
                category=MetricCategory.QUALITY,
                interpretation=f"Working capital is {wc_to_sales:.1%} of sales"
            ))
        
        return metrics
    
    # 5. EFFICIENCY METRICS
    def calculate_efficiency_metrics(self, data: FinancialData) -> List[MetricResult]:
        """Calculate operational efficiency metrics"""
        metrics = []
        
        # Asset Turnover
        if data.revenue and data.total_assets and data.total_assets > 0:
            asset_turnover = data.revenue / data.total_assets
            interpretation = self._interpret_asset_turnover(asset_turnover)
            metrics.append(MetricResult(
                name="Asset Turnover",
                value=asset_turnover,
                category=MetricCategory.EFFICIENCY,
                interpretation=interpretation
            ))
        
        # Inventory Turnover (approximated)
        if data.revenue and data.inventory and data.inventory > 0:
            # Use revenue as proxy for COGS
            cogs_estimate = data.revenue * 0.70  # Assume 70% COGS
            inventory_turnover = cogs_estimate / data.inventory
            days_in_inventory = 365 / inventory_turnover
            metrics.append(MetricResult(
                name="Inventory Turnover",
                value=inventory_turnover,
                category=MetricCategory.EFFICIENCY,
                interpretation=f"Inventory turns {inventory_turnover:.1f}x per year ({days_in_inventory:.0f} days)"
            ))
        
        # Receivables Turnover
        if data.revenue and data.accounts_receivable and data.accounts_receivable > 0:
            receivables_turnover = data.revenue / data.accounts_receivable
            days_sales_outstanding = 365 / receivables_turnover
            metrics.append(MetricResult(
                name="Receivables Turnover",
                value=receivables_turnover,
                category=MetricCategory.EFFICIENCY,
                interpretation=f"Collects receivables {receivables_turnover:.1f}x per year ({days_sales_outstanding:.0f} days DSO)"
            ))
        
        return metrics
    
    # INTERPRETATION METHODS
    def _interpret_pe_ratio(self, pe: float) -> str:
        if pe < 10:
            return f"Low P/E of {pe:.1f} suggests undervaluation or poor growth prospects"
        elif pe < 20:
            return f"Moderate P/E of {pe:.1f} indicates reasonable valuation"
        elif pe < 30:
            return f"High P/E of {pe:.1f} suggests growth expectations or overvaluation"
        else:
            return f"Very high P/E of {pe:.1f} indicates strong growth expectations or speculative premium"
    
    def _interpret_ev_ebitda(self, ev_ebitda: float) -> str:
        if ev_ebitda < 8:
            return f"Low EV/EBITDA of {ev_ebitda:.1f} suggests potential undervaluation"
        elif ev_ebitda < 15:
            return f"Moderate EV/EBITDA of {ev_ebitda:.1f} indicates fair valuation"
        else:
            return f"High EV/EBITDA of {ev_ebitda:.1f} suggests premium valuation"
    
    def _interpret_pb_ratio(self, pb: float) -> str:
        if pb < 1:
            return f"P/B below 1.0 ({pb:.1f}) suggests trading below book value"
        elif pb < 3:
            return f"Moderate P/B of {pb:.1f} indicates reasonable valuation relative to assets"
        else:
            return f"High P/B of {pb:.1f} suggests premium to book value"
    
    def _interpret_ps_ratio(self, ps: float) -> str:
        if ps < 1:
            return f"Low P/S of {ps:.1f} suggests potential value opportunity"
        elif ps < 3:
            return f"Moderate P/S of {ps:.1f} indicates reasonable revenue multiple"
        else:
            return f"High P/S of {ps:.1f} suggests premium revenue valuation"
    
    def _interpret_roe(self, roe: float) -> str:
        roe_pct = roe * 100
        if roe_pct < 5:
            return f"Low ROE of {roe_pct:.1f}% indicates poor equity efficiency"
        elif roe_pct < 15:
            return f"Moderate ROE of {roe_pct:.1f}% shows reasonable profitability"
        else:
            return f"Strong ROE of {roe_pct:.1f}% indicates excellent equity efficiency"
    
    def _interpret_roa(self, roa: float) -> str:
        roa_pct = roa * 100
        if roa_pct < 2:
            return f"Low ROA of {roa_pct:.1f}% suggests poor asset utilization"
        elif roa_pct < 8:
            return f"Moderate ROA of {roa_pct:.1f}% indicates reasonable asset efficiency"
        else:
            return f"Strong ROA of {roa_pct:.1f}% shows excellent asset utilization"
    
    def _interpret_roic(self, roic: float) -> str:
        roic_pct = roic * 100
        if roic_pct < 8:
            return f"Low ROIC of {roic_pct:.1f}% suggests value destruction"
        elif roic_pct < 15:
            return f"Moderate ROIC of {roic_pct:.1f}% indicates value creation"
        else:
            return f"Excellent ROIC of {roic_pct:.1f}% shows strong value creation"
    
    def _interpret_debt_to_equity(self, de: float) -> str:
        if de < 0.3:
            return f"Conservative debt level ({de:.2f}) indicates low financial risk"
        elif de < 0.8:
            return f"Moderate debt level ({de:.2f}) shows balanced capital structure"
        else:
            return f"High debt level ({de:.2f}) indicates elevated financial risk"
    
    def _interpret_interest_coverage(self, coverage: float) -> str:
        if coverage < 2:
            return f"Low interest coverage ({coverage:.1f}x) indicates financial stress"
        elif coverage < 5:
            return f"Moderate interest coverage ({coverage:.1f}x) shows adequate protection"
        else:
            return f"Strong interest coverage ({coverage:.1f}x) indicates low default risk"
    
    def _interpret_current_ratio(self, ratio: float) -> str:
        if ratio < 1:
            return f"Current ratio below 1.0 ({ratio:.2f}) indicates liquidity concern"
        elif ratio < 2:
            return f"Adequate current ratio ({ratio:.2f}) shows reasonable liquidity"
        else:
            return f"Strong current ratio ({ratio:.2f}) indicates excellent liquidity"
    
    def _interpret_fcf_margin(self, margin: float) -> str:
        margin_pct = margin * 100
        if margin_pct < 5:
            return f"Low FCF margin ({margin_pct:.1f}%) suggests cash generation concerns"
        elif margin_pct < 15:
            return f"Moderate FCF margin ({margin_pct:.1f}%) indicates adequate cash generation"
        else:
            return f"Strong FCF margin ({margin_pct:.1f}%) shows excellent cash generation"
    
    def _interpret_fcf_to_ni(self, ratio: float) -> str:
        if ratio < 0.8:
            return f"FCF/NI ratio of {ratio:.2f} suggests earnings quality concerns"
        elif ratio < 1.2:
            return f"FCF/NI ratio of {ratio:.2f} indicates good earnings quality"
        else:
            return f"FCF/NI ratio of {ratio:.2f} shows conservative accounting"
    
    def _interpret_asset_turnover(self, turnover: float) -> str:
        if turnover < 0.5:
            return f"Low asset turnover ({turnover:.2f}) suggests poor asset utilization"
        elif turnover < 1.5:
            return f"Moderate asset turnover ({turnover:.2f}) indicates reasonable efficiency"
        else:
            return f"High asset turnover ({turnover:.2f}) shows excellent asset efficiency"
    
    # RISK ASSESSMENT METHODS
    def _assess_roe_risk(self, roe: float) -> RiskLevel:
        if roe < 0:
            return RiskLevel.VERY_HIGH
        elif roe < 0.05:
            return RiskLevel.HIGH
        elif roe < 0.10:
            return RiskLevel.MODERATE
        elif roe < 0.20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _assess_leverage_risk(self, debt_to_equity: float) -> RiskLevel:
        if debt_to_equity > 2.0:
            return RiskLevel.VERY_HIGH
        elif debt_to_equity > 1.0:
            return RiskLevel.HIGH
        elif debt_to_equity > 0.5:
            return RiskLevel.MODERATE
        elif debt_to_equity > 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    # COMPREHENSIVE ANALYSIS
    def analyze_company(self, financial_data: FinancialData, company_code: str, company_name: str) -> AnalysisResult:
        """Perform comprehensive hedge fund level analysis"""
        
        all_metrics = []
        
        # Calculate all metric categories
        all_metrics.extend(self.calculate_valuation_metrics(financial_data))
        all_metrics.extend(self.calculate_profitability_metrics(financial_data))
        all_metrics.extend(self.calculate_leverage_metrics(financial_data))
        all_metrics.extend(self.calculate_quality_metrics(financial_data))
        all_metrics.extend(self.calculate_efficiency_metrics(financial_data))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(all_metrics)
        
        # Generate investment thesis
        investment_thesis = self._generate_investment_thesis(all_metrics, financial_data)
        
        # Identify key risks and opportunities
        key_risks = self._identify_key_risks(all_metrics)
        key_opportunities = self._identify_key_opportunities(all_metrics)
        
        return AnalysisResult(
            company_code=company_code,
            company_name=company_name,
            analysis_date=datetime.now(),
            metrics=all_metrics,
            overall_score=overall_score,
            investment_thesis=investment_thesis,
            key_risks=key_risks,
            key_opportunities=key_opportunities
        )
    
    def _calculate_overall_score(self, metrics: List[MetricResult]) -> float:
        """Calculate weighted overall investment score (0-100)"""
        score = 0
        weight_sum = 0
        
        for metric in metrics:
            if metric.value is not None:
                # Assign weights based on metric importance
                weight = self._get_metric_weight(metric.name)
                metric_score = self._normalize_metric_score(metric)
                score += metric_score * weight
                weight_sum += weight
        
        return (score / weight_sum * 100) if weight_sum > 0 else 50
    
    def _get_metric_weight(self, metric_name: str) -> float:
        """Get importance weight for each metric"""
        weights = {
            "ROE": 0.15,
            "ROIC": 0.15,
            "P/E Ratio": 0.10,
            "EV/EBITDA": 0.10,
            "Debt-to-Equity": 0.10,
            "FCF Margin": 0.10,
            "Operating Margin": 0.08,
            "Current Ratio": 0.07,
            "Asset Turnover": 0.05,
            "P/B Ratio": 0.05,
            "Interest Coverage (Est.)": 0.05
        }
        return weights.get(metric_name, 0.02)  # Default weight for other metrics
    
    def _normalize_metric_score(self, metric: MetricResult) -> float:
        """Normalize metric value to 0-1 score"""
        if metric.value is None:
            return 0.5
        
        # Define normalization rules for each metric
        normalization_rules = {
            "ROE": lambda x: min(1.0, max(0.0, x * 5)),  # 20% ROE = 1.0
            "ROIC": lambda x: min(1.0, max(0.0, x * 5)),  # 20% ROIC = 1.0
            "P/E Ratio": lambda x: max(0.0, min(1.0, (30 - x) / 25)),  # Lower P/E is better
            "EV/EBITDA": lambda x: max(0.0, min(1.0, (20 - x) / 15)),  # Lower EV/EBITDA is better
            "Debt-to-Equity": lambda x: max(0.0, min(1.0, (1.0 - x) / 0.8)),  # Lower debt is better
            "FCF Margin": lambda x: min(1.0, max(0.0, x * 5)),  # 20% FCF margin = 1.0
            "Operating Margin": lambda x: min(1.0, max(0.0, x * 5)),  # 20% operating margin = 1.0
            "Current Ratio": lambda x: min(1.0, max(0.0, (x - 0.5) / 1.5))  # 2.0 current ratio = 1.0
        }
        
        normalizer = normalization_rules.get(metric.name)
        if normalizer:
            return normalizer(metric.value)
        else:
            return 0.5  # Default neutral score
    
    def _generate_investment_thesis(self, metrics: List[MetricResult], data: FinancialData) -> str:
        """Generate professional investment thesis"""
        profitability_metrics = [m for m in metrics if m.category == MetricCategory.PROFITABILITY]
        valuation_metrics = [m for m in metrics if m.category == MetricCategory.VALUATION]
        leverage_metrics = [m for m in metrics if m.category == MetricCategory.LEVERAGE]
        
        thesis_parts = []
        
        # Profitability assessment
        roe_metric = next((m for m in profitability_metrics if m.name == "ROE"), None)
        if roe_metric and roe_metric.value:
            if roe_metric.value > 0.15:
                thesis_parts.append("Strong return on equity indicates efficient capital allocation")
            elif roe_metric.value > 0.10:
                thesis_parts.append("Moderate profitability with room for improvement")
            else:
                thesis_parts.append("Below-average profitability raises concerns about management effectiveness")
        
        # Valuation assessment
        pe_metric = next((m for m in valuation_metrics if m.name == "P/E Ratio"), None)
        if pe_metric and pe_metric.value:
            if pe_metric.value < 15:
                thesis_parts.append("Attractive valuation with potential upside")
            elif pe_metric.value > 25:
                thesis_parts.append("Premium valuation requires strong execution to justify")
        
        # Risk assessment
        debt_metric = next((m for m in leverage_metrics if m.name == "Debt-to-Equity"), None)
        if debt_metric and debt_metric.value:
            if debt_metric.value > 1.0:
                thesis_parts.append("Elevated leverage increases financial risk")
            else:
                thesis_parts.append("Conservative balance sheet provides downside protection")
        
        return ". ".join(thesis_parts) if thesis_parts else "Requires deeper analysis for investment thesis"
    
    def _identify_key_risks(self, metrics: List[MetricResult]) -> List[str]:
        """Identify key investment risks"""
        risks = []
        
        for metric in metrics:
            if metric.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                risks.append(f"{metric.name}: {metric.interpretation}")
        
        return risks[:5]  # Top 5 risks
    
    def _identify_key_opportunities(self, metrics: List[MetricResult]) -> List[str]:
        """Identify key investment opportunities"""
        opportunities = []
        
        # Look for strong metrics that indicate opportunities
        strong_metrics = [m for m in metrics if m.value is not None and self._is_strong_metric(m)]
        
        for metric in strong_metrics:
            opportunities.append(f"{metric.name}: {metric.interpretation}")
        
        return opportunities[:5]  # Top 5 opportunities
    
    def _is_strong_metric(self, metric: MetricResult) -> bool:
        """Determine if a metric indicates strength/opportunity"""
        strong_thresholds = {
            "ROE": 0.15,
            "ROIC": 0.15,
            "Operating Margin": 0.15,
            "FCF Margin": 0.15,
            "Current Ratio": 2.0
        }
        
        threshold = strong_thresholds.get(metric.name)
        if threshold and metric.value:
            return metric.value >= threshold
        
        # For ratios where lower is better
        low_is_good = {
            "P/E Ratio": 15,
            "EV/EBITDA": 10,
            "Debt-to-Equity": 0.3
        }
        
        threshold = low_is_good.get(metric.name)
        if threshold and metric.value:
            return metric.value <= threshold
        
        return False

# Example usage and testing
if __name__ == "__main__":
    # Create sample financial data
    sample_data = FinancialData(
        revenue=22000_000_000,  # 22 billion JPY
        gross_profit=8800_000_000,  # 40% gross margin
        operating_income=1770_000_000,  # 8% operating margin
        ebitda=2200_000_000,
        net_income=1080_000_000,  # 4.9% net margin
        total_assets=15000_000_000,
        current_assets=6000_000_000,
        cash_and_equivalents=2000_000_000,
        inventory=1500_000_000,
        accounts_receivable=2000_000_000,
        total_debt=4000_000_000,
        current_liabilities=3000_000_000,
        long_term_debt=3000_000_000,
        shareholders_equity=8000_000_000,
        free_cash_flow=1500_000_000,
        market_cap=20000_000_000,
        enterprise_value=22000_000_000,
        shares_outstanding=100_000_000,
        working_capital=3000_000_000,
        period="2025年9月期"
    )
    
    # Initialize analytics engine
    analytics = AdvancedFinancialAnalytics()
    
    # Perform comprehensive analysis
    result = analytics.analyze_company(sample_data, "2485", "株式会社ティア")
    
    print("=== HEDGE FUND LEVEL FINANCIAL ANALYSIS ===")
    print(f"Company: {result.company_name} ({result.company_code})")
    print(f"Overall Score: {result.overall_score:.1f}/100")
    print(f"Analysis Date: {result.analysis_date.strftime('%Y-%m-%d')}")
    print()
    
    print("INVESTMENT THESIS:")
    print(result.investment_thesis)
    print()
    
    # Group metrics by category
    categories = {}
    for metric in result.metrics:
        cat = metric.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(metric)
    
    for category, metrics in categories.items():
        print(f"=== {category.upper()} METRICS ===")
        for metric in metrics:
            value_str = f"{metric.value:.3f}" if metric.value else "N/A"
            risk_str = f" [{metric.risk_level.value}]" if metric.risk_level else ""
            print(f"{metric.name}: {value_str}{risk_str}")
            print(f"  → {metric.interpretation}")
        print()
    
    if result.key_risks:
        print("KEY RISKS:")
        for i, risk in enumerate(result.key_risks, 1):
            print(f"{i}. {risk}")
        print()
    
    if result.key_opportunities:
        print("KEY OPPORTUNITIES:")
        for i, opp in enumerate(result.key_opportunities, 1):
            print(f"{i}. {opp}")