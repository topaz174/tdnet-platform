#!/usr/bin/env python3
"""
Demonstration of Financial Data Extraction
=========================================

This script demonstrates how the extraction system would work with text-based documents
by creating sample text content and showing the extraction process.
"""

import json
from datetime import date
from financial_data_extraction_agent import FinancialDataExtractor, DataType

def create_sample_earnings_text():
    """Create sample earnings revision text (Japanese)"""
    return """
令和7年3月期　業績予想修正に関するお知らせ

当社は、令和7年3月期の業績予想を以下のとおり修正いたしましたので、お知らせいたします。

１．業績予想の修正内容

【連結業績予想】
                     売上高    営業利益   経常利益   純利益
                   （百万円）  （百万円）  （百万円）  （百万円）
前回予想（A）        12,500     1,200      1,180      780
今回修正予想（B）    13,800     1,450      1,420      920
増減額（B-A）        1,300      250        240        140
増減率（%）          10.4       20.8       20.3       17.9

【修正理由】
主力事業の好調により、売上高及び各利益項目において前回予想を上回る見込みとなりました。

２．配当予想について
期末配当金：前回予想 30円 → 今回予想 35円（5円増配）
年間配当金：前回予想 50円 → 今回予想 55円
"""

def create_sample_dividend_text():
    """Create sample dividend announcement text (Japanese)"""
    return """
配当予想の修正（増配）に関するお知らせ

１．配当予想修正の内容
                中間配当  期末配当  年間配当
前回予想          25円     25円     50円
今回修正予想      25円     30円     55円
増減             －       +5円     +5円

２．修正理由
業績好調により株主還元を強化するため、期末配当を5円増配いたします。

３．配当性向
修正後配当性向：28.5%（修正前：25.8%）
"""

async def test_extraction():
    """Test the financial data extraction system"""
    extractor = FinancialDataExtractor()
    
    print("=== Testing Financial Data Extraction ===\n")
    
    # Test 1: Earnings revision
    print("1. EARNINGS REVISION EXTRACTION:")
    print("-" * 40)
    
    earnings_text = create_sample_earnings_text()
    metrics, summary, confidence = extractor.extract_financial_data(
        earnings_text, 
        "業績予想修正に関するお知らせ", 
        DataType.EARNINGS_REVISION
    )
    
    print(f"Summary: {summary}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Extracted {len(metrics)} metrics:")
    
    for metric in metrics:
        print(f"  • {metric.metric_name}:")
        if metric.previous_value is not None:
            print(f"    Previous: {metric.previous_value:,.0f} {metric.unit or ''}")
        if metric.revised_value is not None:
            print(f"    Revised: {metric.revised_value:,.0f} {metric.unit or ''}")
        if metric.change_amount is not None:
            print(f"    Change: {metric.change_amount:+,.0f} {metric.unit or ''}")
        if metric.change_percentage is not None:
            print(f"    Change: {metric.change_percentage:+.1f}%")
        print()
    
    # Test 2: Dividend announcement
    print("\n2. DIVIDEND CHANGE EXTRACTION:")
    print("-" * 40)
    
    dividend_text = create_sample_dividend_text()
    metrics, summary, confidence = extractor.extract_financial_data(
        dividend_text, 
        "配当予想の修正（増配）に関するお知らせ", 
        DataType.DIVIDEND_CHANGE
    )
    
    print(f"Summary: {summary}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Extracted {len(metrics)} metrics:")
    
    for metric in metrics:
        print(f"  • {metric.metric_name}:")
        if metric.previous_value is not None:
            print(f"    Previous: {metric.previous_value} {metric.unit or ''}")
        if metric.revised_value is not None:
            print(f"    Revised: {metric.revised_value} {metric.unit or ''}")
        if metric.change_amount is not None:
            print(f"    Change: {metric.change_amount:+} {metric.unit or ''}")
        if metric.change_percentage is not None:
            print(f"    Change: {metric.change_percentage:+.1f}%")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_extraction())