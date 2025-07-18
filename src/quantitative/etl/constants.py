"""
Constants for XBRL ETL processing.
"""

# Mapping from 2-character suffix to statement role with Japanese and English translations
ATTACHMENT_SUFFIX_MAP = {
    'bs': {'role': 'BalanceSheet', 'ja': '貸借対照表', 'en': 'Balance sheet'},
    'pl': {'role': 'ProfitLoss', 'ja': '損益計算書', 'en': 'Profit-and-loss'},
    'ci': {'role': 'ComprehensiveIncome', 'ja': '包括利益計算書', 'en': 'Comprehensive income'},
    'pc': {'role': 'CombinedPLCI', 'ja': '損益及び包括利益計算書', 'en': 'Combined PL & CI'},
    'fs': {'role': 'FinancialSummary', 'ja': '要約財政状態計算書', 'en': 'Condensed BS'},
    'ss': {'role': 'ChangesInEquity', 'ja': '株主資本等変動計算書', 'en': 'Changes in equity'},
    'cf': {'role': 'CashFlows', 'ja': 'キャッシュ・フロー計算書', 'en': 'Cash flows'},
    'sg': {'role': 'SegmentInformation', 'ja': 'セグメント情報', 'en': 'Segment info'},
    'nb': {'role': 'BSNotes', 'ja': '貸借対照表関係注記', 'en': 'BS notes'},
    'nc': {'role': 'PLCINotes', 'ja': '損益及び包括利益計算書関係注記', 'en': 'PL & CI notes'},
    'np': {'role': 'PLNotes', 'ja': '損益計算書関係注記', 'en': 'PL notes'},
    'ds': {'role': 'DividendSchedule', 'ja': '配当関係注記', 'en': 'Dividend schedule'},
    'qualitative': {'role': 'Narrative', 'ja': 'ナラティブ', 'en': 'Narrative'},
} 