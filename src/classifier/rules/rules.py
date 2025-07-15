# ============================================================================
# TDnet Disclosure Classification Rules (Refactored)
# ============================================================================
# SINGLE SOURCE OF TRUTH:
#   - CATEGORY_TREE holds every parent category exactly once
#   - Each category stores its Japanese translation, broad regex, and a nested
#     dict of its subcategories (each with pattern + JP translation).
# All helper structures (CATEGORIES, SUBCATEGORY_RULES, etc.) are generated
# automatically from CATEGORY_TREE so changing a category name/translation now
# only requires touching ONE place.

CATEGORY_TREE = {
    "EARNINGS & PERFORMANCE": {
        "translation": "業績・パフォーマンス",
        "pattern": r"決算|業績",
        "subcategories": {
            "Earnings Reports": {
                "translation": "決算短信",
                "pattern": r"決算短(?:信)?"
            },
            "Earnings Corrections": {
                "translation": "決算訂正",
                "pattern": (
                    r"(?:決算短(?:信)?).*?(?:訂正|修正)|"
                    r"業績予想.*?(?:訂正|修正|変更|差異)|"
                    r"(?:連結|個別|通期|第\d四半期)?.*業績.*差異|"
                    r"業績.*?(?:上方|下方)修正|"
                    r"配当予想.*?(?:訂正|修正|変更)|"
                    r"数値データ.*?訂正"
                )
            },
            "Earnings Forecasts": {
                "translation": "業績予想",
                "pattern": (
                    r"(?:連結|個別|通期|第\d四半期)?.*業績(?:予想|見通し)|"
                    r"配当予想(?!.*(?:訂正|修正|変更))"
                )
            },
            "Earnings Flash": {
                "translation": "決算速報",
                "pattern": r"決算速報(?:値)?|業績速報(?:値)?|速報値.*決算"
            },
            "Earnings Summaries & Reference": {
                "translation": "決算概要・参考資料",
                "pattern": (
                    r"(?:連結)?決算\s*(?:概要|概況|サマリー|参考資料|ハイライト|要約)|"
                    r"業績\s*(?:概要|概況)|"
                    r"決算\s*Q&A"
                )
            },
            "Earnings & Business Presentations": {
                "translation": "決算・事業説明会資料",
                "pattern": r"(?:決算|IR|投資家|会社|事業)\s*説明(?:会)?資料?"
            },
            "Supplementary Earnings Materials": {
                "translation": "決算補足説明資料",
                "pattern": (
                    r"決算.*(?:補足|FAQ|Q&A|質問と回答)|"
                    r"決算\s*データブック|"
                    r"FACT(?:\s*BOOK|SHEET)"
                )
            },
            "Monthly Performance": {
                "translation": "月次業績",
                "pattern": (
                    r"月次.*(?:売上|業績|収益|速報|概況)|"
                    r"売上(?:高|収益)?.*速報|"
                    r"売上高対前年同月比|"
                    r"運用資産概況|"
                    r"売電量"
                )
            },
            "Performance Variance Analysis": {
                "translation": "業績予想差異分析",
                "pattern": (
                    r"(?:業績予想|通期予想|第\d四半期予想).*(?:差異|修正|変更)|"
                    r"実績との差異|"
                    r"前期実績(?:値)?との差異|"
                    r"決算値との差異"
                )
            }
        }
    },
    # ---------------------------------------------------------------------
    "CORPORATE FINANCE": {
        "translation": "コーポレートファイナンス",
        "pattern": r"配当|株主還元|自己株式|社債|借入|資金調達|増資",
        "subcategories": {
            "Dividend Notices": {
                "translation": "配当予告・決定通知",
                "pattern": r"配当|剰余金.*(?:配当|処分)"
            },
            "Shareholder Return Policy": {
                "translation": "株主還元方針",
                "pattern": r"株主還元方針|利益配分.*方針"
            },
            "Treasury Stock Transactions": {
                "translation": "自己株式の取得・処分",
                "pattern": r"自社株買い|自己株式(?:の取得)?|優先株式.*(取得|消却)|コイン.*(?:焼却|バーン)|所在不明株主.*買取り|自社株価先渡取引"
            },
            "Stock Splits (Forward/Reverse)": {
                "translation": "株式分割・併合",
                "pattern": r"株式分割|株式併合"
            },
            "Reduction of Investment Unit": {
                "translation": "単元株数の変更",
                "pattern": r"投資単位の引下げ"
            },
            "Capital Reductions & Reserve Movements": {
                "translation": "資本減少・準備金処分",
                "pattern": r"資本準備金の額の減少|資本金の額の減少|減資|別途積立金.*取崩し|利益準備金.*減少"
            },
            "Debt & Financing": {
                "translation": "借入・資金調達",
                "pattern": r"社債|投資法人債|償還|借入|コミットメントライン|資金調達|包括決議"
            },
            "Equity Financing & Warrants": {
                "translation": "増資・新株予約権",
                "pattern": r"第三者割当|募集株式発行|新株発行|行使.*?割当|増資|資金使途.*変更|クラウドファンディング"
            },
            "Covenants & Financial Agreements": {
                "translation": "財務制限条項・金融契約",
                "pattern": r"財務上の特約|コベナンツ|covenant"
            }
        }
    },
    # ---------------------------------------------------------------------
    "CORPORATE STRATEGY & RESTRUCTURING": {
        "translation": "企業戦略・事業再編",
        "pattern": r"M&A|業務提携|子会社|事業譲渡|合併|買収|再編",
        "subcategories": {
            "M&A, Alliances & Restructuring": {
                "translation": "M&A・業務提携・組織再編",
                "pattern": (
                    r"M&A|業務提携|戦略的パートナーシップ|資本業務提携|合意|"
                    r"子会社.*(譲渡|取得|異動|設立|解散|吸収合併|株式.*処分|生産活動終了|清算|子会社化|持分法適用会社化)|"
                    r"事業譲渡|事業.*?廃止|会社分割|株式交換|株式移転|グループ再編|組織再編|物流統合|合弁会社設立|MBO|ＭＢＯ|公開買付け|TOB|ＴＯＢ|運営終了|事業移管|スピンオフ"
                )
            },
            "Restructuring: Personnel Reductions": {
                "translation": "人員削減・構造改革",
                "pattern": r"人員削減|合理化|希望退職"
            },
            "Corporate Strategy & Plans": {
                "translation": "経営戦略・中期計画",
                "pattern": (
                    r"(?:中期|中長期|長期)?(?:経営|事業)\s*(?:計画|方針|戦略|ビジョン)|"
                    r"経営ビジョン|経営戦略|中期経営計画|事業戦略|"
                    r"資本コスト.*経営|資本政策.*(?:策定|改定)|"
                    r"マテリアリティ|重要課題|"
                    r"設備投資(?:計画)?|工場.*(?:拡張|建設)|"
                    r"債務超過.*解消"
                )
            }
        }
    },
    # ---------------------------------------------------------------------
    "GOVERNANCE & LEADERSHIP": {
        "translation": "ガバナンス・経営体制",
        "pattern": r"取締役|監査役|役員|人事|異動|体制",
        "subcategories": {
            "Takeover Defense Policy Actions": {
                "translation": "買収防衛策",
                "pattern": r"買収防衛策|対応方針.*(継続|廃止|非継続|導入|変更|更新)|独立委員会.*?委員"
            },
            "Shareholder Proposals & AGM/EGM Agendas": {
                "translation": "株主提案・株主総会議案",
                "pattern": r"株主提案.*(対応|受領|撤回|意見)|定時株主総会.*(付議議案|継続会)|臨時株主総会.*招集"
            },
            "Governance & Senior Personnel Changes": {
                "translation": "役員・経営陣異動",
                "pattern": r"異動|人事|取締役|監査役|代表取締役|執行役員|経営体制|業務執行体制|選任|辞任|退任|就任|体制変更|組織変更|管掌範囲.*変更|監査等委員会設置会社への移行|委員会.*?設置|指名委員会|報酬委員会|財務会計基準機構への加入|本社移転|株主・投資家との対話"
            },
            "Executive Compensation Policy & Changes": {
                "translation": "役員報酬方針・改定",
                "pattern": r"役員報酬制度|報酬制度.*改定|役員報酬.*減額|経営指導料"
            },
            "Corporate Charter Changes": {
                "translation": "定款変更",
                "pattern": r"定款.*変更"
            },
            "Stock-based Compensation & Warrants": {
                "translation": "株式報酬・ストックオプション",
                "pattern": r"株式報酬|譲渡制限付株式|新株予約権|ストックオプション|ストック・ユニット|インセンティブプラン|ESOP|従業員持株|役員持株会|株式給付信託|BIP信託|信託.*?追加拠出|業績連動型株式交付制度"
            }
        }
    },
    # ---------------------------------------------------------------------
    "ACCOUNTING & REPORTING": {
        "translation": "会計・報告",
        "pattern": r"",
        "subcategories": {
            "Non-Operating & Financial P&L": {
                "translation": "営業外損益・金融収支",
                "pattern": r"営業外費用|営業外収益|金融収益|金融費用|為替差損|為替差益|デリバティブ|金利スワップ|その他有価証券評価差額金|持分法による投資利益|投資有価証券評価益"
            },
            "Special P&L, Impairments & Write-offs": {
                "translation": "特別損益・減損・償却",
                "pattern": r"特別損失|特別損益|特別利益|減損損失|固定資産.*減損|負ののれん|債権放棄|債権.*取立不能のおそれ"
            },
            "Reporting & Accounting Policy Changes": {
                "translation": "会計方針の変更",
                "pattern": r"セグメント.*変更|会計方針.*変更|IFRS.*適用|国際財務報告基準.*適用|減価償却方法.*変更|非連結決算への移行|継続企業の前提|保有目的変更"
            },
            "Significant Accounting Provisions & Valuations": {
                "translation": "会計上の引当金・評価",
                "pattern": r"繰延税金資産|繰延税金負債|税効果会計|法人税等調整額|退職給付引当金|貸倒引当金|引当金.*(戻入|繰入)|評価損|過年度法人税|補償対策引当金"
            },
        }
    },
    # ---------------------------------------------------------------------
    "SHAREHOLDER RELATIONS": {
        "translation": "株主対応",
        "pattern": r"",
        "subcategories": {
            "Controlling & Major Shareholder Info": {
                "translation": "支配株主・主要株主の異動",
                "pattern": r"支配株主等に関する事項|主要株主.*異動|による.*当社株式.*取得"
            },
            "Shareholder Benefit Programs": {
                "translation": "株主優待制度",
                "pattern": r"株主優待制度|株主優待|優待品|記念株主優待"
            },
            "Equity Offerings (Secondary)": {
                "translation": "株式売出し（セカンダリー）",
                "pattern": r"売出|立会外分売|貸借銘柄"
            }
        }
    },
    # ---------------------------------------------------------------------
    "ASSET MANAGEMENT": {
        "translation": "アセットマネジメント",
        "pattern": r"",
        "subcategories": {
            "Asset Transactions": {
                "translation": "資産譲渡・取得",
                "pattern": r"固定資産.*(取得|売却|譲渡)|販売用不動産.*購入|政策保有株式.*縮減|資産.*(売却|取得|譲渡)|投資有価証券.*売却|出資|暗号資産.*購入"
            }
        }
    },
    # ---------------------------------------------------------------------
    "MARKET & LISTING": {
        "translation": "市場・上場",
        "pattern": r"",
        "subcategories": {
            "Market/Listing Changes": {
                "translation": "上場市場区分の変更",
                "pattern": r"市場|区分|維持基準|上場廃止|指定替え|上場準備"
            },
            "ETFs & ETNs": {
                "translation": "ETF・ETN関連",
                "pattern": r"上場投信|ETF|ETN|ＥＴＦ|アジア国債・公債ＥＴＦ|SPDRゴールド・シェア|に関する日々の開示事項"
            }
        }
    },
    # ---------------------------------------------------------------------
    "CRISIS & RISK MANAGEMENT": {
        "translation": "危機・リスク管理",
        "pattern": r"訴訟|不祥事|延期|不備|規制|法令違反",
        "subcategories": {
            "Crisis: Security, Misconduct, Insolvency & Supply Chain": {
                "translation": "不祥事・安全保障・倒産・サプライチェーン問題",
                "pattern": r"不正アクセス|不祥事|不適切事案|和解|情報漏洩|サービス停止|サイバーセキュリティ|インシデント|債務超過|供給停止"
            },
            "Crisis: Financial Reporting Weakness": {
                "translation": "財務報告の内部統制不備",
                "pattern": r"財務報告.*不備|開示すべき重要な不備|内部統制.*不備"
            },
            "Crisis: Reporting Delays": {
                "translation": "決算発表遅延",
                "pattern": r"決算発表の延期|有価証券報告書提出.*延期|報告書.*提出期限延長|半期報告書.*期限延長"
            },
            "Legal & Regulatory Actions & Responses": {
                "translation": "訴訟・規制対応",
                "pattern": r"訴訟|公正取引委員会|独占禁止法|立入検査|行政処分|当局.*?命令|調査|規制|法令違反|審査|排除措置|課徴金|本日の報道について|再発防止策"
}
        }
    },
    # ---------------------------------------------------------------------
    "CORRECTIONS & ERRATA": {
        "translation": "訂正・差替え",
        "pattern": r"訂正|差替え|修正",
        "subcategories": {}
    }
}

# ============================================================================
# AUTO-GENERATED DERIVATIVES (Do not edit manually below this line)
# ============================================================================

# Categories (English)
CATEGORIES = sorted(list(CATEGORY_TREE.keys())) + ["OTHER"]

# Category translations (EN -> JA)
CATEGORY_TRANSLATIONS = {cat: data["translation"] for cat, data in CATEGORY_TREE.items()}
CATEGORY_TRANSLATIONS["OTHER"] = "その他"

# Category regex patterns
CATEGORY_RULES = {cat: data["pattern"] for cat, data in CATEGORY_TREE.items() if data["pattern"]}

# Build subcategories from tree
SUBCATEGORY_DEFINITIONS = {}
for cat, cat_data in CATEGORY_TREE.items():
    for subcat, sub_data in cat_data["subcategories"].items():
        SUBCATEGORY_DEFINITIONS[subcat] = {
            "parent": cat,
            "pattern": sub_data["pattern"],
            "translation": sub_data["translation"]
        }

# Taxonomy mapping: category -> [subcategories]
CATEGORY_TAXONOMY = {
    cat: list(cat_data["subcategories"].keys()) for cat, cat_data in CATEGORY_TREE.items()
}

# Subcategory translations (EN -> JA)
SUBCATEGORY_TRANSLATIONS = {
    subcat: data["translation"] for subcat, data in SUBCATEGORY_DEFINITIONS.items()
}

# Regex lists for matching
SUBCATEGORY_RULES = [
    (data["pattern"], subcat) for subcat, data in SUBCATEGORY_DEFINITIONS.items()
]

PARENT_CATEGORY_RULES = [
    (pattern, parent) for parent, pattern in CATEGORY_RULES.items()
]

# Subcategory -> Parent helper mapping
SUBCATEGORY_TO_PARENT = {subcat: data["parent"] for subcat, data in SUBCATEGORY_DEFINITIONS.items()}
SUBCATEGORY_TO_PARENT["Other"] = "OTHER" 