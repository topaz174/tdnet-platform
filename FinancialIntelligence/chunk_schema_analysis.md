# Chunk Schema Analysis

## Field Comparison: XBRL vs PDF Chunks

### Common Fields (Perfect Match)
| Field | XBRL Example | PDF Example | Database Column |
|-------|-------------|-------------|-----------------|
| `disclosure_id` | 673082 | 673014 | `disclosure_id INTEGER` |
| `chunk_index` | 0, 1, 2, 3 | 0, 1 | `chunk_index INTEGER` |
| `content` | "3つの重点項目..." | "上場インデックス..." | `content TEXT` |
| `content_type` | "risk_management", "capital_policy", "accounting_policy" | "forecast", "capital_policy" | `content_type VARCHAR(50)` |
| `section_code` | "general", "per_share_info", "segment_analysis" | "outlook", "segment_analysis" | `section_code VARCHAR(50)` |
| `heading_text` | "(1)経営成績に関する説明" | "上場インデックスファンド..." | `heading_text TEXT` |
| `char_length` | 586, 608, 336, 121 | 341, 171 | `char_length INTEGER` |
| `tokens` | 451, 465, 276, 98 | 265, 129 | `tokens INTEGER` |
| `vectorize` | true | true | `vectorize BOOLEAN` |
| `is_numeric` | false | false | `is_numeric BOOLEAN` |
| `disclosure_hash` | "972c694b..." | "de6ecdbe..." | `disclosure_hash VARCHAR(64)` |
| `source_file` | "15-00_19970_2025年８月期..." | "08-55_13080_ＥＴＦの..." | `source_file TEXT` |

### PDF-Specific Fields
| Field | PDF Example | Database Column | Notes |
|-------|-------------|-----------------|-------|
| `page_number` | 1, 2 | `page_number INTEGER` | NULL for XBRL chunks |

### Metadata Differences
| Source | Metadata Fields | Example |
|--------|----------------|---------|
| XBRL | `company_code`, `filing_date`, `period_end`, `extraction_method`, `language` | `{"company_code": "19970", "filing_date": "2025-08-30", "period_end": "2025-08-30", "extraction_method": "xbrl_qualitative", "language": "ja"}` |
| PDF | `page_number`, `extraction_method`, `language`, `pdf_path`, `mecab_available` | `{"page_number": 1, "extraction_method": "pdf_extraction", "language": "ja", "pdf_path": "tmpntgrdr8b.pdf", "mecab_available": true}` |

## Schema Compatibility Analysis

### ✅ Perfect Compatibility
All chunk fields from both XBRL and PDF can be stored in the unified schema without data loss.

### 🔄 Mapping Strategy
1. **Direct mapping**: Most fields map 1:1 to database columns
2. **Page number handling**: 
   - PDF chunks: Store actual page number
   - XBRL chunks: Store NULL (since XBRL doesn't have page concept)
3. **Metadata preservation**: Store all source-specific metadata in JSONB column

### 📊 Content Type Analysis
**XBRL Content Types Observed:**
- `risk_management`
- `capital_policy` 
- `accounting_policy`

**PDF Content Types Observed:**
- `forecast`
- `capital_policy`

**Section Codes Observed:**
- `general`, `per_share_info`, `segment_analysis`, `outlook`

### 🎯 Database Insert Strategy
```sql
INSERT INTO document_chunks (
    disclosure_id, chunk_index, content, content_type, section_code,
    heading_text, char_length, tokens, vectorize, is_numeric,
    disclosure_hash, source_file, page_number, metadata
) VALUES (
    %(disclosure_id)s, %(chunk_index)s, %(content)s, %(content_type)s, %(section_code)s,
    %(heading_text)s, %(char_length)s, %(tokens)s, %(vectorize)s, %(is_numeric)s,
    %(disclosure_hash)s, %(source_file)s, %(page_number)s, %(metadata)s
);
```

### 🔧 Required Pipeline Changes
1. **Add database insertion logic** to unified pipeline
2. **Batch insertion** for performance (chunks per disclosure)
3. **Error handling** for duplicate chunks
4. **Option to save chunks to both files and database**
5. **Resume logic** to skip already-inserted chunks

## Conclusion
The designed schema fully supports both XBRL and PDF chunk formats with zero data loss. The `page_number` field elegantly handles the PDF-specific requirement while remaining NULL for XBRL chunks.