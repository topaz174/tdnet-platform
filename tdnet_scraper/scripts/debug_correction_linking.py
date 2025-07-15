import sys
from pathlib import Path
from datetime import date
from sqlalchemy import create_engine, text

# Ensure project root imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_URL


def inspect_company(company_code: str):
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        query = text(
            """
            SELECT xf.id,
                   xf.period_start,
                   xf.period_end,
                   xf.consolidated_flag,
                   xf.accounting_standard,
                   xf.amendment_flag,
                   xf.parent_filing_id,
                   EXISTS (
                       SELECT 1
                         FROM disclosure_labels dl
                         JOIN disclosure_subcategories ds ON ds.id = dl.subcat_id
                        WHERE dl.disclosure_id = xf.disclosure_id
                          AND LOWER(ds.name) = 'earnings corrections'
                   ) AS has_corr_label
            FROM xbrl_filings xf
            JOIN companies c ON c.id = xf.company_id
            WHERE c.company_code = :code
            ORDER BY xf.period_end DESC, xf.id
            """
        )
        rows = conn.execute(query, {"code": company_code}).fetchall()

        if not rows:
            print("No filings found for company", company_code)
            return

        print(f"Inspection for company {company_code} ({len(rows)} filings):")
        for r in rows:
            print(
                f"ID={r.id} {r.period_start}→{r.period_end} consolidated={r.consolidated_flag} std={r.accounting_standard} | "
                f"amend_flag={r.amendment_flag} corr_label={r.has_corr_label} parent={r.parent_filing_id}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_correction_linking.py <COMPANY_CODE>")
        sys.exit(1)
    inspect_company(sys.argv[1]) 