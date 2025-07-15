#!/usr/bin/env python3
"""
Populate context_dims table from lego-style contexts.json.
This script should be run once (or whenever you update contexts.json) so every
acceptable XBRL context ID is stored in the database ahead of ETL.
"""

import json
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.config import DB_URL


class ContextsPopulator:
    """Generate context IDs from components and insert into context_dims."""

    def __init__(self):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.contexts_def = self._load_contexts_def()

    def _load_contexts_def(self):
        contexts_file = Path(__file__).parent / 'contexts.json'
        if not contexts_file.exists():
            raise FileNotFoundError(f"{contexts_file} not found")
        with open(contexts_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ---------------------------------------------------------------------
    # Context-ID generation helpers
    # ---------------------------------------------------------------------
    def _generate_context_ids(self):
        """Yield dictionaries containing all context_dims columns for every valid combination."""
        tense_tokens = self.contexts_def.get("tense", [])
        accumulation_tokens = self.contexts_def.get("accumulation", [])
        period_terms: dict = self.contexts_def.get("periods", {})  # term -> span
        period_types = self.contexts_def.get("period_types", [])
        consolidated_tokens = self.contexts_def.get("consolidated", [])
        suffix_tokens = self.contexts_def.get("suffixes", [])

        # Mapping helpers --------------------------------------------------
        variant_lookup = {
            "": None,
            "ResultMember": "Result",
            "ForecastMember": "Forecast",
            "UpperMember": "Upper",
            "LowerMember": "Lower",
        }

        for tense in tense_tokens:
            for term, span in period_terms.items():
                for accum in accumulation_tokens:
                    # Skip meaningless combinations like AccumulatedYear
                    if term == "Year" and accum == "Accumulated":
                        continue

                    period_token = f"{tense}{accum}{term}" if accum else f"{tense}{term}"

                    for ptype in period_types:
                        base_id = f"{period_token}{ptype}"

                        for cons in consolidated_tokens:
                            consolidated_bool = None
                            if cons == "ConsolidatedMember":
                                consolidated_bool = True
                            elif cons == "NonConsolidatedMember":
                                consolidated_bool = False

                            for suff in suffix_tokens:
                                forecast_variant = variant_lookup.get(suff, None)

                                # Build context_id ------------------------------------------------
                                parts = [base_id]
                                if cons:
                                    parts.append(cons)
                                if suff:
                                    parts.append(suff)
                                context_id = "_".join(parts)

                                yield {
                                    "context_id": context_id,
                                    "period_token": period_token,
                                    "period_type": ptype,
                                    "fiscal_span": span,
                                    "consolidated": consolidated_bool,
                                    "forecast_variant": forecast_variant,
                                }

    # ---------------------------------------------------------------------
    # Insert logic
    # ---------------------------------------------------------------------
    def populate(self):
        session = self.Session()
        inserted = 0
        try:
            insert_sql = text(
                """
                INSERT INTO context_dims (
                    context_id, period_token, period_type, fiscal_span,
                    consolidated, forecast_variant
                ) VALUES (
                    :context_id, :period_token, :period_type, :fiscal_span,
                    :consolidated, :forecast_variant
                ) ON CONFLICT (context_id) DO NOTHING
                """
            )

            for record in self._generate_context_ids():
                session.execute(insert_sql, record)
                inserted += 1

            session.commit()
            print(f"Inserted/kept {inserted} context IDs in context_dims.")
        except Exception as e:
            session.rollback()
            print(f"Error populating contexts: {e}")
        finally:
            session.close()


def main():
    print("Starting contexts population from contexts.json...")
    populator = ContextsPopulator()
    populator.populate()
    print("\nContexts population completed.")


if __name__ == "__main__":
    main() 