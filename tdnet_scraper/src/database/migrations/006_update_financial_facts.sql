-- ============================================================
-- Migration 007: Update Financial Facts Table (Corrected)
-- Add foreign key columns to reference normalized concept and context tables
-- ============================================================

-- Add new columns to financial_facts
ALTER TABLE financial_facts 
ADD COLUMN concept_id INTEGER REFERENCES concepts(id) NOT NULL,
ADD COLUMN context_id TEXT NOT NULL REFERENCES context_dims(context_id);

-- Add unique constraint as specified (without canonical_name since it will be removed)
ALTER TABLE financial_facts 
ADD CONSTRAINT financial_facts_section_concept_context_unique 
UNIQUE (section_id, concept_id, context_id);

-- Create indexes for performance
CREATE INDEX idx_financial_facts_concept_id ON financial_facts(concept_id);
CREATE INDEX idx_financial_facts_context_id ON financial_facts(context_id); 