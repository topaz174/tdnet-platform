-- Migration 010: Revamp concepts and concept_tags tables
-- Drop old tables (already backed up)
DROP TABLE IF EXISTS concept_tags CASCADE;
DROP TABLE IF EXISTS concepts CASCADE;

-- Create new concepts table that combines both old tables
CREATE TABLE concepts (
    id              serial PRIMARY KEY,
    standard        varchar(6) NOT NULL,
    local_name      text       NOT NULL,
    std_label_en    text,
    std_label_ja    text,
    item_type       varchar(12),
    UNIQUE (standard, local_name)
);

-- Create label_overrides table for corrections/normalizations
CREATE TABLE concept_overrides (
    concept_id        int PRIMARY KEY REFERENCES concepts(id) ON DELETE CASCADE,
    override_label_en text,
    override_label_ja text
);

-- Add indexes for performance
CREATE INDEX idx_concepts_standard ON concepts(standard);
CREATE INDEX idx_concepts_item_type ON concepts(item_type);
CREATE INDEX idx_concepts_local_name ON concepts(local_name); 