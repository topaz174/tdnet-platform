-- Add classification tables for storing classification results
-- Rules remain in Python code, only results are stored in database

CREATE TABLE IF NOT EXISTS disclosure_categories (
    id      SERIAL PRIMARY KEY,
    name    TEXT UNIQUE NOT NULL,         -- e.g. 'EARNINGS & PERFORMANCE'
    name_jp TEXT                          
);

CREATE TABLE IF NOT EXISTS disclosure_subcategories (
    id          SERIAL PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES disclosure_categories(id),
    name        TEXT UNIQUE NOT NULL,     -- e.g. 'Quarterly Earnings'
    name_jp     TEXT                    
);

-- Link table (derived metadata) - this is where classification results are stored
-- Multiple rows per disclosure are allowed to support multiple categories/subcategories
CREATE TABLE IF NOT EXISTS disclosure_labels (
    id            SERIAL PRIMARY KEY,
    disclosure_id INTEGER NOT NULL REFERENCES disclosures(id) ON DELETE CASCADE,
    category_id   INTEGER REFERENCES disclosure_categories(id),
    subcat_id     INTEGER REFERENCES disclosure_subcategories(id),
    labeled_at    TIMESTAMP DEFAULT now(),
    
    -- Ensure we don't have duplicate category/subcategory combinations per disclosure
    UNIQUE(disclosure_id, category_id, subcat_id)
);

-- Add indexes for performance
CREATE INDEX idx_disclosure_labels_disclosure_id ON disclosure_labels(disclosure_id);
CREATE INDEX idx_disclosure_labels_category_id ON disclosure_labels(category_id);
CREATE INDEX idx_disclosure_labels_subcat_id ON disclosure_labels(subcat_id);
CREATE INDEX idx_disclosure_subcategories_category_id ON disclosure_subcategories(category_id); 