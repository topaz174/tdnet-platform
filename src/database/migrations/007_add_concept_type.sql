-- ============================================================
-- Migration 008: Add Type Column to Concepts Table
-- Add type field to categorize concepts based on XSD element types
-- ============================================================

-- Add type column to concepts table
ALTER TABLE concepts 
ADD COLUMN IF NOT EXISTS type TEXT; 