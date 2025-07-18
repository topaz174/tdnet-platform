from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Time, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import unified config
from config.config import DB_URL

engine = create_engine(DB_URL)
Base = declarative_base()


class Disclosure(Base):
    __tablename__ = 'disclosures'
    
    id = Column(Integer, primary_key=True)
    disclosure_date = Column(Date)
    time = Column(Time)
    company_code = Column(String)
    company_name = Column(String)
    title = Column(Text)
    xbrl_url = Column(Text, nullable=True)
    pdf_path = Column(Text, nullable=True)
    exchange = Column(String)
    update_history = Column(Text, nullable=True)
    page_number = Column(Integer)
    scraped_at = Column(DateTime, default=func.now())
    category = Column(Text, nullable=True)
    subcategory = Column(Text, nullable=True)
    # Note: embedding field requires pgvector extension, so we'll skip it for now
    # embedding = Column(Vector(1024), nullable=True)
    xbrl_path = Column(Text, nullable=True)
    extraction_status = Column(String(20), default='pending')
    extraction_method = Column(String(20), nullable=True)
    extraction_date = Column(DateTime, nullable=True)
    extraction_error = Column(Text, nullable=True)
    chunks_extracted = Column(Integer, default=0)
    extraction_duration = Column(Float, default=0.0)
    extraction_file_path = Column(Text, nullable=True)
    extraction_metadata = Column(JSON, nullable=True)
    has_xbrl = Column(Boolean, nullable=False, default=False)
    
    def __repr__(self):
        return f"<Disclosure(id={self.id}, company_code={self.company_code}, title={self.title})>"


class DisclosureCategory(Base):
    __tablename__ = 'disclosure_categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True, nullable=False)
    name_jp = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<DisclosureCategory(id={self.id}, name={self.name}, name_jp={self.name_jp})>"


class DisclosureSubcategory(Base):
    __tablename__ = 'disclosure_subcategories'
    
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('disclosure_categories.id'), nullable=False)
    name = Column(Text, unique=True, nullable=False)
    name_jp = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<DisclosureSubcategory(id={self.id}, name={self.name}, name_jp={self.name_jp})>"


class DisclosureLabel(Base):
    __tablename__ = 'disclosure_labels'
    
    id = Column(Integer, primary_key=True)
    disclosure_id = Column(Integer, ForeignKey('disclosures.id'), nullable=False)
    category_id = Column(Integer, ForeignKey('disclosure_categories.id'), nullable=True)
    subcat_id = Column(Integer, ForeignKey('disclosure_subcategories.id'), nullable=True)
    labeled_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<DisclosureLabel(disclosure_id={self.disclosure_id}, category_id={self.category_id}, subcat_id={self.subcat_id})>"


class DisclosureCategoryPattern(Base):
    __tablename__ = 'disclosure_category_patterns'
    
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('disclosure_categories.id'), nullable=False)
    regex_pattern = Column(Text, nullable=False)
    
    def __repr__(self):
        return f"<DisclosureCategoryPattern(id={self.id}, category_id={self.category_id})>"


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")