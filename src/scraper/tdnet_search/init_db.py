from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Time, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from config.config_tdnet_search import DB_URL_SEARCH


engine = create_engine(DB_URL_SEARCH)
Base = declarative_base()


class Disclosure(Base):
    __tablename__ = 'disclosures'
    
    id = Column(Integer, primary_key=True)
    disclosure_date = Column(Date)
    time = Column(Time)
    company_code = Column(String)
    company_name = Column(String)
    title = Column(Text)
    xbrl_path = Column(Text, nullable=True)
    pdf_path = Column(Text)
    exchange = Column(String)
    update_history = Column(Text, nullable=True)
    page_number = Column(Integer)
    scraped_at = Column(DateTime, default=func.now())
    category = Column(Text, nullable=True)  # Parent categories (comma-separated)
    subcategory = Column(Text, nullable=True)  # Subcategories (comma-separated, empty for "Other")
    
    def __repr__(self):
        return f"<Disclosure(id={self.id}, company_code={self.company_code}, title={self.title})>"


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("TDnet Search database tables created successfully.") 