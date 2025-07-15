import json
import csv
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import openai  # You can also use anthropic, google-generativeai, etc.
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

@dataclass
class CompanyKeywords:
    """Data class to store company information and extracted keywords"""
    securities_code: str
    description: str
    keywords: List[str]
    categories: List[str]
    
class KeywordExtractor:
    """Extract keywords from business descriptions using LLM"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the keyword extractor
        
        Args:
            api_key: OpenAI API key (or your preferred LLM provider)
            model: Model name to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def create_extraction_prompt(self, description: str) -> str:
        """Create a prompt for keyword extraction"""
        return f"""
You are an expert business analyst. Extract comprehensive keywords from the following Japanese company business description that would be useful for semantic search and retrieval.

Business Description: "{description}"

Please extract keywords in the following categories:

1. PRODUCTS & SERVICES: Specific products, services, or offerings
2. INDUSTRIES & SECTORS: Industry classifications, business sectors
3. TECHNOLOGIES: Technical terms, technologies, platforms
4. BUSINESS ACTIVITIES: Key business functions, operations
5. MARKET SEGMENTS: Target markets, customer segments
6. RELATED CONCEPTS: Associated terms, synonyms, broader concepts

Guidelines:
- Include both directly mentioned terms and reasonable inferences
- Use both English and Japanese terms when relevant
- Include technical jargon and industry-specific terminology
- Consider parent/child relationships (e.g., "electronics" for "televisions")
- Include acronyms and full forms
- Think about what someone might search for when looking for this type of company

Return the results in the following JSON format:
{{
    "products_services": ["keyword1", "keyword2", ...],
    "industries_sectors": ["keyword1", "keyword2", ...],
    "technologies": ["keyword1", "keyword2", ...],
    "business_activities": ["keyword1", "keyword2", ...],
    "market_segments": ["keyword1", "keyword2", ...],
    "related_concepts": ["keyword1", "keyword2", ...]
}}
"""

    def extract_keywords(self, description: str, securities_code: str = "") -> Dict[str, Any]:
        """
        Extract keywords from a business description
        
        Args:
            description: Business description text
            securities_code: Optional securities code for context
            
        Returns:
            Dictionary containing extracted keywords by category
        """
        try:
            prompt = self.create_extraction_prompt(description)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business analysis expert specializing in keyword extraction for search optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            
            # Extract JSON from the response (in case there's extra text)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            
            keywords_data = json.loads(json_str)
            
            # Flatten all keywords and get unique categories
            all_keywords = []
            categories = []
            
            for category, keywords in keywords_data.items():
                all_keywords.extend(keywords)
                categories.append(category.replace('_', ' ').title())
            
            # Remove duplicates while preserving order
            unique_keywords = list(dict.fromkeys(all_keywords))
            
            return {
                'keywords': unique_keywords,
                'categories': categories,
                'detailed_keywords': keywords_data
            }
            
        except Exception as e:
            print(f"Error extracting keywords for {securities_code}: {str(e)}")
            return {
                'keywords': [],
                'categories': [],
                'detailed_keywords': {}
            }
    
    def process_descriptions(self, df: pd.DataFrame, 
                           delay: float = 1.0) -> List[CompanyKeywords]:
        """
        Process company descriptions from pandas DataFrame
        
        Args:
            df: DataFrame with 'Securities_code' and 'Description' columns
            delay: Delay between API calls to avoid rate limiting
            
        Returns:
            List of CompanyKeywords objects
        """
        results = []
        
        for i, row in df.iterrows():
            securities_code = row['Securities_code']
            description = row['Description']
            
            print(f"Processing {i+1}/{len(df)}: {securities_code}")
            
            # Extract keywords
            extraction_result = self.extract_keywords(description, securities_code)
            
            # Create result object
            result = CompanyKeywords(
                securities_code=securities_code,
                description=description,
                keywords=extraction_result['keywords'],
                categories=extraction_result['categories']
            )
            
            results.append(result)
            
            # Add delay to avoid rate limiting
            if delay > 0 and i < len(df) - 1:
                time.sleep(delay)
        
        return results
    
    def save_results(self, results: List[CompanyKeywords], 
                    output_file: str = 'extracted_keywords.csv'):
        """Save results to CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['securities_code', 'description', 'keywords', 'categories']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'securities_code': result.securities_code,
                    'description': result.description,
                    'keywords': '; '.join(result.keywords),
                    'categories': '; '.join(result.categories)
                })
        
        print(f"Results saved to {output_file}")
    
    def save_results_json(self, results: List[CompanyKeywords], 
                         output_file: str = 'extracted_keywords.json'):
        """Save results to JSON file"""
        json_data = []
        for result in results:
            json_data.append({
                'securities_code': result.securities_code,
                'description': result.description,
                'keywords': result.keywords,
                'categories': result.categories
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")

def main():
    """Example usage of the KeywordExtractor"""
    
    # Initialize the extractor
    # Replace with your actual API key
    API_KEY = os.getenv("OPENAI_API_KEY")
    extractor = KeywordExtractor(api_key=API_KEY, model="gpt-4o")
    
    # Load company descriptions from CSV
    company_descriptions = pd.read_csv('data/Japan_company_descriptions.csv')
    
    # Process descriptions
    print("Starting keyword extraction...")
    results = extractor.process_descriptions(company_descriptions, delay=1.0)
    
    # Save results
    extractor.save_results(results, 'company_keywords.csv')
    extractor.save_results_json(results, 'company_keywords.json')
    
    # Print sample results
    print("\nSample Results:")
    for result in results[:2]:  # Show first 2 results
        print(f"\nSecurities Code: {result.securities_code}")
        print(f"Keywords: {', '.join(result.keywords[:10])}...")  # Show first 10 keywords

if __name__ == "__main__":
    main()