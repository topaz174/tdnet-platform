import json
import os
import sys
import pandas as pd
import torch
import argparse
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


from tdnet_classifier import scrape_tdnet_titles
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


class SemanticDisclosureClassifier:
    def __init__(self, model_name="sonoisa/sentence-bert-base-ja-mean-tokens-v2", training_data_path="ml_training_data.json", similarity_threshold=0.3):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.training_data_path = training_data_path
        self.model = None
        self.categories = []
        self.category_embeddings = None
        
        print(f"Initializing semantic classifier with model: {model_name}")
        print(f"Loading training data from: {training_data_path}")
        
        
        self.load_training_data()
        
        
        self.init_model()
        
    def load_training_data(self):
        """Load training data from JSON file"""
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
                
            
            self.categories = [item["category"] for item in self.training_data]
            self.samples = [item["samples"] for item in self.training_data]
            
            print(f"Loaded {len(self.categories)} categories with examples")
        except Exception as e:
            print(f"Error loading training data: {e}")
            sys.exit(1)
    
    def init_model(self):
        """Initialize the BERT model"""
        try:
            print(f"Loading model {self.model_name}...")
            self.model = SentenceBertJapanese(self.model_name)
            print("Model loaded successfully")
            
            
            print("Computing embeddings for category samples...")
            self.compute_category_embeddings()
        except Exception as e:
            print(f"Error initializing model: {e}")
            sys.exit(1)
    
    def compute_category_embeddings(self):
        """Compute embeddings for all category samples"""
        all_sample_embeddings = []
        
        
        for samples in tqdm(self.samples, desc="Encoding category samples"):
            
            sample_embeddings = self.model.encode(samples)
            
            
            avg_embedding = torch.mean(sample_embeddings, dim=0)
            all_sample_embeddings.append(avg_embedding)
        
        
        self.category_embeddings = torch.stack(all_sample_embeddings)
        print(f"Category embeddings shape: {self.category_embeddings.shape}")
    
    def classify_disclosures(self, titles):
        """Classify a list of disclosure titles based on semantic similarity"""
        print(f"Classifying {len(titles)} disclosures...")
        
        
        title_embeddings = self.model.encode(titles)
        
        
        
        title_embeddings_np = title_embeddings.numpy()
        category_embeddings_np = self.category_embeddings.numpy()
        
        
        similarity_matrix = cosine_similarity(title_embeddings_np, category_embeddings_np)
        
        
        classifications = []
        for i, title in enumerate(titles):
            similarities = similarity_matrix[i]
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            
            if max_similarity < self.similarity_threshold:
                category = "Other [その他]"
            else:
                category = self.categories[max_idx]
            
            classifications.append({
                "title": title,
                "category": category,
                "similarity": max_similarity
            })
        
        return classifications
    
    def save_results(self, classifications, output_file="ml_classification_results.xlsx"):
        """Save classification results to Excel"""
        df = pd.DataFrame(classifications)
        
        
        result_df = pd.DataFrame({
            "title": df["title"],
            "categories": df["category"]
        })
        
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Classifications', index=False)
            
            
            stats = {
                "total": len(classifications),
                "other_count": sum(1 for c in classifications if c["category"] == "Other [その他]"),
                "average_similarity": np.mean([c["similarity"] for c in classifications])
            }
            stats["properly_classified"] = stats["total"] - stats["other_count"]
            stats["percent_other"] = (stats["other_count"] / stats["total"] * 100) if stats["total"] > 0 else 0
            stats["percent_classified"] = 100 - stats["percent_other"] if stats["total"] > 0 else 0
            
            stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"Results saved to {output_file}")
        return stats


def main():
    """Main function to handle command-line arguments and run the classifier"""
    parser = argparse.ArgumentParser(description='TDnet Semantic Disclosure Classifier')
    parser.add_argument('date', help='Date in MM/DD or MM/DD/YYYY format')
    parser.add_argument('--start-hour', type=int, help='Start hour (24-hour format)')
    parser.add_argument('--end-hour', type=int, help='End hour (24-hour format)')
    parser.add_argument('--model', default="sonoisa/sentence-bert-base-ja-mean-tokens-v2", 
                        help='Pretrained model name or path')
    parser.add_argument('--training-data', default="ml_training_data.json", 
                        help='Path to training data JSON file')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Similarity threshold for "Other" category (0.0-1.0)')
    parser.add_argument('--output', default="ml_classification_results.xlsx", 
                        help='Output Excel file path')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"TDnet Semantic Disclosure Classifier")
    print(f"{'='*80}")
    print(f"Scraping and classifying TDnet disclosures for {args.date}" + 
          (f" from {args.start_hour}:00" if args.start_hour is not None else "") + 
          (f" to {args.end_hour}:00" if args.end_hour is not None else ""))
    print(f"Using model: {args.model}")
    print(f"Using training data: {args.training_data}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"{'='*80}\n")
    
    
    titles = scrape_tdnet_titles(args.date, args.start_hour, args.end_hour)
    
    if not titles:
        print("No disclosures found. Exiting.")
        sys.exit(0)
    
    
    classifier = SemanticDisclosureClassifier(
        model_name=args.model,
        training_data_path=args.training_data,
        similarity_threshold=args.threshold
    )
    
    
    classifications = classifier.classify_disclosures(titles)
    
    
    stats = classifier.save_results(classifications, args.output)
    
    
    print(f"\n{'='*80}")
    print(f"Classification Summary")
    print(f"{'='*80}")
    print(f"Total disclosures: {stats['total']}")
    print(f"Properly classified: {stats['properly_classified']} ({stats['percent_classified']:.2f}%)")
    print(f"Classified as Other: {stats['other_count']} ({stats['percent_other']:.2f}%)")
    print(f"Average similarity score: {stats['average_similarity']:.4f}")
    print(f"Results saved to {args.output}")
    print(f"{'='*80}")
    
    
    if os.name == 'nt':  
        os.startfile(args.output)


if __name__ == "__main__":
    main() 