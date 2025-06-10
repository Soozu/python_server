#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os
import re
import nltk
import spacy

# Force CPU usage for stability
device = torch.device('cpu')
print(f"Using device: {device}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab', quiet=True)
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize

# Download spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys
    logger.info("Downloading spaCy model for NER...")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    """Load the dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the dataset"""
    # Fill NaN values
    df = df.fillna('')

    # Split combined categories and create a list of all categories
    df['all_categories'] = df['category'].apply(lambda x: [cat.strip() for cat in str(x).split(',')])

    # Create a combined text field with weighted importance
    df['combined_text'] = df['description'] + ' ' + df['description'] + ' ' + \
                         df['name'] + ' ' + \
                         df['category'] + ' ' + \
                         df['metadata']

    # Encode the categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'].apply(lambda x: str(x).split(',')[0].strip()))

    return df, label_encoder

def extract_query_info(query_text, available_cities, available_categories, available_budgets=None):
    """Extract city, category, and budget information from a user query"""
    query_lower = query_text.lower()

    # Initialize variables
    extracted_city = None
    extracted_category = None
    extracted_budget = None
    budget_amount = None
    
    # Process with spaCy for NER
    doc = nlp(query_text)
    
    # City extraction
    potential_cities = []
    
    # Get cities from spaCy NER
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            potential_cities.append(ent.text)
    
    # Check for direct city mentions
    for city in available_cities:
        city_lower = city.lower()
        if re.search(r'\b' + re.escape(city_lower) + r'\b', query_lower):
            potential_cities.append(city)
    
    # Find best matching city
    for potential_city in potential_cities:
        for city in available_cities:
            if potential_city.lower() == city.lower() or potential_city.lower() in city.lower():
                extracted_city = city
                break
        if extracted_city:
            break

    # Category mapping
    category_mapping = {
        "hotel": ["hotel", "resort", "lodge", "inn", "stay", "accommodation"],
        "cafe": ["cafe", "coffee", "coffee shop", "coffeehouse"],
        "restaurant": ["restaurant", "eat", "food", "dining", "meal"],
        "historical site": ["historical", "history", "heritage", "museum", "shrine"],
        "natural attraction": ["nature", "natural", "outdoors", "mountain", "lake", "falls"],
        "leisure": ["park", "amusement", "rides", "attraction", "entertainment"],
        "beach resort": ["beach resort", "seaside resort", "beach"],
        "resort": ["resort", "spa", "wellness", "retreat"],
        "farm": ["farm", "agriculture", "organic"],
        "religious site": ["church", "chapel", "cathedral", "temple"],
        "spa": ["spa", "massage", "relaxation"]
    }

    # Category detection
    for category in available_categories:
        category_lower = category.lower()
        if re.search(r'\b' + re.escape(category_lower) + r'\b', query_lower):
            extracted_category = category
            break

    # If no direct match, check synonyms
    if not extracted_category:
        for category in available_categories:
            category_lower = category.lower()
            for mapped_cat, synonyms in category_mapping.items():
                if mapped_cat in category_lower or category_lower in mapped_cat:
                    for synonym in synonyms:
                        if re.search(r'\b' + re.escape(synonym) + r'\b', query_lower):
                            extracted_category = category
                            break
                    if extracted_category:
                        break
                if extracted_category:
                    break

    # Budget extraction
    budget_patterns = [
        r'under\s*(\d+)\s*(?:pesos|php|₱)?',
        r'below\s*(\d+)\s*(?:pesos|php|₱)?',
        r'less than\s*(\d+)\s*(?:pesos|php|₱)?',
        r'budget of\s*(\d+)\s*(?:pesos|php|₱)?',
        r'₱\s*(\d+)',
        r'(\d+)\s*(?:pesos|php|₱)',
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, query_lower)
        if match:
            budget_amount = int(match.group(1))
            break
    
    # Budget keywords
    budget_keywords = {
        'cheap': 'budget',
        'affordable': 'budget', 
        'budget': 'budget',
        'expensive': 'luxury',
        'luxury': 'luxury'
    }
    
    if not budget_amount:
        for keyword, budget_type in budget_keywords.items():
            if keyword in query_lower:
                extracted_budget = budget_type
                break
    
    # Clean up query
    cleaned_query = query_text
    if extracted_city:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_city) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    if extracted_category:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_category) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return extracted_city, extracted_category, extracted_budget, cleaned_query, None, budget_amount, {}

# Dataset class
class DestinationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Model definition
class DestinationRecommender(torch.nn.Module):
    def __init__(self, num_labels, dropout=0.5):
        super(DestinationRecommender, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.intermediate = torch.nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size // 2)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size // 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pooled_output = torch.relu(self.intermediate(pooled_output))
        pooled_output = self.dropout2(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_recommendations(query_text, tokenizer, model, embeddings, df, city=None, category=None, budget=None, budget_amount=None, top_n=5):
    """Get destination recommendations based on a query text and optional filters"""
    try:
        # Tokenize the query
        query_encoding = tokenizer(
            query_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)

        # Get the query embedding
        model.eval()
        with torch.no_grad():
            outputs = model.roberta(
                input_ids=query_encoding['input_ids'],
                attention_mask=query_encoding['attention_mask']
            )
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Create a series of similarities with df indices
        similarity_series = pd.Series(similarities, index=df.index)

        # Apply filters
        filtered_df = df.copy()

        # Apply city filter
        if city:
            city_mask = filtered_df['city'].str.lower() == city.lower()
            if any(city_mask):
                filtered_df = filtered_df[city_mask]

        # Apply category filter
        if category:
            category_mask = filtered_df['all_categories'].apply(
                lambda cats: any(cat.lower() == category.lower() for cat in cats)
            )
            if any(category_mask):
                filtered_df = filtered_df[category_mask]
        
        # Apply budget filter
        if budget_amount is not None:
            filtered_df['budget'] = pd.to_numeric(filtered_df['budget'], errors='coerce')
            budget_mask = filtered_df['budget'] <= (budget_amount * 1.2)  # 20% buffer
            if any(budget_mask):
                filtered_df = filtered_df[budget_mask]

        # Get top recommendations
        if len(filtered_df) > 0:
            filtered_indices = filtered_df.index
            filtered_similarities = similarity_series[filtered_indices]
            top_indices = filtered_similarities.nlargest(min(top_n, len(filtered_similarities))).index
            recommendations = df.loc[top_indices]
            scores = filtered_similarities[top_indices].values
        else:
            # No filters matched, return top overall results
            top_indices = similarity_series.nlargest(top_n).index
            recommendations = df.loc[top_indices]
            scores = similarity_series[top_indices].values

        return recommendations, scores
    
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return pd.DataFrame(), np.array([])

def main():
    """Main function for testing"""
    file_path = "final_dataset.csv"
    
    try:
        # Load and preprocess data
        df = load_data(file_path)
        df, label_encoder = preprocess_data(df)
        
        print(f"Loaded {len(df)} destinations")
        print(f"Available cities: {df['city'].unique()[:10]}")
        print(f"Available categories: {df['category'].unique()[:10]}")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 