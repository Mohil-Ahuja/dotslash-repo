import pandas as pd
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import re
import numpy as np


def load_data(csv_path):
    """Load product data from the CSV file."""
    try:
        data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding="UTF-8")
    
    data.fillna({"Name": "", "Category": "", "Subcategory": "", "Description": "", "Price": 0}, inplace=True)
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce").fillna(0)
    return data


def translate_query(query):
    """Translate non-English query words to English."""
    translator = Translator()
    words = query.split()
    translated_words = []
    for word in words:
        try:
            translated = translator.translate(word, src="auto", dest="en")
            translated_words.append(translated.text)
        except Exception:
            translated_words.append(word)  # Use original word if translation fails
    return " ".join(translated_words)


def detect_price_keywords(query):
    """Identify price-related keywords in the query."""
    price_sort = None
    if "cheap" in query.lower():
        price_sort = "cheap"
        query = re.sub(r"\bcheap\b", "", query, flags=re.IGNORECASE)
    elif "expensive" in query.lower():
        price_sort = "expensive"
        query = re.sub(r"\bexpensive\b", "", query, flags=re.IGNORECASE)
    return query.strip(), price_sort


def initialize_vectorizer(data):
    """Prepare TF-IDF vectors for products."""
    data["CombinedText"] = data["Name"] + " " + data["Category"] + " " + data["Subcategory"] + " " + data["Description"]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["CombinedText"])
    return vectorizer, tfidf_matrix


def adjust_for_retailer_type(data, retailer_type, similarity_scores):
    """Adjust similarity scores based on retailer type."""
    if retailer_type == "small":
        price_weight = 0.1
        normalized_prices = np.log1p(data["Price"])  # Normalize price
        adjusted_scores = similarity_scores - (normalized_prices * price_weight)
    else:
        adjusted_scores = similarity_scores

    return np.maximum(adjusted_scores, 0)  # Ensure scores are non-negative


def search_products(query, vectorizer, tfidf_matrix, data, retailer_type="big", top_k=10, fuzzy_threshold=80):
    """Perform a search based on cosine similarity and fuzzy matching."""
    product_names = data["Name"].tolist()
    filtered_query = re.sub(r"\bcheap\b|\bexpensive\b", "", query, flags=re.IGNORECASE).strip()
    best_match = process.extractOne(filtered_query, product_names, scorer=fuzz.token_sort_ratio)

    if best_match and best_match[1] >= fuzzy_threshold:
        corrected_query = best_match[0]
    else:
        corrected_query = filtered_query

    query_vector = vectorizer.transform([corrected_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Adjust for retailer type
    adjusted_scores = adjust_for_retailer_type(data, retailer_type, similarity_scores)
    data["SimilarityScore"] = adjusted_scores

    top_results = data.sort_values(by="SimilarityScore", ascending=False).head(top_k)
    return top_results
