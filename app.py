import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from fuzzywuzzy import process
from googletrans import Translator

# 1. Load and preprocess the product data
def load_data(file_path):
    """ Load product data from CSV file """
    data = pd.read_csv(file_path)
    return data

# Example: Load your CSV (adjust the file path accordingly)
data = load_data('products.csv')

# 2. Initialize Translator
translator = Translator()

def translate_query(query, target_lang='en'):
    """
    Translate a query to the target language (default is English).
    """
    translated = translator.translate(query, dest=target_lang)
    return translated.text

# 3. Vectorize Product Names, Descriptions, and Other Relevant Columns using TF-IDF
def vectorize_data(data):
    """ Vectorize product names and descriptions using TF-IDF """
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Concatenate name, category, sub_category, and description
    text_data = data['name'] + " " + data['category'] + " " + data['sub_category'] + " " + data['description']
    
    vectors = vectorizer.fit_transform(text_data)
    return vectors, vectorizer

# Example: Get the vectors
vectors, vectorizer = vectorize_data(data)

# 4. Build FAISS Index for Similarity Search
def build_faiss_index(vectors):
    """ Build a FAISS index for fast similarity search """
    dense_vectors = np.asarray(vectors.todense(), dtype=np.float32)  # Convert sparse matrix to dense format
    index = faiss.IndexFlatL2(dense_vectors.shape[1])  # L2 distance (Euclidean distance)
    index.add(dense_vectors)  # Add the vectors to the FAISS index
    return index

# Example: Build the index
index = build_faiss_index(vectors)

# 5. Perform Semantic Search with FAISS
def search_products(query, vectorizer, index, top_k=5):
    """ Perform a similarity search for the most relevant products based on the query """
    query_vector = vectorizer.transform([query]).todense().astype(np.float32)  # Convert query to vector
    distances, indices = index.search(query_vector, top_k)  # Search for the top_k closest products
    return indices, distances

# 6. Fuzzy Search for Similar Product Names
def fuzzy_search(query, data):
    """ Perform fuzzy matching to find close product names """
    product_names = data['name'].tolist()
    
    # Get the best match from the list
    best_match = process.extractOne(query, product_names)
    return best_match

# 7. Detecting Cheap or Expensive Queries
def detect_price_intent(query):
    """
    Detect whether the query is asking for cheap or expensive products.
    Returns 'cheap' or 'expensive' based on the keywords in the query.
    """
    # Define cheap and expensive indicators (translations to English will be handled by googletrans)
    cheap_keywords = ['cheap', 'sasta', 'low price', 'affordable']
    expensive_keywords = ['expensive', 'mehenga', 'high price', 'luxury']
    
    # Translate the query to English if necessary
    translated_query = translate_query(query, target_lang='en').lower()
    
    # Check for keywords in the query
    if any(keyword in translated_query for keyword in cheap_keywords):
        return 'cheap'
    elif any(keyword in translated_query for keyword in expensive_keywords):
        return 'expensive'
    else:
        return 'neutral'  # Default if no price intent is detected

# 8. Sort Products Based on Price
def sort_by_price(data, order='asc'):
    """ Sort the products by price in either ascending or descending order """
    if order == 'asc':
        return data.sort_values(by='price', ascending=True)
    elif order == 'desc':
        return data.sort_values(by='price', ascending=False)
    return data

# 9. Combined Search Function
def combined_search(query, vectorizer, index, data, top_k=5, fuzzy_threshold=80):
    """
    Perform both fuzzy and semantic search. 
    1. Translate the query into English.
    2. Perform fuzzy search for product names.
    3. If fuzzy match is strong, use that for semantic search.
    4. Otherwise, perform semantic search directly.
    5. Sort by price intent if necessary.
    """
    # Step 1: Translate the query into English first
    translated_query = translate_query(query, target_lang='en')
    print(f"Translated query: {translated_query}")
    
    # Step 2: Perform fuzzy search based on translated query
    fuzzy_match = fuzzy_search(translated_query, data)
    
    # Step 3: If fuzzy match score is above the threshold, use that for semantic search
    if fuzzy_match and fuzzy_match[1] >= fuzzy_threshold:
        print(f"Fuzzy match found: '{fuzzy_match[0]}' with score: {fuzzy_match[1]}")
        # Perform semantic search using fuzzy match result
        indices, distances = search_products(fuzzy_match[0], vectorizer, index, top_k)
        result = data.iloc[indices[0]]
    else:
        print("No strong fuzzy match found, performing semantic search directly...")
        # Perform semantic search directly with the translated query
        indices, distances = search_products(translated_query, vectorizer, index, top_k)
        result = data.iloc[indices[0]]
    
    # Step 4: Handle price intent (cheap or expensive)
    price_intent = detect_price_intent(query)
    if price_intent == 'cheap':
        result = sort_by_price(result, order='asc')  # Sort by price ascending for cheap
    elif price_intent == 'expensive':
        result = sort_by_price(result, order='desc')  # Sort by price descending for expensive
    
    return result

# 10. Display Only Final Top Results
def display_top_results(query, vectorizer, index, data, top_k=5, fuzzy_threshold=80):
    """
    Display the top results after processing fuzzy and semantic search in the backend.
    Only the final top results are returned.
    """
    final_results = combined_search(query, vectorizer, index, data, top_k, fuzzy_threshold)
    print("\nFinal Top Results:")
    print(final_results)

# Main function to allow user input
def main():
    query = input("Enter your product query: ")  # Allow user to input the query
    display_top_results(query, vectorizer, index, data, top_k=1)

if __name__ == "__main__":
    main()
