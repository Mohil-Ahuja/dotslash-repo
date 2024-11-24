from flask import Flask, render_template, request
from search_engine import load_data, translate_query, detect_price_keywords, initialize_vectorizer, search_products

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    top_results = None  # Default when no results are found
    retailer_type = "big"  # Default retailer type

    if request.method == "POST":
        query = request.form["query"]
        retailer_type = request.form["retailer_type"]

        # Load data and initialize vectorizer
        data = load_data("search_engine_final/search_engine/semantic_search_dataList.csv")
        vectorizer, tfidf_matrix = initialize_vectorizer(data)

        # Translate and process query
        translated_query = translate_query(query)
        filtered_query, price_sort = detect_price_keywords(translated_query)

        # Search products
        top_results = search_products(filtered_query, vectorizer, tfidf_matrix, data, retailer_type=retailer_type)

        # Sort by price if applicable
        if price_sort == "cheap":
            top_results = top_results.sort_values(by="Price", ascending=True)
        elif price_sort == "expensive":
            top_results = top_results.sort_values(by="Price", ascending=False)

    return render_template("index.html", query=query, retailer_type=retailer_type, top_results=top_results)


if __name__ == "__main__":
    app.run(debug=True)
