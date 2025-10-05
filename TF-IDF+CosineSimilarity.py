import json
import os
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

# Folder containing JSON files
JSON_FOLDER="POETI"

# Configuration for test evaluation
TEST_LENGTH=400 # Length of test segments
N_TESTS=100  # Number of test iterations
TEST_SPECIFIC_POEM=False  # Set to True to test a specific poem, False for accuracy tests

# Specific poem to test when TEST_SPECIFIC_POEM = True

#ACEST POEM ESTE UN LUCEAFAR MODIFICAT DAR TOT GASESTE CA AUTORUL ESTE MIHAI EMINESCU
SPECIFIC_TEST_POEM="""Fost-a cândva în basme scris,
Fost-a ca-n vis pierdut,
Din neam de domni, în aur nins,
O fată-n rai crescut.

Și una-n casă,-n gând și vis,
Cu farmec lin de stea,
Așa cum sfinții în abis,
Și luna-n noaptea grea.

Din umbra boltelor domnești,
Pășește lin pe prag,
Spre geam, acolo unde-n colț,
Luceafărul e drag.
"""

SPECIFIC_TEST_AUTHOR="Mihai Eminescu"
SPECIFIC_TEST_TITLE="Luceafărul"

# Load all JSON files from the folder
def load_all_poems(folder):
    poems_list=[]
    
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path=os.path.join(folder, filename)
            author_name=filename.replace(" - Poezii Modificat.json","")

            with open(file_path, "r", encoding="utf-8") as f:
                poems_data = json.load(f)

            for poem in poems_data:
                text = poem["Text"].strip()
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                title = lines[1] if len(lines) > 1 else "Unknown Title"

                poems_list.append({
                    "author": author_name,
                    "title": title,
                    "text": text
                })
    
    return pd.DataFrame(poems_list)

# Precompute TF-IDF vectors for poems
def compute_tfidf_index(df):
    vectorizer=TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # Unigrams and bigrams for better phrase matching
        max_features=100000,  # Increase vocabulary size
        sublinear_tf=True,  # Reduce weight of frequent words
        min_df=1  # Keep all words
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    return vectorizer, tfidf_matrix

# Find the best matching author and poem
def find_best_match_with_probabilities(input_lyrics,vectorizer,tfidf_matrix,df):
    input_vec=vectorizer.transform([input_lyrics])
    similarities=1-cdist(input_vec.toarray(),tfidf_matrix.toarray(),metric="cosine")

    similarity_df=df.copy()
    similarity_df["similarity_score"]=similarities.flatten()

    author_probabilities=(
        similarity_df.groupby("author")["similarity_score"]
        .mean()
        .sort_values(ascending=False)
    )

    author_probabilities/=author_probabilities.max()

    best_author=author_probabilities.idxmax()
    best_poems_from_author=similarity_df[similarity_df["author"]==best_author]

    if best_poems_from_author.empty:
        return {"author": best_author, "title": "Unknown Title", "text": "No matching poem found."}, author_probabilities

    best_match_index=best_poems_from_author["similarity_score"].idxmax()
    best_match=best_poems_from_author.loc[best_match_index]

    return best_match, author_probabilities

# Function to evaluate precision
def evaluate_precision(df,vectorizer,tfidf_matrix,num_tests,test_length):
    correct_authors=0
    correct_titles=0
    correct_full_matches=0

    for _ in range(num_tests):
        random_poem=df.sample(1).iloc[0]
        true_author=random_poem["author"]
        true_title=random_poem["title"]
        text=random_poem["text"]

        if len(text) > test_length:
            start_idx=random.randint(0,len(text)-test_length)
            test_excerpt=text[start_idx:start_idx+test_length]
        else:
            test_excerpt=text

        best_match,_=find_best_match_with_probabilities(test_excerpt,vectorizer,tfidf_matrix,df)

        predicted_author=best_match["author"]
        predicted_title=best_match["title"]

        if predicted_author==true_author:
            correct_authors+=1
        if predicted_title==true_title:
            correct_titles+=1
        if predicted_author==true_author and predicted_title==true_title:
            correct_full_matches+=1

    author_precision=correct_authors/num_tests
    title_precision=correct_titles/num_tests
    full_match_precision=correct_full_matches/num_tests

    print("\n**Model Precision Results**")
    print(f"Author Precision: {author_precision:.2%}")
    print(f"Title Precision: {title_precision:.2%}")
    print(f"Full Match Precision: {full_match_precision:.2%}")

# Function to test a specific poem
def test_specific_poem(df, vectorizer, tfidf_matrix):
    print("\n**Testing Specific Poem**")
    best_match,author_probabilities=find_best_match_with_probabilities(SPECIFIC_TEST_POEM,vectorizer,tfidf_matrix,df)

    print("\n**Author Probabilities (Normalized, Sorted):**")
    print(author_probabilities)

    print("\n**Best Match Found:**")
    print(f"Predicted Title: {best_match['title']}")
    print(f"Predicted Author: {best_match['author']}")

    print(f"\n**True Author:** {SPECIFIC_TEST_AUTHOR}")
    print(f"**True Title:** {SPECIFIC_TEST_TITLE}")

    if best_match["author"]==SPECIFIC_TEST_AUTHOR and best_match["title"]==SPECIFIC_TEST_TITLE:
        print("Prediction is FULLY CORRECT!")
    elif best_match["author"]==SPECIFIC_TEST_AUTHOR:
        print("Author is CORRECT, but title is INCORRECT.")
    else:
        print("Prediction is INCORRECT.")

    print(f"\n**Poem Excerpt:**\n{best_match['text'][:300]}...\n")

# Main function (non-interactive)
def main():
    print("Loading poems from JSON files...")
    df = load_all_poems(JSON_FOLDER)
    
    if df.empty:
        print("No poems found! Check your JSON files.")
        return

    print(f"Loaded {len(df)} poems from {df['author'].nunique()} authors.")
    
    print("Indexing poems for fast search...")
    vectorizer,tfidf_matrix=compute_tfidf_index(df)
    print("Indexing complete!")

    if TEST_SPECIFIC_POEM:
        test_specific_poem(df,vectorizer,tfidf_matrix)
    else:
        evaluate_precision(df,vectorizer,tfidf_matrix,N_TESTS,TEST_LENGTH)

# Run the script
if __name__ == "__main__":
    main()
