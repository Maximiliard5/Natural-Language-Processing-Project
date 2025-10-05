import os
import json
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#setari limba
nltk.download("stopwords")
stop_words = set(stopwords.words("romanian"))
stemmer = SnowballStemmer("romanian")

poeti_folder_path = "C:/Users/maria/Desktop/FACULTATE/INTRODUCERE ÎN PRELUCRAREA LIMBAJULUI NATURAL/POETI"

# Extragere nume autor si titlu
def extract_author_and_title(poem_text):
    lines = poem_text.strip().split("\n")
    author = "Autor Necunoscut"
    title = "Titlu Necunoscut"
    found_author = False

    for line in lines:
        line = line.strip()
        if line:
            if not found_author:
                author = line
                found_author = True
            else:
                title = line
                break
    return author, title

# Procesare text BoW
def preprocess_text(text):
    text = re.sub(r"[^a-zA-ZăîâșțĂÎÂȘȚ0-9]", " ", text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Dictionar autor si poezii
poets_data = {}

# Procesare fisiere json
for file in os.listdir(poeti_folder_path):
    if file.endswith(".json"):
        file_path = os.path.join(poeti_folder_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            poems = json.load(f)
        for poem_text in poems:
            if "Text" in poem_text:
                author, title = extract_author_and_title(poem_text["Text"])
                if author not in poets_data:
                    poets_data[author] = []
                poets_data[author].append((title, poem_text["Text"]))
poets_data = {
    author: poems
    for author, poems in poets_data.items()
    if len(poems) >= 2 and author != "Autor Necunoscut"
}

# Filtrare autori
poets_data = {author: poems for author, poems in poets_data.items() if len(poems) >= 2}
# Construim vocabularul + eliminăm cuvintele rare
all_tokens = [word for author, poems in poets_data.items() for _, poem in poems for word in preprocess_text(poem)]
word_counts = Counter(all_tokens)
wordset = set(word for word, count in word_counts.items() if count > 1)

# Funcție pentru calculul BoW
def calculateBOW(wordset, tokens):
    tf_diz = dict.fromkeys(wordset, 0)
    for word in tokens:
        if word in tf_diz:
            tf_diz[word] += 1
    return tf_diz

# Creare lista
formatted_data = []
for author, poems in poets_data.items():
    for title, poem_text in poems:
        tokens = preprocess_text(poem_text)
        bow = calculateBOW(wordset, tokens)
        formatted_data.append((author, title, bow))

# DataFrame
df_formatted = pd.DataFrame(formatted_data, columns=["Autor", "Titlu Poezie", "BoW"])

# Convertim BoW într-un format numeric utilizabil
vectorizer = DictVectorizer(sparse=True)
X = vectorizer.fit_transform(df_formatted["BoW"].tolist())
y = df_formatted["Autor"]

# Impartim datele pentru atrenament si test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Antrenarea modelului
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calcul acuratete
accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratețea modelului Multinomial Naive Bayes: {accuracy:.4f}")

def predict_poem_author(poem_text, true_author=None):
    tokens = preprocess_text(poem_text)
    bow_vector = dict.fromkeys(wordset, 0)
    for word in tokens:
        if word in bow_vector:
            bow_vector[word] += 1
    X_test = vectorizer.transform([bow_vector])
    probabilities = model.predict_proba(X_test)[0]
    author_classes = model.classes_
    author_probabilities = {author: prob for author, prob in zip(author_classes, probabilities)}
    predicted_author = author_classes[probabilities.argmax()]

    print("\nVector de probabilități pentru fiecare autor:")
    for author, prob in author_probabilities.items():
        print(f"{author}: {prob:.4f}")

    print(f"\nPredicție: {predicted_author}")
    if true_author:
        if predicted_author == true_author:
            print("Predicția este CORECTĂ!")
        else:
            print(f"Predicția este GREȘITĂ! Autorul real este: {true_author}")

    return predicted_author, author_probabilities


poem_fragment = """
Pustie si albă e camera moartă...
Si focul sub vatră se stinge scrumit...--
Poetul, ălături, trăsnit stă de soartă,
"""

true_author = "Alexandru Macedonski"
predicted_author, author_probabilities = predict_poem_author(poem_fragment, true_author)


results_df = pd.DataFrame([author_probabilities], index=[predicted_author])
results_df.to_csv("rezultate_predictie.csv", encoding="utf-8")

print("\nRezultatele au fost salvate în 'rezultate_predictie.csv'.")