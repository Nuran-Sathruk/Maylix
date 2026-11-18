import random
import json
import string
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from pathlib import Path

def nltk_data():
    for pkg in ('punkt', 'wordnet', 'punkt_tab'):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

nltk_data()
lemmatizer = WordNetLemmatizer()

def load_intents():
    file = Path(__file__).resolve().with_name('data.json')
    with file.open('r', encoding='utf-8') as file:
        data = json.load(file)
    intents = data.get('intents')
    if intents is None and isinstance(data.get('response'), dict):
        intents = data['response'].get('intents')
    if not isinstance(intents, list) or not intents:
        raise ValueError("ERROR!")
    return intents

intents = load_intents()

words = []
documents = []
ignore_letters = set(string.punctuation)

for intent in intents:
    for pattern in intent.get('patterns', []):
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        words = sorted({
            lemmatizer.lemmatize(word.lower())
            for word in words
            if word and word.lower() not in ignore_letters and not word.isdigit()
})

word_to_index = {word: index for index, word in enumerate(words)}

def bag_of_words(sentence):
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]
    bag = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        index = word_to_index.get(word)
        if index is not None:
            bag[index] = 1.0
    return bag

training_vectors = [(bag_of_words(' '.join(doc[0])), doc[1]) for doc in documents]

def get_response(user_input):
    input_vec = bag_of_words(user_input)
    if np.linalg.norm(input_vec) == 0:
        return "I'm not sure I understand. Could you rephrase that?"
    best_score = 0.0
    best_tag = None
    for pattern_vec, tag in training_vectors:
        norm = np.linalg.norm(input_vec) * np.linalg.norm(pattern_vec)
        if norm > 0:
            similarity = np.dot(input_vec, pattern_vec) / norm
            if similarity > best_score:
                best_score = similarity
                best_tag = tag
                if best_score < 0.3 or best_tag is None:
                    return "Sorry, I didn't quite catch that."
                for intent in intents:
                    if intent.get('tag') == best_tag:
                        return random.choice(intent.get('responses', ['I am here to help.']))

if __name__ == '__main__':
    print("Maylix is ready! Type 'quit' to exit.")
    while True:
        try:
            message = input("You: ")
        except EOFError:
            break
        if message.lower() == 'quit':
            break
        print("Maylix:", get_response(message))