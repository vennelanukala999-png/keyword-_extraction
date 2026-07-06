# keyword-_extraction
An NLP model that have been built an keyword extraction as mini project
Perform TF-IDF based keyword extraction on sample social media sentences, including text preprocessing, TF-IDF calculation, and keyword ranking, and analyze how TF-IDF ensures keyword relevance and uniqueness by handling common and repeated terms.

social_media_sentences = [
    "I'm loving the new coffee shop downtown! Their latte is amazing. #coffee #latte",
    "Just finished a great workout at the gym. Feeling energized! #fitness #gymlife",
    "Excited for the weekend! Planning to relax and read a book. #weekendvibes #reading",
    "New software update is causing some issues. Hope they fix it soon. #tech #bugs",
    "Enjoying a sunny day at the park with friends. Perfect weather! #friends #park",
    "Tried a new recipe tonight, and it was a disaster! Back to basics. #cookingfail #food",
    "Just saw the new superhero movie. Incredible action sequences! #movies #superhero",
    "My cat is sleeping in the most awkward position again. So cute! #catlife #pets",
    "Looking for recommendations for a good sci-fi series. Any suggestions? #scifi #tvshows",
    "Traffic is terrible today. Wish I could work from home. #commute #work"
]
print("Sample social media sentences created.")

output:-


Preprocess Text Data
Subtask:
Clean and normalize the sample social media text. This involves steps such as converting text to lowercase, removing special characters, URLs, mentions, hashtags, and stopwords, and tokenizing the text into individual words.
________________________________________
Reasoning: I need to import the necessary libraries for text preprocessing, specifically re for regular expressions and nltk for tokenization and stopwords. I will also download the punkt and stopwords resources from nltk.
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

print("Libraries imported and NLTK data downloaded.")

Libraries imported and NLTK data downloaded.


Reasoning: The previous code failed because nltk.downloader.DownloadError is not the correct exception type for nltk.data.find when a resource is not found; LookupError should be caught instead. I will fix the exception handling and then proceed with defining the text preprocessing function as outlined in the subtask instructions, which includes converting text to lowercase, removing URLs, mentions, hashtags, special characters, and stopwords, and then tokenizing the text.

import re
import nltkLibraries imported and NLTK data downloaded.
$0
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Remove all non-alphabetic characters and numbers, keeping only spaces between words
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and short words (e.g., single letters)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]

    return tokens

# Apply preprocessing to all sentences
preprocessed_sentences = [preprocess_text(sentence) for sentence in social_media_sentences]

print("Libraries imported, NLTK data downloaded, and text preprocessing function defined and applied.")
print("First 3 preprocessed sentences:")
for i in range(min(3, len(preprocessed_sentences))):
    print(preprocessed_sentences[i])



Libraries imported, NLTK data downloaded, and text preprocessing function defined and applied.
First 3 preprocessed sentences:
['im', 'loving', 'new', 'coffee', 'shop', 'downtown', 'latte', 'amazing']
['finished', 'great', 'workout', 'gym', 'feeling', 'energized']
['excited', 'weekend', 'planning', 'relax', 'read', 'book']
________________________________________
add nltk.download('punkt_tab') to ensure all necessary NLTK resources for tokenization are available.

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    # Added to fix the punkt_tab LookupError
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Define English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Remove all non-alphabetic characters and numbers, keeping only spaces between words
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and short words (e.g., single letters)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]

    return tokens

# Apply preprocessing to all sentences
preprocessed_sentences = [preprocess_text(sentence) for sentence in social_media_sentences]

print("Libraries imported, NLTK data downloaded, and text preprocessing function defined and applied.")
print("First 3 preprocessed sentences:")
for i in range(min(3, len(preprocessed_sentences))):
    print(preprocessed_sentences[i])
Libraries imported, NLTK data downloaded, and text preprocessing function defined and applied.
First 3 preprocessed sentences:
['im', 'loving', 'new', 'coffee', 'shop', 'downtown', 'latte', 'amazing']
['finished', 'great', 'workout', 'gym', 'feeling', 'energized']
['excited', 'weekend', 'planning', 'relax', 'read', 'book']
________________________________________


from sklearn.feature_extraction.text import TfidfVectorizer

# Join the preprocessed tokens back into strings for the TfidfVectorizer
preprocessed_sentences_str = [" ".join(tokens) for tokens in preprocessed_sentences]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed sentences to get the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sentences_str)

print("TF-IDF matrix created.")
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

F-IDF matrix created.
Shape of TF-IDF matrix: (10, 64)


import pandas as pd

# Get feature names (terms) from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array
dense_tfidf_matrix = tfidf_matrix.toarray()

# Create a DataFrame to display the TF-IDF scores
tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=feature_names)

print("TF-IDF Scores DataFrame created:")
print(tfidf_df.head())



TF-IDF Scores DataFrame created:
   action   amazing  awkward  back  basics      book  cat   causing    coffee  \
0     0.0  0.366686      0.0   0.0     0.0  0.000000  0.0  0.000000  0.366686   
1     0.0  0.000000      0.0   0.0     0.0  0.000000  0.0  0.000000  0.000000   
2     0.0  0.000000      0.0   0.0     0.0  0.408248  0.0  0.000000  0.000000   
3     0.0  0.000000      0.0   0.0     0.0  0.000000  0.0  0.366686  0.000000   
4     0.0  0.000000      0.0   0.0     0.0  0.000000  0.0  0.000000  0.000000   

   could  ...  today  tonight  traffic  tried    update   weather   weekend  \
0    0.0  ...    0.0      0.0      0.0    0.0  0.000000  0.000000  0.000000   
1    0.0  ...    0.0      0.0      0.0    0.0  0.000000  0.000000  0.000000   
2    0.0  ...    0.0      0.0      0.0    0.0  0.000000  0.000000  0.408248   
3    0.0  ...    0.0      0.0      0.0    0.0  0.366686  0.000000  0.000000   
4    0.0  ...    0.0      0.0      0.0    0.0  0.000000  0.377964  0.000000   

   wish  work   workout  
0   0.0   0.0  0.000000  
1   0.0   0.0  0.408248  
2   0.0   0.0  0.000000  
3   0.0   0.0  0.000000  
4   0.0   0.0  0.000000  

[5 rows x 64 columns]
________________________________________
Reasoning: Now that the TF-IDF scores are available in tfidf_df, I will proceed to rank keywords for each sentence. This involves iterating through each row of the DataFrame, sorting the terms by their TF-IDF scores in descending order, and extracting the top keywords for each social media sentence.
________________________________________
[ ]
 
Top 5 Keywords for each social media sentence:

Sentence 1:
  - amazing (Score: 0.3667)
  - coffee (Score: 0.3667)
  - downtown (Score: 0.3667)
  - im (Score: 0.3667)
  - latte (Score: 0.3667)


Sentence 2:
  - energized (Score: 0.4082)
  - feeling (Score: 0.4082)
  - finished (Score: 0.4082)
  - great (Score: 0.4082)
  - gym (Score: 0.4082)


Sentence 3:
  - book (Score: 0.4082)
  - excited (Score: 0.4082)
  - planning (Score: 0.4082)
  - read (Score: 0.4082)
  - relax (Score: 0.4082)


print("Top 5 Keywords for each social media sentence:\n")
for i, row in tfidf_df.iterrows():
    # Sort the row by TF-IDF scores in descending order and get the top 5 terms
    top_keywords = row.nlargest(5)
    print(f"Sentence {i+1}:")
    for keyword, score in top_keywords.items():
        print(f"  - {keyword} (Score: {score:.4f})")
    print("\n")
Top 5 Keywords for each social media sentence:

Sentence 1:
  - amazing (Score: 0.3667)
  - coffee (Score: 0.3667)
  - downtown (Score: 0.3667)
  - im (Score: 0.3667)
  - latte (Score: 0.3667)


Sentence 2:
  - energized (Score: 0.4082)
  - feeling (Score: 0.4082)
  - finished (Score: 0.4082)
  - great (Score: 0.4082)
  - gym (Score: 0.4082)


Sentence 3:
  - book (Score: 0.4082)
  - excited (Score: 0.4082)
  - planning (Score: 0.4082)
  - read (Score: 0.4082)
  - relax (Score: 0.4082)


Sentence 4:
  - causing (Score: 0.3667)
  - fix (Score: 0.3667)
  - hope (Score: 0.3667)
  - issues (Score: 0.3667)
  - software (Score: 0.3667)


Sentence 5:
  - day (Score: 0.3780)
  - enjoying (Score: 0.3780)
  - friends (Score: 0.3780)
  - park (Score: 0.3780)
  - perfect (Score: 0.3780)


Sentence 6:
  - back (Score: 0.3941)
  - basics (Score: 0.3941)
  - disaster (Score: 0.3941)
  - recipe (Score: 0.3941)
  - tonight (Score: 0.3941)


Sentence 7:
  - action (Score: 0.3941)
  - incredible (Score: 0.3941)
  - movie (Score: 0.3941)
  - saw (Score: 0.3941)
  - sequences (Score: 0.3941)


Sentence 8:
  - awkward (Score: 0.4472)
  - cat (Score: 0.4472)
  - cute (Score: 0.4472)
  - position (Score: 0.4472)
  - sleeping (Score: 0.4472)


Sentence 9:
  - good (Score: 0.4082)
  - looking (Score: 0.4082)
  - recommendations (Score: 0.4082)
  - scifi (Score: 0.4082)
  - series (Score: 0.4082)


Sentence 10:
  - could (Score: 0.3780)
  - home (Score: 0.3780)
  - terrible (Score: 0.3780)
  - today (Score: 0.3780)
  - traffic (Score: 0.3780)





We can extract `up to 500 words at a time
