import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

"""
Applied the same preprocessing pipeline as the one used in task 2 for indexing to ensure consistency. 
"""


# list of stopwords
STOPWORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','is','was','are','were','be','been',
    'has','have','had','it','its','this','that','these','those',
    'as','not','he','she','they','we','you','i','do','did','will',
    'would','could','should','may','might','can','also','just','more'
])

def clean_text(text):
    
    # Must not be empty
    if not text: 
        return ""
    
    # Convert everything to lowercase
    text = text.lower()
    
    # Remove punctuation & non-alphanumeric chars
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove whitespaces
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_token(token):
    
    # Must be between 4 and 15 characters
    if len(token) < 4 or len(token) > 15: 
        return False
    
    # Must be at least 60% letters
    letter_count = sum(1 for c in token if c.isalpha())
    if letter_count / len(token) < 0.6: 
        return False
    
    # Must not contain 4+ consecutive digits
    if re.search(r'\d{4,}', token): 
        return False
    
    # Must not look like a hex hash
    if re.search(r'[0-9a-f]{8,}', token): 
        return False
    
    # Filter long random-looking strings — real words are rarely over 15 chars
    # redundant conition (?); max length changes from 20 to 15
    if len(token) > 15: 
        return False
    
    # Must have at least one vowel (real words have vowels)
    if not re.search(r'[aeiou]', token): 
        return False
    
    return True

def tokenize(text):
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t not in STOPWORDS and is_valid_token(t) and t.isalpha()]
    return tokens

def preprocess_query(query):
    tokens = tokenize(query)
    return [stemmer.stem(t) for t in tokens]