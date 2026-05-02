from .document_loader import get_docs_by_ids
from .preprocessing import tokenize, preprocess_query
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import math

stemmer = PorterStemmer()

def get_rel_non_rel_doc_labels(results):
    """
    Manual relevance feedback
    """
    rel_docs = []
    non_rel_docs = []

    print("\n--- Relevance Feedback ---")
    print("Enter 'y' for relevant, 'n' for non-relevant\n")

    for i, doc in enumerate(results):
        doc_id = doc.get("doc_id")
        title = doc.get("title")
        url = doc.get("url")
        score = doc.get("score")

        print(f"\nResult {i+1}")
        print(f"Doc ID: {doc_id}")
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Score: {score}")

        while True:
            label = input("Relevant? (y/n): ").strip().lower()
            if label == 'y':
                rel_docs.append(doc_id)
                break
            elif label == 'n':
                non_rel_docs.append(doc_id)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    return rel_docs, non_rel_docs

def process(text):
    tokens = tokenize(text)
    return [stemmer.stem(t) for t in tokens]

def compute_df(tokenized_docs):
    df = defaultdict(int)
    for tokens in tokenized_docs.values():
        for term in set(tokens):
            df[term] += 1
    return df

def compute_tfidf(tokens, df, N):
    tf = Counter(tokens)
    vec = {}

    for term, count in tf.items():
        tf_val = 1 + math.log(count)
        idf_val = math.log((N+1) / (df[term] + 1))
        vec[term] = tf_val * idf_val

    return vec

def centroid(doc_ids, doc_vectors):
        vec = defaultdict(float)
        if not doc_ids:
            return vec

        for d in doc_ids:
            for term, val in doc_vectors[d].items():
                vec[term] += val

        for term in vec:
            vec[term] /= len(doc_ids)

        return vec

def is_a_word(word, local_vocab):
    return word in local_vocab

def complete_word(stem, local_vocab):
    for w in local_vocab:
        if w.startswith(stem):
            return w
    return None

def rocchio_main(query = None, results=None):
    """
    Main Rocchio function

    Inputs: Original User Query (str), 
            Initial top k retrieval results (dict) 

    Output: Expanded Query (str)
    """
    rel_docs, non_rel_docs = get_rel_non_rel_doc_labels(results)
    
    rel_doc_contents = get_docs_by_ids(rel_docs)

    local_vocab = set()

    rel_tokens = {}

    for doc_id, text in rel_doc_contents.items():
        tokens = tokenize(text)          # raw tokens
        rel_tokens[doc_id] = []

        for t in tokens:
            local_vocab.add(t)           # BEFORE stemming
            rel_tokens[doc_id].append(stemmer.stem(t))

    
    non_rel_doc_contents = get_docs_by_ids(non_rel_docs)

    non_rel_tokens = {}

    for doc_id, text in non_rel_doc_contents.items():
        tokens = tokenize(text)

        non_rel_tokens[doc_id] = []

        for t in tokens:
            local_vocab.add(t)
            non_rel_tokens[doc_id].append(stemmer.stem(t))

    all_docs = {**rel_tokens, **non_rel_tokens}

    N = len(all_docs)

    df = compute_df(all_docs)

    doc_vectors = {
        doc_id: compute_tfidf(tokens, df, N)
        for doc_id, tokens in all_docs.items()
    }

    query_tokens = preprocess_query(query)

    query_vec = compute_tfidf(query_tokens, df, N)

    alpha = 1.0
    beta = 0.75
    gamma = 0.15

    rel_centroid = centroid(rel_docs, doc_vectors)
    non_rel_centroid = centroid(non_rel_docs, doc_vectors)

    new_query = defaultdict(float)

    for term, val in query_vec.items():
        new_query[term] += alpha * val

    for term, val in rel_centroid.items():
        new_query[term] += beta * val

    for term, val in non_rel_centroid.items():
        new_query[term] -= gamma * val

    original_terms = set(query_tokens)

    expansion_terms = sorted(
        [(t, v) for t, v in new_query.items() if t not in original_terms],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    print(expansion_terms)

    final_terms = []

    for w, _ in expansion_terms:
        if is_a_word(w, local_vocab):
            final_terms.append(w)
        else:
            cw = complete_word(w, local_vocab)
            if cw:
                final_terms.append(cw)

    print(final_terms)
    expanded_query = query + " " + " ".join([t for t in final_terms])

    return expanded_query