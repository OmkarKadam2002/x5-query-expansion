from .document_loader import get_docs_by_ids
from .preprocessing import tokenize, preprocess_query
from collections import defaultdict
import math
from nltk.stem import PorterStemmer
from nltk.corpus import words

stemmer = PorterStemmer()

word_list = words.words()

def score_with_window(pos_i, pos_j, window=10):
    """
    Only considers pairs where |p - q| <= window.
    """
    total_score = 0.0

    # ensure sorted (important safety check)
    pos_i = sorted(pos_i)
    pos_j = sorted(pos_j)

    j_start = 0

    for p in pos_i:
        # move j_start to first position within window
        while j_start < len(pos_j) and pos_j[j_start] < p - window:
            j_start += 1

        j = j_start

        # accumulate only within window
        while j < len(pos_j) and pos_j[j] <= p + window:
            total_score += 1.0 / (abs(p - pos_j[j]) + 1)
            j += 1

    return total_score

def build_full_metric_clusters(stem_positions_docs, local_stem_vocab, stem_to_words, window=10):

    """
    Building metric clusters across all stems based on correlation values obtained from positional index statistics

    Inputs: stem_positions_docs (dict) - {doc_ids -> {stem -> [list of indexes where it occurred]}}
            local_stem_vocab (set) - local stem vocabulary
            stem_to_words (dict) - mapping of {stem -> [words]}

    Outputs: full_metric_clusters (dict) - {stem -> [(neighbor_stem, score), ...]}
    """

    full_metric_clusters = {s: defaultdict(float) for s in local_stem_vocab}

    #vocab_list = list(local_stem_vocab)

    for doc_id, stem_positions in stem_positions_docs.items():
        #stems = list(stem_positions.keys())
        stems = [
            s for s, pos in stem_positions.items() if 2 <= len(pos) <= 50
            ]

        for i in range(len(stems)):
            stem_i = stems[i]
            pos_i = stem_positions[stem_i]

            for j in range(i + 1, len(stems)):
                stem_j = stems[j]
                pos_j = stem_positions[stem_j]

                total_score = score_with_window(pos_i, pos_j, window)
                
                size_i = len(stem_to_words.get(stem_i, [stem_i]))
                size_j = len(stem_to_words.get(stem_j, [stem_j]))

                norm = size_i * size_j
                if norm > 0:
                    total_score /= norm
                
                if total_score > 0:
                    full_metric_clusters[stem_i][stem_j] += total_score
                    full_metric_clusters[stem_j][stem_i] += total_score
        
    return full_metric_clusters

def cosine_similarity_calc(c_u, c_v):
    """
    c_u & c_v are Metric clusters around the stems u and v
    """
    if not c_u or not c_v:
        return 0.0
    
    dot = 0.0
    
    if len(c_u) < len(c_v):
        for k, v in c_u.items():
            if k in c_v:
                dot += v * c_v[k]
    else:
        for k, v in c_v.items():
            if k in c_u:
                dot += v * c_u[k]
    
    norm1 = math.sqrt(sum(v*v for v in c_u.values()))
    norm2 = math.sqrt(sum(v*v for v in c_v.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

def scalar_expansion(query_tokens, full_metrics):
    query_set = set(query_tokens)
    expansion_terms = {}
    
    for q in query_set: 
        
        if q not in full_metrics: 
            continue

        q_vec = full_metrics[q]

        scores = [] 
        
        for stem, stem_vec in full_metrics.items(): 
            if stem == q: 
                continue 
            
            sim = cosine_similarity_calc(q_vec, stem_vec) 
            
            if sim > 0: 
                scores.append((stem, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        expansion_terms[q] = [stem for stem, _ in scores[:10]]

    return expansion_terms

def is_a_word(word, local_vocab):
    return word in local_vocab

def complete_word(stem, local_vocab):
    for w in local_vocab:
        if w.startswith(stem):
            return w
    return None

def get_expanded_query(query, q_tok, scalar_values, local_vocab, k=2):

    """
    Generating the expanded query using scalar clustering output

    Inputs:
        query (str) - original user query
        q_tok (list) - stemmed query tokens without duplicates
        scalar_values (dict) - {query_stem: [expansion_term1, expansion_term2, ...]}
        local_vocab (set) - local vocabulary (unstemmed)

    Outputs:
        expanded_string (str)
    """

    query_set = set(q_tok)
    used_expansions = set()
    final_terms = []

    for q in query_set:

        candidates = scalar_values.get(q, [])

        added = 0

        for term in candidates:

            if not term or not term.strip():
                continue

            if term not in local_vocab:
                continue

            if term in query_set:
                continue

            if term in used_expansions:
                continue

            if not any(w.startswith(term) for w in word_list):
                continue

            final_terms.append(term)
            used_expansions.add(term)

            added += 1

            if added == k:
                break

    cleaned_terms = []

    for w in final_terms:
        if is_a_word(w, local_vocab):
            cleaned_terms.append(w)
        else:
            cw = complete_word(w, local_vocab)
            if cw:
                cleaned_terms.append(cw)

    return query + " " + " ".join(cleaned_terms)


def scalar_main(query = None, results=None):

    """
    Main Scalar Clustering function

    Inputs: Original User Query (str), 
            Initial top k retrieval results (dict) 

    Output: Expanded Query (str)
    """
    doc_ids = [r.get('doc_id') for r in results if r.get('doc_id') is not None]

    doc_contents = get_docs_by_ids(doc_ids)

    tokenized_docs = {
        doc_id: tokenize(text)
        for doc_id, text in doc_contents.items()
        }
    
    local_vocab = set()

    local_stem_vocab = set()

    stemmed_docs = {}

    # stem to words mapping
    stem_to_words = defaultdict(set)

    stem_positions_docs = {}

    for doc_id, tokens in tokenized_docs.items():
        stemmed_list = []
        position_map = defaultdict(list)

        for index, t in enumerate(tokens):
            #adding the token to the local vocabulary (unstemmed)
            local_vocab.add(t)

            #adding the stemmed word to the list of stems associated with the corresponding doc
            stem = stemmer.stem(t)
            stemmed_list.append(stem)

            #adding the *stemmed* token to the local stem vocabulary
            local_stem_vocab.add(stem)

            # Build {stem -> [(word)]} mapping
            stem_to_words[stem].add(t)
            position_map[stem].append(index)

        #stemmed_list contains duplicate stems
        stemmed_docs[doc_id] = stemmed_list

        # {doc_ids -> {stem -> [list of indexes where it occurred]}}
        stem_positions_docs[doc_id] = dict(position_map)

    #print(stem_positions_docs)

    query_tokens = preprocess_query(query)

    full_metrics = build_full_metric_clusters(stem_positions_docs,local_stem_vocab,stem_to_words)

    scalar_values = scalar_expansion(query_tokens, full_metrics)

    expanded_query = get_expanded_query(query, query_tokens, scalar_values, local_vocab, k=2)

    return expanded_query