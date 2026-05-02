from .document_loader import get_docs_by_ids
from .preprocessing import tokenize, preprocess_query
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import words

stemmer = PorterStemmer()

word_list = words.words()

def build_association_clusters(doc_tf, vocab, query_tokens):
    """
    Building association clusters based on correlation values obtained from co-occurrence statistics

    Inputs: doc_tf (dict) - {doc_id -> [(stem -> frequency)]}
            vocab (set) - local stem vocabulary
            query_tokens - list of stemmed query tokens

    Outputs: association_clusters (list) - list of tuples  
    """

    association_clusters = []

    q_tok = list(set(query_tokens))

    for voc in vocab:
        for qword in q_tok:

            c1, c2, c3 = 0, 0, 0

            for tf in doc_tf.values():

                freq_voc = tf.get(voc,0)
                freq_qword = tf.get(qword,0)

                c1 += freq_voc * freq_qword
                c2 += freq_voc * freq_voc
                c3 += freq_qword * freq_qword

            denom = (c1 + c2 + c3)
            score = c1 / denom if denom != 0 else 0
            if score > 0:
                association_clusters.append((voc, qword, score))

    return association_clusters

def is_a_word(word, local_vocab):
    return word in local_vocab

def complete_word(stem, local_vocab):
    for w in local_vocab:
        if w.startswith(stem):
            return w
    return None

def get_expanded_query(query, q_tok, assoc_map, local_vocab, k=2):

    """
    Generating the expanded query with new terms obtained from association clusters

    Inputs: Query (str) - original user query
            q_tok (list) - stemmed query tokens without duplicates
            assoc_map (dict) - sorted dictionary sorted as => {query: [(vocab_term, association_score)]}
            local_vocab - set of local vocabulary obtained from local document set contents (unstemmed)

    Outputs: expanded_string (str)
    """

    expansion_terms = []
    query_set = set(q_tok)
    used_expansions = set()

    for q in query_set:
        
        candidates = assoc_map.get(q, [])
        
        #new terms for this stemmed query token added
        added = 0

        for term, score in candidates:

            if not term.strip(): 
                continue 
            
            if term not in local_vocab: 
                continue 
            
            if term in query_set: 
                continue

            if term in used_expansions:
                continue

            if not any(w.startswith(term) for w in word_list):
                continue

            expansion_terms.append(term) 
            used_expansions.add(term)

            added += 1

            if added == k:
                break
        
    final_terms = []

    """
    Logic to check if the selected candidate stem is a word in the local vocab 
    if not then retrieve the word for which this candidate term is the stem
    """

    for w in expansion_terms:
        if is_a_word(w, local_vocab):
            final_terms.append(w)
        else:
            cw = complete_word(w, local_vocab)
            if cw:
                final_terms.append(cw)

    return query + " " + " ".join(final_terms)


def association_main(query = None, results=None):

    """
    Main Association Clustering function

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

    stem_to_words = defaultdict(set)

    for doc_id, tokens in tokenized_docs.items():
        stemmed_list = []

        for t in tokens:
            #adding the token to the local vocabulary (unstemmed)
            local_vocab.add(t)

            #adding the stemmed word to the list of stems associated with the corresponding doc
            stem = stemmer.stem(t)
            stemmed_list.append(stem)

            #adding the *stemmed* token to the local stem vocabulary
            local_stem_vocab.add(stem)

            # Build {stem -> [(word)]} mapping
            stem_to_words[stem].add(t)
        
        #stemmed_list contains duplicate stems
        stemmed_docs[doc_id] = stemmed_list

    doc_tf = {
        doc_id: Counter(tokens)
        for doc_id, tokens in stemmed_docs.items()
    }

    query_tokens = preprocess_query(query)

    associations = build_association_clusters(doc_tf, local_stem_vocab, query_tokens)

    assoc_map = defaultdict(list)

    for voc, qword, score in associations:
        assoc_map[qword].append((voc, score))

    for qword in assoc_map:
        assoc_map[qword].sort(key=lambda x: x[1], reverse=True)
    
    expanded_query = get_expanded_query(query, query_tokens, assoc_map, local_vocab, k=2)

    return expanded_query