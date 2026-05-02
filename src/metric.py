from .document_loader import get_docs_by_ids
from .preprocessing import tokenize, preprocess_query
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import words

stemmer = PorterStemmer()

word_list = words.words()

def build_metric_clusters(stem_positions_docs, local_stem_vocab, stem_to_words, query_tokens):

    """
    Building metric clusters based on correlation values obtained from positional index statistics

    Inputs: stem_positions_docs (dict) - {doc_ids -> {stem -> [list of indexes where it occurred]}}
            local_stem_vocab (set) - local stem vocabulary
            stem_to_words (dict) - mapping of {stem -> [words]}
            query_tokens - list of stemmed query tokens

    Outputs: metric_clusters (dict) - {query_term -> [(candidate_stem, score), ...]}
    """

    metric_clusters = {}

    # only stems that can form clusters
    valid_query_stems = {q for q in query_tokens if q in local_stem_vocab}

    for qstem in valid_query_stems:

        score_map = defaultdict(float)

        candidate_stems = local_stem_vocab

        for cand in candidate_stems:

            if cand == qstem:
                continue

            total_score = 0.0

            # iterate over all documents
            for doc_id, stem_positions in stem_positions_docs.items():

                if qstem not in stem_positions or cand not in stem_positions:
                    continue

                q_positions = stem_positions[qstem]
                c_positions = stem_positions[cand]

                # positional interaction
                for i in q_positions:
                    for j in c_positions:
                        total_score += 1.0 / (abs(i - j) + 1)

            # normalization (based on associated vocab sizes)
            q_size = len(stem_to_words.get(qstem, [qstem]))
            c_size = len(stem_to_words.get(cand, [cand]))

            norm_factor = q_size * c_size

            if norm_factor > 0:
                total_score /= norm_factor

            if total_score > 0:
                score_map[cand] = total_score
        
        top_expansions = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        metric_clusters[qstem] = top_expansions
    
    return metric_clusters

def is_a_word(word, local_vocab):
    return word in local_vocab

def complete_word(stem, local_vocab):
    for w in local_vocab:
        if w.startswith(stem):
            return w
    return None

def get_expanded_query(query, query_tokens, metrics, local_vocab, k=2):
    """
    Generating the expanded query with new terms obtained from metric clusters

    Inputs: Query (str) - original user query
            query_tokens (list) - stemmed query tokens without duplicates
            metrics (dict) - sorted dictionary sorted as => {query_term -> [(candidate term, metric_score)]}
            local_vocab - set of local vocabulary obtained from local document set contents (unstemmed)

    Outputs: expanded_string (str)
    """

    expansion_terms = []
    used_expansions = set()
    query_set = set(query_tokens)

    for q in query_set:

        # get ranked candidates for this query stem
        candidates = metrics.get(q, [])

        added = 0

        for term, score in candidates:

            # safety checks
            if not term or not term.strip():
                continue

            # must exist in vocabulary
            if term not in local_vocab:
                continue

            # avoid adding original query terms
            if term in query_set:
                continue

            # avoid duplicates across all query tokens
            if term in used_expansions:
                continue

            if not any(w.startswith(term) for w in word_list):
                continue

            expansion_terms.append(term)
            used_expansions.add(term)

            added += 1

            if added == k:
                break

    cleaned_terms = []

    for w in expansion_terms:
        if is_a_word(w, local_vocab):
            cleaned_terms.append(w)
        else:
            cw = complete_word(w, local_vocab)
            if cw:
                cleaned_terms.append(cw)

    return query + " " + " ".join(cleaned_terms)

def metric_main(query = None, results=None):

    """
    Main Metric Clustering function

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

    query_tokens = preprocess_query(query)

    metrics = build_metric_clusters(stem_positions_docs, local_stem_vocab, stem_to_words, query_tokens)

    expanded_query = get_expanded_query(query, query_tokens, metrics, local_vocab, k=2)

    return expanded_query