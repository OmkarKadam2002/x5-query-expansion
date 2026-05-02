import duckdb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "metadata.parquet")

con = duckdb.connect() 
#df = con.execute("SELECT doc_id, text FROM read_parquet('{DATA_PATH}')").df()
df = con.execute(f"""SELECT doc_id, text FROM read_parquet('{DATA_PATH}')""").df() 
    
doc_map = dict(zip(df["doc_id"], df["text"]))

def get_docs_by_ids(doc_ids=None):

    """
    Loads document content for local analysis

    Inputs: list of doc_ids obtained in initial retrieval results

    Outputs: Dictionary with doc_ids as keys and their 'text' content as values
            preserving the order in which ranked results were initially retrieved
    """

    if not doc_ids:
        return {}

    ordered = {d: doc_map[d] for d in doc_ids if d in doc_map}

    return ordered