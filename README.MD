# X5 - Query Expansion Module

This module provides different query expansion (QE) techniques to generate a new user query using information from initial search results. The expanded query can then be reused to retrieve search results.

###  Inputs

- **Query (str)** → Original user query  
- **Results (list[dict])** → Initial top-k retrieved documents (Set K = 10)

### Output 

- **Expanded query (str)** - Original query concatenated with expansion terms

### 4 techniques 

- Rocchio Algorithm: Manual Relevance Labelling 
- Association Clustering: Term co-occurrence statistics 
- Metric Clustering: Term positional proximity 
- Scalar Clustering: Neighborhood similarity (built on-top of Metric)

## Installation & Setup

### I). Place the module inside your project directory.

### II). Import the required functions:

```python
from x5queryexpansionmodule.src.rocchio import rocchio_main
from x5queryexpansionmodule.src.association import association_main
from x5queryexpansionmodule.src.metric import metric_main
from x5queryexpansionmodule.src.scalar import scalar_main
```

### III). Usage - All methods follow the same interface:

```python
expanded_query = method_name(query, results)
```

For example,

- Rocchio Expansion

```python 
expanded_query = rocchio_main(query, results)
```

- Association Clustering

```python 
expanded_query = association_main(query, results)
```

- Metric Clustering

```python 
expanded_query = metric_main(query, results)
```

- Scalar Clustering

```python 
expanded_query = scalar_main(query, results)
```


## Workflow

```
User Query
    ↓
Initial Retrieval (Top-k Documents)
    ↓
Query Expansion Module (THIS MODULE)
    ↓
Expanded Query
    ↓
Re-Retrieval
```