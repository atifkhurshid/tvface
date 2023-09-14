# Hierarchical Retrieval Index

Fast feature matching in real-time video face recognition.

Inspired from search trees, the hierarchical retireval index consists of a tree-based data structure where leaves contain feature vectors representing facial images while the internal nodes store their mean feature vectors. This results in a hierarchical ordering of feature vectors similar to a dendrogram. Given a query feature vector of a new face image, its approximate nearest neighbors can be determined by traversing down the tree like tree search.

### Usage

Given a list of n-dimensional feature vectors X, the `match` function performs online clustering on X and returns a list of cluster labels for the input vectors. The index can be `reset` before processing a different dataset in order to delete previously inserted feature vectors.

```python
from hri import HierarchicalRetrievalIndexMatching

# Initialize index
matcher = HierarchicalRetrievalIndexMatching(metric='cosine', threshold=0.5)

# Add features from X to the index one-by-one, and get their labels
labels, _ = matcher.match(X)

# Delete all features from index to reset it
matcher.reset()
```
