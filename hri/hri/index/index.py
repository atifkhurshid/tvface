import numpy as np

from .internal_node import InternalNode
from .leaf_node import LeafNode


class HierarchicalRetrievalIndex(object):

    def __init__(self, matching_fn, threshold):

        self.matching_fn = matching_fn
        self.threshold = threshold

        self._root = None
        self._num_samples = 0
        self._next_label = 0


    def insert(self, val, sibling, pairdist):
        """Insert new node (with val and label) in index next to sibling

        Args:
            val (ndarray):      Feature vector as a numpy array
            mindist (float):    Minimum possible distance, used as distance to self
            sibling (Node):     Existing node that will be paired with new node
            pairdist (float):   Distance of val with sibling
        
        Returns:
            label (int):        Label assigned to inserted feature vector                
        """
        id = self._get_new_id()

        leafnode = LeafNode(id, val, None)

        if pairdist <= self.threshold:
            label = sibling.label
            leafnode.label = label
            leafnode.parent = sibling
            newnode = sibling

        else:
            newnode = InternalNode(endnode=True)
            leafnode.parent = newnode

            label = self._get_new_label()
            leafnode.label = label

            if sibling is None:
                self._root = InternalNode(endnode=False)
                newnode.parent = self._root

            else:
                sibling_pairdist = self._get_current_minimum_distance(sibling)
                if pairdist < sibling_pairdist:
                    newparent = InternalNode(endnode=False)
                    newnode.parent = newparent
                    newparent.parent = sibling.parent
                    sibling.parent = newparent

                else:
                    newnode.parent = sibling.parent

        self._update(newnode)

        return label


    def search(self, query):
        """Find best match for query according to matching_fn

        Args:
            query (ndarray):    Feature vector as a numpy array

        Returns:
            closest (Node):     Endnode in index closest to query
            dist (float):       Distance between query and closest node
        """
        closest = self._root
        dist = np.inf
        if closest is not None:
            dist = self.matching_fn([query], [closest.val])[0, 0]
            while isinstance(closest, InternalNode) and closest.endnode is False:
                values = np.array([c.val for c in closest.children])
                dists = self.matching_fn([query], values)[0]
                closestidx = np.argmin(dists)
                closest = closest.children[closestidx]
                dist = dists[closestidx]
        
        return closest, dist


    def get_leaf_depths(self):
        """Calculate depths of all leaf nodes in the index

        Returns:
            depths (ndarray):   List of depths ordered by ID
        """
        depths = -np.ones(self._num_samples, dtype=np.int)
        queue = [self._root]
        prev_depth = -1
        queue.append(0)
        while queue:
            current = queue.pop(0)
            depth = queue.pop(0)

            if depth > prev_depth:
                prev_depth += 1

            if isinstance(current, InternalNode):
                for c in current.children:
                    queue.append(c)
                    queue.append(depth + 1)

            if isinstance(current, LeafNode):
                depths[current.id] = depth
                
        return depths


    def _get_new_id(self):

        id = self._num_samples
        self._num_samples += 1

        return id


    def _get_new_label(self):

        label = self._next_label
        self._next_label += 1

        return label


    def _get_current_minimum_distance(self, node):

        distance = 0.0

        if node.parent is not None and len(node.parent.children) > 1:

            sibling_values = np.array([c.val for c in node.parent.children])
            dists = self.matching_fn([node.val], sibling_values)[0]
            dists[dists == 0.0] = np.inf    # Remove distance to self
            distance = np.min(dists)

        return distance


    def _update(self, node):

        current = node
        while current is not None:

            values, n_descendents, labels = [], [], []
            for c in current.children:
                values.append(c.val)
                n_descendents.append(c.n_descendents)
                labels.append(c.label)
            values, n_descendents = np.array(values), np.array(n_descendents)

            if len(values):
                current.n_descendents = n_descendents.sum()
                current.val = np.sum(values * n_descendents[:, np.newaxis], axis=0) / current.n_descendents
                label = list(set(labels))
                if len(label) == 1:
                    current.label = label[0]
                else:
                    current.label = None
                    
            current = current.parent
