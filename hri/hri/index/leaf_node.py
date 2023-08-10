from .node import Node


class LeafNode(Node):
    def __init__(self, id=None, val=None, label=None):
        super().__init__(val=val, label=label)
        
        self._id = None

        self.id = id
        self.n_descendents = 1

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id
