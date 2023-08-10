from abc import ABC


class Node(ABC):
    def __init__(self, val=None, label=None, n_descendents=0):
        self._val = None
        self._label = None
        self._parent = None
        self._n_descendents = 0

        self.val = val
        self.label = label
        self.n_descendents = n_descendents

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, id):
        self._label = id

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        if self.parent is not None:
            self.parent.delchild(self)
        self._parent = node
        node.child = self

    @property
    def n_descendents(self):
        return self._n_descendents

    @n_descendents.setter
    def n_descendents(self, num):
        self._n_descendents = num
