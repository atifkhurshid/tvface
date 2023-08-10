from .node import Node


class InternalNode(Node):
    def __init__(self, endnode=False):
        super().__init__()
        self._children = []
        self.endnode = endnode

    @property
    def children(self):
        return self._children

    @children.setter
    def child(self, node):
        self._children.append(node)

    def delchild(self, node):
        if node in self.children:
            self.children.remove(node)
