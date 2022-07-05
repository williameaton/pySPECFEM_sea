import numpy as np

class hex_face():
    def __init__(self, node_dim=None, edof_dim=None):
        self.gnode = np.zeros(4, dtype=int)

        # Allows us to initialise these now or later
        if node_dim != None:
            self.init_node(node_dim)
        else:
            self.node = None

        if edof_dim != None:
            self.init_edof(edof_dim)
        else:
            self.edof = None

    def init_node(self, node_dim):
        self.node = np.zeros(node_dim, dtype=int)

    def init_edof(self, edof_dim):
        self.edof = np.zeros(edof_dim, dtype=int)


class hex_face_edge():
    def __init__(self):
        pass

    def init_node(self, node_dim):
        self.node = np.zeros(node_dim, dtype=int)

    def init_fnode(self, fnode_dim):
        self.fnode = np.zeros(fnode_dim, dtype=int)

    def set_fnode(self, fnode):
        self.fnode = fnode

    def set_node(self, node):
        self.node = node

