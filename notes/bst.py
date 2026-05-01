class bstNode:
    def __init__(self, value):
        self.left = self.right = None
        self.value = value

class bst:
    def __init__(self):
        self.root = None

    def insert(self, value):
        self.root = self._insert(self.root, value)

    def _insert(self, node, value):
        if not node:
            return bstNode(value)
        if value < node.value:
            node.left = self._insert(node.left, value)
        else:
            node.right = self._insert(node.right, value)
        return node

    def search(self, value):
        return self._search(self.root, value)
    
    def _search(self, node, value):
        if not node:
            return False
        if node.value == value:
            return True
        if value < node.value:
            return self._search(node.left, value)
        elif value > node.value:
            return self._search(node.right, value)
        return False

    def delete(self, value):
        self._delete(self.root, value)
    
    def _delete(self, node, value):
        if not node:
            return None
        if value < node.value:
            node.left = self._delete(node.left, value)
        elif value > node.value:
            node.right = self._delete(node.right, value)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            node.value = self._successor(node.right).value
            node.right = self._delete(node.right, node.value)
        return node
    
    def _successor(self, node): #helper function to find the successor, only when the node has two children
        if not node or not node.right:
            return None
        node = node.right
        while node and node.left:
            node = node.left
        return node

    #serialization and deserialization
    #inorder traversal
    def serialize(self, node):
        if not node:
            return "#"
        return self.serialize(node.left) + "," + str(node.value) + "," + self.serialize(node.right)

    def deserialize(self, data):
        serialized = data.split(",")
        i = 0
        def _deserialize():
            nonlocal i
            if i >= len(serialized) or serialized[i] == "#": #cannot work without #
                return None
            node = bstNode(serialized[i])
            i += 1
            node.left = _deserialize()
            node.right = _deserialize()
            return node
        return _deserialize()

    #preorder traversal
    def preorder_serialize(self, node):
        if not node:
            return "#"
        return str(node.value) + "," + self.preorder(node.left) + "," + self.preorder(node.right)

    def preorder_deserialize(self, data): #still works without #
        serialized = data.split(",")
        i = 0
        def _deserialize(low, high):
            nonlocal i
            if i >= len(serialized):
                return None
            val = serialized[i] 
            if val < low or val > high or val == "#":
                return None
            node = bstNode(val)
            i += 1
            node.left = _deserialize(low, val)
            node.right = _deserialize(val, high)
            return node
        return _deserialize(float('-inf'), float('inf'))


    #postorder traversal    
    def postorder_serialize(self, node):
        if not node:
            return "#"
        return self.postorder(node.left) + "," + self.postorder(node.right) + "," + str(node.value)

    def postorder_deserialize(self, data): #still works without #
        serialized = data.split(",")
        i = len(serialized) - 1
        def _deserialize(low, high):
            nonlocal i
            if i >= len(serialized):
                return None
            val = serialized[i]
            if val < low or val > high or val == "#":
                return None
            node = bstNode(val)
            i -= 1
            node.right = _deserialize(val, high)
            node.left = _deserialize(low, val)
            return node
        return _deserialize(float('-inf'), float('inf'))

    #preorder+inorder deserialize
    def preorder_inorder_deserialize(self, preorder, inorder):
        if not preorder or not inorder:
            return None
        root = bstNode(preorder[0])
        root_index = inorder.index(preorder[0])
        root.left = self.preorder_inorder_deserialize(preorder[1:root_index+1], inorder[:root_index])
        root.right = self.preorder_inorder_deserialize(preorder[root_index+1:], inorder[root_index+1:])
        return root
    
    #postorder+inorder deserialize
    def postorder_inorder_deserialize(self, postorder, inorder):
        if not postorder or not inorder:
            return None
        root = bstNode(postorder[-1])
        root_index = inorder.index(postorder[-1])
        root.left = self.postorder_inorder_deserialize(postorder[:root_index], inorder[:root_index])
        root.right = self.postorder_inorder_deserialize(postorder[root_index:], inorder[root_index+1:])
        return root

