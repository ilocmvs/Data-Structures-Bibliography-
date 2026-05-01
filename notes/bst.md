implementation itself
1.search
basic, nothing to note
2.insert
for unbalanced tree it's pretty easy, just recursively add nodes and return to the root
3.delete
key logic: find the successor after this node is deleted, then node.val = succ.val, node.right = _delete(node.right, succ.val)

serialization
| Scenario                                   |          Enough to recover original tree? | Core logic               |
| ------------------------------------------ | ----------------------------------------: | ------------------------ |
| General binary tree preorder without nulls |                                        No | Structure missing        |
| General binary tree preorder with nulls    |                                       Yes | `#` marks empty children |
| General binary tree level-order with nulls |                                       Yes | Queue reconstruction     |
| BST preorder without nulls                 |                     Yes, if unique values | Use value bounds         |
| BST postorder without nulls                |                     Yes, if unique values | Scan backward + bounds   |
| BST inorder only                           |                                        No | Only gives sorted values | *very important
| BST level-order without nulls              | Usually yes if treated as insertion order | Insert values one by one |
| Preorder + inorder                         |               Yes for general binary tree | Root + inorder split     |
| Postorder + inorder                        |               Yes for general binary tree | Root + inorder split     |

The core logic is: using bound.
root:
-inf, inf
left:
-inf,root
right:
root,inf

Or alternatively, use coordinates. But use more space, not recommended.