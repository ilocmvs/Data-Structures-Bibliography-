#Segment Tree implementation with lazy propagation
#using interval max as an example
class STNode:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.left, self.right = None, None
        self.max = 0
        self.lazy = 0

class SegmentTree:
    def __init__(self, start, end):
        self.root = STNode(start, end)
    
    def _apply(self, node, delta):
        node.max = delta #cleverly change with requirement, may be replace or add
        node.lazy = delta #cleverly change with requirement, may be replace or add

    def _push(self, node):
        delta = node.lazy
        if not delta or node.start == node.end:
            return
        if node.left:
            self._apply(node.left, delta)
        if node.right:
            self._apply(node.right, delta)
        node.lazy = 0

    def _pull(self, node):
        node.max = max(node.left.max, node.right.max)

    def _create(self, node):
        m = (node.start + node.end) // 2
        if not node.left:
            node.left = STNode(node.start, m)
        if not node.right:
            node.right = STNode(m + 1, node.end)
        
    def range_add(self, ql, qr, delta):
        self._range_add(self.root, ql, qr, delta)
    
    def _range_add(self, node, ql, qr, delta):
        if qr < node.start or ql > node.end:
            return
        if ql <= node.start and node.end <= qr:
            self._apply(node, delta)
            return
        self._create(node)
        self._push(node)
        self._range_add(node.left, ql, qr, delta)
        self._range_add(node.right, ql, qr, delta)
        self._pull(node)
    
    def range_query(self, ql, qr):
        if ql > qr:
            return 0
        return self._range_query(self.root, ql, qr)

    def _range_query(self, node, ql, qr):
        if qr < node.start or ql > node.end:
            return 0
        if ql <= node.start and node.end <= qr:
            return node.max
        self._create(node)
        self._push(node)
        return max(self._range_query(node.left, ql, qr), self._range_query(node.right, ql, qr)) #choose the right operation

if __name__ == "__main__":
    #using LIS II as a test example
    # nums = [4,2,1,4,3,4,5,8,15]
    # k = 3
    nums = [7,4,5,1,8,12,4,7]
    k = 5
    #solution
    M = max(nums)
    dp = [0] * (M + 1)
    ST = SegmentTree(1, M)
    for i in range(len(nums)):
        dp[nums[i]] = max(dp[nums[i]], 1 + ST.range_query(max(0, nums[i] - k), nums[i] - 1))
        ST.range_add(nums[i], nums[i], dp[nums[i]])
    print(max(dp))
