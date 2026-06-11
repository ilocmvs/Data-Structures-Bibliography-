class LCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.LOG = n.bit_length()
        self.depth = [0] * n
        self.up = [[-1] * self.LOG for _ in range(n)]

        graph = [[] for _ in range(n)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        self.dfs(root, -1, graph)

    def dfs(self, node, parent, graph):
        self.up[node][0] = parent

        for k in range(1, self.LOG):
            prev = self.up[node][k - 1]
            if prev != -1:
                self.up[node][k] = self.up[prev][k - 1]

        for nei in graph[node]:
            if nei == parent:
                continue
            self.depth[nei] = self.depth[node] + 1
            self.dfs(nei, node, graph)

    def lift(self, node, dist):
        for k in range(self.LOG):
            if dist & (1 << k):
                node = self.up[node][k]
        return node

    def lca(self, a, b):
        # Make a the deeper node
        if self.depth[a] < self.depth[b]:
            a, b = b, a

        # Lift a up to the same depth as b
        diff = self.depth[a] - self.depth[b]
        a = self.lift(a, diff)

        if a == b:
            return a

        # Lift both up while their 2^k ancestors are different
        for k in range(self.LOG - 1, -1, -1):
            if self.up[a][k] != self.up[b][k]:
                a = self.up[a][k]
                b = self.up[b][k]

        # Now a and b are direct children of the LCA
        return self.up[a][0]