#general bfs
def bfs(edges, start, end):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    q = collections.deque([start])
    seen = set([start])
    while q:
        cur = q.popleft()
        if cur == end:
            return 1
        for v in graph[cur]:
            if v in seen:
                continue
            q.append(v)
    return -1

#topological sort/cycle detection (Kahn's algorithm)
def topo_order(n, edges):
    graph = defaultdict(list)
    indeg = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    if len(order) != n:
        return []
    return order

#Dijkstra
def shortest_path(edges, start, end):
    graph = defaultdict(defaultdict(int))
    for u, v, w in edges:
        graph[u][v] = w
        graph[v][u] = w
    
    dist = {start:0}
    q = [(0, start)]

    while q:
        d, u = heapq.heappop(q)
        if u == end:
            return d
        for v in graph[u]:
            if v in dist and dist[v] <= d + w:
                continue
            dist[v] = d + w
            heapq.heappush(q, (d + w, v))
    return -1

#UnionFind/DSU
class DSU:
    def __init__(self):
        self.p, self.size, self.num_of_reg = {}, {}, 0
    
    def add(self, x):
        if x in self.p:
            return
        self.p[x] = x
        self.size[x] = 1
        self.num_of_reg += 1

    def find(self, x):
        return x if self.p[x] == x else self.find(self.p[x])
    
    def merge(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            self.p[rx] = ry
            self.size[ry] += self.size[rx]
        else:
            self.p[ry] = rx
            self.size[rx] += self.size[ry]
        self.num_of_reg -= 1
