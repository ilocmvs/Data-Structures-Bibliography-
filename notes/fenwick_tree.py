class basicBIT:
    def __init__(self, arr):
        self.n = len(arr)
        self.BIT = [0] * (self.n + 1)
        for i in range(self.n):
            self.add(i, arr[i])

    def add(self, i, delta):
        i += 1
        while i <= self.n:
            self.BIT[i] += delta
            i += i & (-i)
    
    def sum(self, i):
        i += 1
        res = 0
        while i > 0:
            res += self.BIT[i]
            i -= i & (-i)
        return res

class BIT:
    def __init__(self, arr):
        self.n = len(arr)
        d = [0] * self.n
        d[0] = arr[0]
        for i in range(1, self.n):
            d[i] = arr[i] - arr[i - 1]
        a2 = [0] * self.n
        for i in range(self.n):
            a2[i] = d[i] * i
        self.BIT1 = basicBIT(d)
        self.BIT2 = basicBIT(a2)

    def range_add(self, l, r, delta):
        self.BIT1.add(l, delta)
        if r < self.n - 1:
            self.BIT1.add(r + 1, -delta)
        self.BIT2.add(l, delta * l)
        if r < self.n - 1:
            self.BIT2.add(r + 1, -delta * (r + 1))
    
    def prefix_sum(self, x):
        return (x + 1) * self.BIT1.sum(x) - self.BIT2.sum(x)
    

if __name__ == "__main__":
    freq = [2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9]
    bit = BIT(freq)
    print("Sum of elements in arr[0..4] is " + str(bit.prefix_sum(4)))
    print("Now adding 1 to index 0~5")
    bit.range_add(0, 5, 2)
    print("Sum of elements in arr[0..4]"+
                        " after update is " + str(bit.prefix_sum(4)))