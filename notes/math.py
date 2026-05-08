def primeFactorization(num):
    ans = []
    if num <= 1:
        return ans
    while num % 2 == 0:
        ans.append(2)
        num //= 2
    for i in range(3, num, 2):
        while num % i == 0:
            ans.append(i)
            num //= i
    if num > 1:
        ans.append(num)
    return ans
