class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: int
        """
        n, m = len(matrix), len(matrix[0]) # n columns rows
        run = [[matrix[i][j] for j in range(m)] for i in range(n)]
        for i in range(1, n):
            for j in range(m):
                run[i][j] += run[i - 1][j]
        run = [[0] * m] + run

        #        print(run)

        def helper(a):
            d = {0: 1}
            s = res = 0
            for i in range(len(a)):
                s += a[i]
                res += d.get(s - target, 0)
                d[s] = d.get(s, 0) + 1
            return res

        res = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                a = [k2 - k1 for k1, k2 in zip(run[i], run[j])]
                #                print(a)
                res += helper(a)
        return res


s = Solution()
print(s.numSubmatrixSumTarget(matrix=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], target=0))
print(s.numSubmatrixSumTarget(matrix=[[1, -1], [-1, 1]], target=0))
print(s.numSubmatrixSumTarget(matrix=[[0, 0, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]], target=0))

