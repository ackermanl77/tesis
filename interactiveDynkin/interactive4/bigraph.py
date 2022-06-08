from collections.abc import Set
from collections.abc import Mapping
from array import array
from enum import Enum
from enum import IntEnum

class LineStyle(IntEnum):
    solid = -1
    dotted = 1

class Stroke(Enum):
    undirected_dotted = '..'
    undierected_solid = '--'
    forward_dotted = '.>'
    forward_solid = '->'
    backward_dotted = '<.'
    backward_solid = '<-'

    @property
    def linestyle(self):
        return LineStyle.solid if '-' in self.value else LineStyle.dotted

    @property
    def is_directed(self):
        return '>' in self.value or '<' in self.value

    @property
    def is_forward(self):
        return '>' in self.value

    @property
    def is_backward(self):
        return '<' in self.value

class BigraphVertexSetView(Set):
    def __init__(self, G):
        self._bigraph = G

    def __contains__(self, u):
        A = self._bigraph._adj
        return 0 <= u < len(A) and A[u][u] != 0

    def __iter__(self):
        A = self._bigraph._adj
        for i in range(len(A)):
            if A[i][i] == 2:
                yield i

    def __len__(self):
        return sum(1 for u in self)

class BigraphEdgeSetView(Set):
    def __init__(self, G):
        self._bigraph = G

    def __contains__(self, edge):
        A = self._bigraph._adj
        u, v = edge
        if not (0 <= u < len(A) and 0 <= v < len(A)): return False
        if A[u][u] == 0 or A[v][v] == 0: return False
        return A[u][v] !=0 and A[v][u] != 0

    def __iter__(self):
        A = self._bigraph._adj
        for u in range(len(A) - 1):
            for v in range(u + 1, len(A)):
                if A[u][v] != 0 and A[v][u] != 0:
                    yield (u, v)

    def __len__(self):
        return sum(1 for e in self)

class BigraphOrientationSetView(Set):
    def __init__(self, G):
        self._bigraph = G

    def __contains__(self, arc):
        A = self._bigraph._adj
        u, v = arc
        if not (0 <= u < len(A) and 0 <= v < len(A)): return False
        if A[u][v] == 0 or A[v][u] == 0: return False
        return abs(A[u][v]) < abs(A[v][u])

    def __iter__(self):
        A = self._bigraph._adj
        for u in range(len(A) - 1):
            for v in range(u + 1, len(A)):
                if A[u][v] != A[v][u]:
                    if abs(A[u][v]) < abs(A[v][u]):
                        yield (u, v)
                    else: yield (v, u)

    def __len__(self):
        return sum(1 for a in self)

class BigraphNeighborsView(Set):
    def __init__(self, G, u):
        self._bigraph = G
        self._u = u

    def __contains__(self, v):
        A, u = self._bigraph._adj, self._u
        return 0 <= v < len(A) and 0 <= v < len(A) and A[u][v]*A[v][u] != 0

    def __iter__(self):
        A, u = self._bigraph._adj, self._u
        if 0 <= u < len(A):
            for v in range(u + 1, len(A)):
                if A[u][v]*A[v][u] != 0:
                    yield v

    def __len__(self):
        return sum(1 for v in self)

class BigraphLinestyleView(Mapping):
    def __init__(self, G):
        self._bigraph = G

    def __getitem__(self, edge):
        A = self._bigraph._adj
        u, v = edge
        if not (0 <= u < len(A) and 0 <= v < len(A)) or  A[u][v]*A[v][u] == 0:
            raise KeyError
        return LineStyle.solid if A[u][v] < 0 else LineStyle.dotted

    def __iter__(self):
        A = self._bigraph._adj
        for u in range(len(A) - 1):
            for v in range(u + 1, len(A)):
                if A[u][v]*A[v][u] != 0:
                    yield (u, v)

    def __len__(self):
        return sum(1 for v in self)

class Bigraph:
    def __init__(self):
        self._adj = []
        self._n = 0

    def _set_size(self, n):
        A = self._adj
        if n > len(A):
            for i in range(len(A)):
                A[i].extend(array('b', (0 for j in range(n - len(A)))))
            for i in range(n - len(A)):
                A.append(array('b', (0 for j in range(n))))
        elif len(A) > n:
            self._adj = A = A[:n]
            for i in range(n):
                A[i] = A[i][:n]

    def add(self, *path):
        A = self._adj
        vmax = max(path[i] for i in range(0, len(path), 2))
        if vmax >= len(A):
            self._set_size(vmax + 1)
        u = int(path[0])
        A[u][u] = 2
        for i in range(1, len(path), 2):
            e = Stroke(path[i])
            v = int(path[i + 1])
            A[v][v] = 2
            (x, y) = (v, u) if e.is_backward else (u, v)
            A[x][y] += e.linestyle
            A[y][x] += e.linestyle*(1 + int(e.is_directed))
            u = v

    def flation(self, s, r):
        A = self._adj
        n = len(A)
        c, d = -A[r][s], -A[s][r]
        for j in range(n):
            A[r][j] += c*A[s][j]
        for i in range(n):
            A[i][r] += d*A[i][s]

    def swap(self, s, r):
        A = self._adj
        n = len(A)
        for j in range(n):
            A[r][j], A[s][j] = A[s][j], A[r][j]
        for i in range(n):
            A[i][r], A[i][s] = A[i][s], A[i][r]

    def delete_vertex(self, v):
        A = self._adj
        for i in range(len(A)):
            A[v][i] = 0
            A[i][v] = 0

    def delete_edge(self, u, v):
        A = self._adj
        A[u][v] = 0
        A[v][u] = 0

    def matrix(self):
        return [list(row) for row in self._adj]

    def vertices(self):
        return BigraphVertexSetView(self)

    def edges(self):
        return BigraphEdgeSetView(self)

    def orientation(self):
        return BigraphOrientationSetView(self)

    def neighbors(self, u):
        return BigraphNeighborsView(self, u)

    def line_style(self):
        return BigraphLinestyleView(self)

if __name__ == '__main__':
    n = 10
    G = Bigraph()
    G.add(*('--' if bool(i % 2) else i//2 for i in range(2*n - 1)))
