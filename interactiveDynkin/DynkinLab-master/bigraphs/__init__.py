from collections import namedtuple
from enum import Enum

__all__ = ['Edge', 'Style', 'Dir', 'Bigraph']

Edge = namedtuple('Edge', ['tail', 'head', 'style', 'label', 'dir'])


class Style(str, Enum):
    dotted = 'dotted'
    solid = 'solid'


class Dir(str, Enum):
    forward = 'forward'
    back = 'back'
    none = 'none'
    both = 'both'


class Bigraph:
    def __init__(self):
        self._adj = dict()

    @property
    def vertices(self):
        return self._adj.keys()

    @property
    def edges(self):
        visited = set()
        for u in self.vertices:
            for v in self.neighbours(u):
                if v not in visited:
                    yield self.get_edge(u, v)
            visited.add(u)

    def neighbours(self, u):
        yield from (v for v in self._adj[u] if u != v)

    def get_edge(self, u, v):
        A, u, v = self._adj, str(u), str(v)
        if u == v or u not in A or v not in A or v not in A[u]:
            return None
        assert self._check_pair(u, v)
        style = Style('solid' if A[u][v] <= 0 else 'dotted')
        dir_, label = Dir.none.value, ''
        if abs(A[u][v]) < abs(A[v][u]):
            dir_ = Dir.forward
            label = abs(A[v][u])
        elif abs(A[u][v]) > abs(A[v][u]):
            dir_ = Dir.back
            label = abs(A[u][v])
        return Edge(u, v, style, label, dir_)

    def add_vertex(self, u):
        u = str(u)
        self._adj.setdefault(u, {u: 2})

    def add_edge(self, edge: Edge):
        A = self._adj
        u, v, style, label, dir_ = edge
        self.add_vertex(u)
        self.add_vertex(v)
        A[u][v] = A[v][u] = 1 if style == Style.dotted else -1
        if label > 0:
            if dir_ in (Dir.forward, Dir.none):
                A[v][u] *= label
            if dir_ in (Dir.back, Dir.none):
                A[u][v] *= label

    def _check_pair(self, i, j):
        a, b = self._adj[i][j], self._adj[j][i]
        return ((a >= 0) == (b >= 0)) \
               and (a != 0) and (b != 0) \
               and ((a == b) or (1 in [abs(a), abs(b)]))

    def flation(self, s, r, validate=True):
        A, s, r, flag = self._adj, str(s), str(r), True
        if s not in A or r not in A:
            return None

        (σ, τ) = (2, 2) if s == r else (A[s][r], A[r][s])

        for j in A[s]:  # Restar τ veces el renglón s al renglón r
            A[r].setdefault(j, 0)
            A[r][j] -= τ * A[s][j]

        for i in A[s]:  # Restar σ veces la columna s a la columna r
            A[i].setdefault(r, 0)
            A[i][r] -= σ * A[i][s]

        for x in list(A[r]):  # Borrar las aristas de peso cero
            if A[x][r] == 0 and A[r][x] == 0:
                del A[r][x]
                del A[x][r]
        if validate and not all(self._check_pair(i, r) for i in A[r]):
            flag = False
            self.flation(s, r, validate=False)  # Deshacer la flación
        return flag
