from abc import ABC, abstractmethod
from enum import Enum

from bigraphs import Bigraph, Style, Dir
from bigraphs.drawing import Embedding2D, Point2D

__all__ = ('DOTDescriptor', 'TikZDescriptor', 'MatrixDescriptor',
           'PolynomialDescriptor')


class AbstractDescriptor(ABC):
    def __init__(self, bigraph: Bigraph):
        self._bigraph = bigraph
        self._vertices = sorted(self.bigraph.vertices)
        self._index_of = {u: i for i, u in enumerate(self._vertices)}

    @property
    def bigraph(self):
        return self._bigraph

    @abstractmethod
    def describe(self):
        raise NotImplementedError


class DOTDescriptor(AbstractDescriptor):
    def __init__(self, bigraph: Bigraph, tab='\t'):
        super().__init__(bigraph)
        self._tab = tab

    def describe(self):
        G, L, tab = self.bigraph, [], self._tab
        if hasattr(G, 'name'):
            L.append('graph {} {{'.format(getattr(G, 'name')))
        else:
            L.append('graph {')

        stmt = tab + '{};'
        for v in G.vertices:
            L.append(stmt.format(v))

        visited = set()
        stmt = tab + '{} -- {}{};'
        for u in G.vertices:
            for v in G.neighbours(u):
                if v not in visited:
                    args = []
                    e = G.get_edge(u, v)
                    if e.style != Style.solid:
                        args.append('style = {}'.format(e.style.value))
                    if e.dir != Dir.none:
                        args.append('dir = {}'.format(e.dir.value))
                    if e.label != '':
                        args.append('label = "{}"'.format(e.label))
                    if len(args) > 0:
                        args = ' [{}]'.format(', '.join(args))
                    else:
                        args = ''
                    L.append(stmt.format(u, v, args))
            visited.add(u)
        L.append('}')
        return '\n'.join(L)


class TikZDescriptor(AbstractDescriptor):
    def __init__(self, embedding: Embedding2D, tab='  '):
        super().__init__(embedding.graph)
        self._embedding = embedding
        self.tab = tab
        self.auto_update = True
        self.rounding = 2

    @property
    def embedding(self):
        return self._embedding

    def describe(self):
        if self.auto_update:
            self.embedding.update()
        bigraph, position = self.bigraph, self.embedding.positions
        vertices = sorted(bigraph.vertices)
        index = {u: i for i, u in enumerate(vertices)}
        rounded = lambda P: Point2D(round(P.x, self.rounding),
                                    round(P.y, self.rounding))
        d = self.rounding
        lines = [r'\begin{tikzpicture}']
        stmt = self.tab + r'\node (v{i}) at ({pos.x}, {pos.y}) {{{label}}};'
        for i, u in enumerate(vertices):
            lines.append(stmt.format(i=i, pos=rounded(position[u]), label=u))

        stmt = self.tab + r'\draw{args} (v{i}) -- (v{j});'
        visited = set()
        for u in vertices:
            i = index[u]
            for v in self.bigraph.neighbours(u):
                if v in visited: continue
                j, args, e = index[v], [], bigraph.get_edge(u, v)
                if e.style == Style.dotted:
                    args.append('dotted')
                if e.dir == Dir.forward:
                    args.append('->')
                elif e.dir == Dir.back:
                    args.append('<-')
                args = '[{}]'.format(', '.join(args)) if args else ''
                lines.append(stmt.format(args=args, i=i, j=j))
        lines.append(r'\end{tikzpicture}')
        return '\n'.join(lines)


class MatrixLang(Enum):
    tabular = 'tabular'
    python = 'python'
    latex = 'latex'
    maxima = 'maxima'
    csv = 'csv'


class MatrixDescriptor(AbstractDescriptor):
    def __init__(self, bigraph: Bigraph, language=MatrixLang.tabular):
        super().__init__(bigraph)
        self.language = language

    def _get_matrix(self):
        vertices = self._vertices
        n = len(vertices)
        adj = self.bigraph._adj
        A = [[adj[vertices[j]].get(vertices[i], 0) for j in range(n)]
             for i in range(n)]
        return A

    def describe(self):
        return {
            MatrixLang.tabular: self._get_natural_description,
            MatrixLang.latex: self._get_latex_description,
            MatrixLang.maxima: self._get_maxima_description,
            MatrixLang.python: self._get_python_description,
            MatrixLang.csv: self._get_csv_description
        }[self.language]()

    def _get_maxima_description(self):
        A = self._get_matrix()
        lines = [repr(row) for row in A]
        return 'matrix({})'.format(', '.join(lines))

    def _get_latex_description(self):
        A = self._get_matrix()
        tab = ' ' * 2
        lines = [r'\begin{bmatrix}']
        for row in A:
            lines.append(r'{}{}\\'.format(tab, ' & '.join(map(str, row))))
        lines.append(r'\end{bmatrix}')
        return '\n'.join(lines)

    def _get_python_description(self):
        A = self._get_matrix()
        lines = [repr(row) for row in A]
        return '[{}]'.format(',\n '.join(lines))

    def _get_csv_description(self):
        A = self._get_matrix()
        lines = [','.join(map(str, row)) for row in A]
        return '\n'.join(lines)

    def _get_natural_description(self):
        vertices, A = self._vertices, self._get_matrix()
        n = len(vertices)
        tableau = [['' for j in range(n + 1)] for i in range(n + 1)]
        for i, u in enumerate(vertices):
            tableau[0][i + 1] = tableau[i + 1][0] = str(u)
        for i in range(n):
            for j in range(n):
                tableau[i + 1][j + 1] = str(A[i][j])
        column_width = [max(len(tableau[i][j]) for i in range(n + 1))
                        for j in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                tableau[i][j] = tableau[i][j].rjust(column_width[j])
        lines = ['{}│{}'.format(tableau[0][0], ' '.join(tableau[0][1:])),
                 '{}┼{}'.format('─' * column_width[0],
                                '─' * (sum(column_width[1:]) + n - 1))]
        for i in range(n):
            lines.append('{}│{}'.format(tableau[i + 1][0],
                                        ' '.join(tableau[i + 1][1:])))
        return '\n'.join(lines)


class PolynomialLanguage(Enum):
    latex = 'latex'
    maxima = 'maxima'
    python = 'python'


class BiTerm:
    def __init__(self, coefficient, i, j, x, language, is_first=False):
        self.language = language
        self.x = x
        self.is_first = is_first
        self.j = j
        self.i = i
        self.c = coefficient

    def __str__(self):
        if self.language == PolynomialLanguage.latex:
            x = lambda i: '{}_{{{}}}'.format(self.x, i)
            mult, pow = '*', '^'
        elif self.language == PolynomialLanguage.maxima:
            x = lambda i: '{}[{}]'.format(self.x, i)
            mult, pow = '*', '^'
        else:
            x = lambda i: '{}[{}]'.format(self.x, i)
            mult, pow = '*', '**'
        i, j = self.i, self.j
        monomial = x(i) + pow + '2' if i == j else x(i) + mult + x(j)
        if abs(self.c) == 1:
            coefficient = ''
        else:
            coefficient = str(abs(self.c)) + mult
        if self.is_first:
            sign = '' if self.c >= 0 else '-'
        else:
            sign = '+ ' if self.c >= 0 else '- '
        return ''.join([sign, coefficient, monomial])


class PolynomialDescriptor(AbstractDescriptor):
    def __init__(self, bigraph: Bigraph,
                 language: PolynomialLanguage = PolynomialLanguage.maxima,
                 variable: str = 'x') -> 'PolynomialDescriptor':
        super().__init__(bigraph)
        self.language = language
        self.variable = variable
        self.index_start = 0 if language == PolynomialLanguage.python else 1

    def describe(self):
        A = self.bigraph._adj
        vertices, index_of = self._vertices, self._index_of
        x, i0, language = self.variable, self.index_start, self.language
        terms = [BiTerm(A[u][u], i, i, x, language)
                 for i, u in enumerate(vertices, i0)]
        terms[0].is_first = True
        for i, u in enumerate(vertices, i0):
            for v in A[u]:
                if u == v: continue
                j = index_of[v] + i0
                terms.append(BiTerm(A[u][v], i, j, x, language))
        return ' '.join(map(str, terms))
