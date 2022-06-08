from array import array
from os.path import splitext
from itertools import chain
from math import pi, sqrt, sin, cos
from cmath import rect
from random import uniform, choice, getrandbits, randint
from collections import deque
from tkinter import *
from tkinter import ttk

class bigraph:
    __slots__ = ('_adj', '_vert', '_idx')
        
    def __init__(self):
        self._adj = []
        self._vert = [] #vert[i]: name of vertex i
        self._idx = dict() #idx[v]: index of vertex v

    def __len__(self):
        return len(self._adj)

    @property
    def adj(self):
        return self._adj
    
    def vertices(self):
        return self._idx.keys()

    def _edges(self):
        A = self._adj
        for i in range(len(A) - 1):
            for j in range(i + 1, len(A)):
                if i != j and A[i][j] != 0:
                    dotted = (A[i][j] > 0)
                    direction = abs(A[j][i]) - abs(A[i][j])
                    if direction > 0:
                        e = '$>'
                    elif direction == 0:
                        e = '$$'
                    else: e = '<$'
                    if dotted:
                        e = e.replace('$', '.')
                    else: e = e.replace('$', '-')
                    yield (i, e, j)
                    
    def edges(self):
        for (i, e, j) in self._edges():
            yield (self._vert[i], e, self._vert[j])
        
    def _add_vertex(self, name):
        n = len(self._adj)
        self._idx[name] = n
        self._vert.append(name)
        for row in self._adj:
            row.append(0)
        self._adj.append(array('b', (0 for k in range(n + 1))))
        self._adj[n][n] = 2
        return n
    
    def _add_edge(self, u, v, e):
        i, j = self._idx[u], self._idx[v]
        if e == '--':
            self._adj[i][j] -= 1
            self._adj[j][i] -= 1
        elif e == '->':
            self._adj[i][j] -= 1
            self._adj[j][i] -= 2
        elif e == '<-':
            self._adj[i][j] -= 2
            self._adj[j][i] -= 1
        elif e == '..':
            self._adj[i][j] += 1
            self._adj[j][i] += 1
        elif e == '.>':
            self._adj[i][j] += 1
            self._adj[j][i] += 2
        elif e == '<.':
            self._adj[i][j] += 2
            self._adj[j][i] += 1
        else: raise ValueError
    
    def add(self, path):
        sentinel, item = object(), iter(path)
        u = next(item)
        if u not in self._idx: self._add_vertex(u)
        while True:
            e = next(item, sentinel)
            if e == sentinel:
                break
            if not isinstance(e, str):
                raise TypeError
            v = next(item)
            if u == v: raise ValueError
            if v not in self._idx: self._add_vertex(v)
            self._add_edge(u, v, e)
            u = v

    def delete(self, u):
        i = self._idx[u]
        del self._adj[i]
        for row in self._adj:
            del row[i]
        del self._vert[i]
        for (j, v) in enumerate(self._vert):
            self._idx[v] = j
        del self._idx[u]
    
    def shrink(self, X, v):
        if v not in self._idx:
            self._add_vertex(v)
        for u in X:
            if u == v: continue
            i = self._idx[u]
            j = self._idx[v]
            for (k, w) in enumerate(self._adj[i]):
                if k != j and w != 0:
                    self._adj[k][j] += self._adj[k][i]
                    self._adj[j][k] += self._adj[i][k]
            self.delete(u)

    def neighbors(self, u):
        i = self._idx[u]
        for (j, w) in enumerate(self._adj[i]):
            if w != 0:
                yield self._vert[j]

    def __str__(self, sort = False):
        A, n, idx, vert = self._adj, len(self._adj), self._idx, self._vert
        if n*n > 0:
            I = list(idx[v] for v in  sorted(self._idx)) if sort else range(n)
            N = [str(vert[i]) for i in I]
            B = [[str(A[i][j]) for j in I] for i in I]
            p = [max(max(len(B[i][j]) for i in I), len(N[j])) for j in I]
            N = [N[j].center(p[j]) for j in I]
            B = [[B[i][j].center(p[j]) for j in I] for i in I]
            B = [' '.join(B[i]) for i in I]
            N = ' ' + ' '.join(N)
        else: return '[]'
        if n > 1:
            B[0] = ''.join(['┌', B[0] ,'┐'])
            B[1 : n - 1] = [''.join(['│', B[i] ,'│']) for i in range(1, n - 1)]
            B[n - 1] = ''.join(['└', B[n - 1] ,'┘'])
            return '\n'.join([N, '\n'.join(row for row in B)])
        else: return '\n'.join([N, ''.join(['[', B[0] ,']'])])


    def _embed(self, pos = None, temp = 1.0, steps = None):
        #Uses Fruchterman-Reingold force embedder
        n = len(self._adj)
        if pos == None:
            pos = [rect(temp/2, uniform(-pi, pi)) for u in range(n)]
        if steps == None:
            steps = n*(n - 1) + 1
        cool = temp/steps
        #self.plot(pos = {self._vert[i]:(pos[i].real, pos[i].imag) for i in range(n)})
        for i in range(steps):
            disp = [complex(0.0, 0.0) for v in range(n)]
            for u in range(n - 1):
                for v in range(u + 1, n):
                    D = pos[v] - pos[u]
                    rf = D/(D.real**2 + D.imag**2)
                    disp[v] += rf
                    disp[u] -= rf
                    if self._adj[u][v] != 0:
                        D = pos[v] - pos[u]
                        af = abs(D)*D
                        disp[u] += af
                        disp[v] -= af
            for v in range(n):
                pos[v] += (disp[v]/abs(disp[v]))*min(abs(disp[v]), temp)
            #self.plot(pos = {self._vert[i]:(pos[i].real, pos[i].imag) for i in range(n)})
            temp -= cool
        return pos

    def embedding(self, start_pos = None, raw_output = False,
                  temp = 1.0, steps = None):
        idx = self._idx
        if len(self._adj) == 0: return dict()
        if start_pos != None:
            pos = [None]*len(self._adj)
            for (v, (x, y)) in start_pos.items():
                pos[idx[v]] = complex(x, y)
            start_pos = pos
        pos = self._embed(start_pos, temp, steps)
        if raw_output:
            return pos
        return {self._vert[i]:(p.real, p.imag) for (i, p) in enumerate(pos)}

    def _edge_partition(self):
        #Separates the edges in four types: 0. Nondirected solid edges
        #1. Directed solid edges 2. Nondirected dotted edges and
        #3. Directed dotted edges
        A = self._adj
        n = len(A)
        edges = [[], [], [], []]
        for i in range(n - 1):
            for j in range(i + 1, n):
                if A[i][j] == 0: continue
                edge_type = 2*int(A[i][j] > 0) + int(A[i][j] != A[j][i])
                if abs(A[i][j]) < abs(A[j][i]):
                    (u, v) = (i, j)
                else: (u, v) = (j, i) 
                edges[edge_type].append((u, v))
        return edges

    def tex(self, edge_length = 1.0, pos = None):
        def coords(z):
            return (round(z.real, 2), round(z.imag, 2))
        
        V = self._vert
        n = len(V)
        s = '    \\node (v{}) at {} {{{}}};'
        if pos == None:
            pos = self._embed()
            nodes = '\n'.join(s.format(i, coords(pos[i]), V[i])\
                              for i in range(n))
        else:
            pos = [pos[v] for v in self._vert]
            nodes = '\n'.join(s.format(i, pos[i], V[i]) for i in range(n))
        edges = self._edge_partition()
        for t in range(4):
            edges[t] = ', '.join('v{}/v{}'.format(u, v) for (u, v) in edges[t])
        return '''\\begin{{tikzpicture}}[scale = {0}, every node/.style={{circle, inner sep=1pt}}]
{1}
    \\foreach \\from/\\to in {{{2}}}
        \\draw (\\from) -- (\\to);
    \\foreach \\from/\\to in {{{3}}}
        \\draw[->] (\\from) -- (\\to);
    \\foreach \\from/\\to in {{{4}}}
        \\draw[dotted] (\\from) -- (\\to);
    \\foreach \\from/\\to in {{{5}}}
        \\draw[->, dotted] (\\from) -- (\\to);
\\end{{tikzpicture}}'''.format(edge_length, nodes, *edges)
    
    @staticmethod
    def load(file_name):
        (path, ext) = splitext(file_name)
        if not ext: ext = '.csv'
        file = open(path + ext)
        vert = [eval(item) for item in (file.readline().strip()).split(',')]
        adj = file.readlines()
        file.close()
        n = len(vert)
        if len(adj) != n:
            raise ValueError('File does not contain a valid bigraph')
        for k in range(n):
            adj[k] = array('b', map(int, (adj[k].strip()).split(',')))
            if len(adj[k]) != n or adj[k][k] != 2:
                raise ValueError('File does not contain a valid bigraph')
        G = bigraph()
        G._adj = adj
        G._vert = vert
        G._idx = {u:i for (i, u) in enumerate(vert)}
        return G

    def save(self, file_name):
        lines = [','.join(str(item) for item in row) for row in self._adj]
        (path, ext) = splitext(file_name)
        if not ext: ext = '.csv'
        file = open(path + ext, 'w')
        file.write(','.join(map(repr, self._vert)))
        file.write('\n')
        file.write('\n'.join(lines))
        file.close()

    def plot(self, typeface = 'times', size = 14, border = 8, thickness = 1,
             pos = None):
        def coords(z):
            return (z.real, z.imag)
        
        def redraw(event):
            canvas.delete('all')
            k = min((event.width - blft - brgt)/w,
                    (event.height - btop - bbot)/h)
            p = [k*z + b for z in pos]
            for (i, e, j) in elist:
                x1, y1 = coords(p[i])
                x2, y2 = coords(p[j])
                if '-' in e:
                    canvas.create_line(x1, y1, x2, y2)
                else:
                    canvas.create_line(x1, y1, x2, y2, dash = (2,))
                if ('>' in e) or ('<' in e):
                    if '<' in e: i, j = j, i
                    arg = p[i] - p[j]
                    arg = arg/abs(arg)
                    ort = complex(arg.imag, -arg.real)
                    ti0 = p[j] + rad[j]*arg
                    ti1 = ti0 + (0.5*size)*arg + (0.25*size)*ort
                    ti2 = ti0 + (0.25*size)*arg
                    ti3 = ti0 + (0.5*size)*arg - (0.25*size)*ort
                    x0, y0 = coords(ti0)
                    x1, y1 = coords(ti1)
                    x2, y2 = coords(ti2)
                    x3, y3 = coords(ti3)
                    canvas.create_polygon(x0, y0, x1, y1, x2, y2, x3, y3,
                                          fill = 'black', outline = '')
                    
            for i in range(n):
                x, y = coords(p[i])
                canvas.create_oval(x - rad[i], y - rad[i],
                                   x + rad[i], y + rad[i],
                                   fill = 'lightgray', outline = '')
                canvas.create_text(x, y, text = vlist[i],
                                   font = (typeface, size))
        
        if len(self._adj) == 0: raise RuntimeError('Nothing to plot!')
        if pos == None:
            pos = self._embed()
        else:
            C = lambda z: complex(z[0], z[1])
            pos = [C(pos[v]) for v in self._vert]
        vlist = list(self._vert)
        elist = list(self._edges())
        n, m = len(vlist), len(elist)

        ilft, irgt, itop, ibot = 0, 0, 0, 0
        for i in range(n):
            if pos[i].real < pos[ilft].real:
                ilft = i
            if pos[i].real > pos[irgt].real:
                irgt = i
            if pos[i].imag < pos[itop].imag:
                itop = i
            if pos[i].imag > pos[ibot].imag:
                ibot = i

        d = complex(pos[ilft].real, pos[itop].imag)
        for i in range(n):
            pos[i] -= d
        
        w = pos[irgt].real
        h = pos[ibot].imag
        
        root = Tk()
        root.title('Plot')
        root.columnconfigure(0, weight = 1)
        root.rowconfigure(0, weight = 1)
        canvas = Canvas(root, background = 'white')

        rad = [0.0 for i in range(n)]
        aux = canvas.create_text(0, 0, fill = 'white', font = (typeface, size))
        for i in range(n):
            canvas.itemconfig(aux, text = vlist[i])
            x1, y1, x2, y2 = canvas.bbox(aux)
            rad[i] = 0.5*sqrt((x2 - x1)**2 + (y2 - y1)**2)
        canvas.delete(aux)
        del aux

        blft, btop, brgt, bbot = (rad[i] + border \
                                  for i in (ilft, itop, irgt, ibot))
        b = complex(blft, btop)

        k = 5*size
        canvas.config(width = k*w + blft + brgt, height = k*h + btop + bbot)
        
        canvas.grid(column = 0, row = 0, sticky=(N, W, E, S))
        canvas.bind('<Configure>', redraw)
        root.bind('<Return>', lambda e: root.destroy())
        root.mainloop()

    def _flation(self, s, r):
        A = self._adj
        V = range(len(A))
        row = [-A[r][s]*A[s][k] for k in V]
        col = [-A[s][r]*A[k][s] for k in V]
        A[r][r] += A[s][s]*A[s][r]*A[r][s]
        for k in V:
            A[r][k] += row[k]
            A[k][r] += col[k]

    def flation(self, s, r):
        self._flation(self._idx[s], self._idx[r])

    def inversion(self, r):
        A, r = self._adj, self._idx[r]
        n = len(A)
        for k in range(n):
            A[k][r] *= -1
            A[r][k] *= -1

    def permutation(self, s, r):
        i, j = self._idx[s], self._idx[r]
        self._idx[s], self._idx[r] = j, i
        self._vert[i], self._vert[j] = r, s

    def frame(self):
        F = bigraph()
        F._vert = self._vert[:]
        F._idx = self._idx.copy()
        F._adj = [array('b', (-(a % 2) for a in row)) for row in self._adj]
        return F

    @staticmethod
    def finite_A(n):
        B = bigraph()
        path = lambda i: '--' if i % 2 else i//2 + 1
        B.add(path(i) for i in range(2*n - 1))
        return B

    @staticmethod
    def finite_B(n):
        B = bigraph.finite_A(n - 1)
        B.add([n - 1, '->', n])
        return B

    @staticmethod
    def finite_C(n):
        B = bigraph.finite_A(n - 1)
        B.add([n - 1, '<-', n])
        return B

    @staticmethod
    def finite_D(n):
        B = bigraph.finite_A(n)
        B.add([2, '..', 1, '--', 3])
        return B

    @staticmethod
    def finite_E(n):
        B = bigraph.finite_A(n)
        B.add([2, '..', 1, '--', 4])
        return B

    @staticmethod
    def finite_F(n):
        B = bigraph.finite_A(n)
        B.add([3, '..', 2, '->', 3])
        return B

    def reduce(self, plot = False):
        A, n, v = self._adj, len(self._adj), self._vert
        while True:
            dotted = [(i, j) for (i, e, j) in self._edges() if A[i][j] > 0]
            if not dotted: break
            (s, r) = choice(dotted)
            if bool(getrandbits(1)):
                s, r = r, s
            self._flation(s, r)
            yield v[s], v[r]

    def treeify(self):
        A, n, vname = self._adj, len(self._adj), self._vert
        V = [i for (i, u) in sorted(enumerate(self._vert),
                                    key = lambda t: t[1])]
        def dfs_chrodless_cycle(s):
            nonlocal A
            nonlocal V
            nonlocal n
            visited = [False for v in V]
            parent = [None for v in V]
            dist = [float('inf') for v in V]
            visited[s], dist[s] = True, 0
            Q = deque([s])
            flag = False
            while len(Q) > 0:
                u = Q.popleft()
                for v in V:
                    if bool(A[u][v] % 2) and u != v:
                        if not visited[v]:
                            visited[v] = True
                            parent[v] = u
                            dist[v] = dist[u] + 1
                            Q.append(v)
                        elif parent[u] != v:
                            flag = True
                            break
                if flag: break
            if not flag: return None
            #print('cross edge:', self._vert[u], self._vert[v], (dist[u], dist[v]))
            P = deque()
            if dist[v] > dist[u]:
                P.append(v)
                v = parent[v]
            while u != v:
                P.appendleft(u)
                P.append(v)
                u, v = parent[u], parent[v]
            P.appendleft(u)
            return list(P)

        #r = randint(0, n - 1)
        r = V[0]
        while True:
            p = dfs_chrodless_cycle(r)
            if p is None: break
            for k in range(1, len(p) - 1):
                self._flation(p[k], p[k + 1])
            yield list(vname[u] for u in p[1:])

    def deflate(self):
        pass

if __name__ == '__main__':
    G = bigraph()
    G.add([1, '..', 2, '--', 3, '.>', 4, '<-' ,2])
    for row in G._adj:
        print(list(row))
    print(G.tex())
    for e in G.edges():
        print(e)
    G.plot(size = 12)
