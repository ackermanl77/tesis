from bigraph import LineStyle
from bigraph import Stroke
from bigraph import Bigraph
from random import choice
from random import getrandbits
from random import shuffle
from itertools import product
from itertools import combinations
from enum import Enum
import pickle
import math
import cmath
import collections

import tkinter as tk
from tkinter import ttk

class DynkinType(Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'

def flate(G, h):
    edges, l = G.edges(), G.line_style()
    S = list(e for e in edges if l[e] == h)
    while S:
        r, s = choice(S)
        if getrandbits(1): r, s = s, r
        G.flation(s, r)
        yield s,r
        S = list(e for e in edges if l[e] == h)

def inflate(G):
    return flate(G, LineStyle.dotted)

def deflate(G):
    return flate(G, LineStyle.solid)

def mflation(G, *iterables):
    if len(iterables) == 1:
        for u in iterables[0]:
            G.flation(u, u)
    else:
        for i in range(len(iterables) - 1):
            for u, v in product(iterables[i], iterables[i + 1]):
                G.flation(u, v)

def strmatrix(M):
    m = len(M)
    n = len(M[0]) if m else 0
    S = [['' for j in range(n + 2)] for i in range(m + 2)]
    for i in range(m):
        S[i + 2][0] = str(i)
    for j in range(n):
        S[0][j + 2] = str(j)
    for i in range(m):
        for j in range(n):
            S[i + 2][j + 2] = str(M[i][j])
    for j in range(n + 2):
        column_width = max(len(S[i][j]) for i in range(m + 2)) + 1
        for i in range(m + 2):
            S[i][j] = S[i][j].rjust(column_width)
        S[1][j] = '─'*column_width
    for i in range(m + 2):
        S[i][1] = '│'
    S[1][1] = '┼'
    return '\n'.join(''.join(S[i]) for i in range(m + 2))

def new_dynkin(dynkin_type, n):
    _type = DynkinType(dynkin_type)
    assert isinstance(n, int) and n >= 0
    G = Bigraph()
    if _type == DynkinType.A:
        G.add(*('--' if bool(i % 2) else i//2 for i in range(2*n - 1)))
    elif _type == DynkinType.B:
        G.add(n - 2, '->', n - 1)
        G.add(*('--' if bool(i % 2) else i//2 for i in range(2*n - 3)))
    elif _type == DynkinType.C:
        G.add(n - 1, '->', n - 2)
        G.add(*('--' if bool(i % 2) else i//2 for i in range(2*n - 3)))
    elif _type == DynkinType.D:
        G.add(*('--' if bool(i % 2) else i//2 for i in range(2, 2*n - 1)))
        G.add(0, '--', 2)
    elif _type == DynkinType.E and 6 <= n <= 8:
        G.add(*('--' if bool(i % 2) else i//2 for i in range(2, 2*n - 1)))
        G.add(0, '--', 3)
    elif _type == DynkinType.F and n == 4:
        G.add(0, '--', 1, '->', 2, '--', 3)
    else: raise NotImplementedError
    return G

#---Bigraph drawing------------------------------------------------------------
class Point(collections.namedtuple('Point', 'x y')):
    __slots__ = ()

    def __add__(p, q):
        if isinstance(p, Point):
            return Point(p.x + q.x, p.y + q.y)
        return NotImplemented

    def __radd__(p, q):
        if isinstance(p, Point):
            return Point(p.x + q.x, p.y + q.y)
        return NotImplemented

    def __sub__(p, q):
        if isinstance(p, Point):
            return Point(p.x - q.x, p.y - q.y)
        else: return NotImplemented

    def __mul__(p, c):
        if isinstance(c, (int, float)):
            return Point(p.x*c, p.y*c)
        else: return NotImplemented

    def __rmul__(p, c):
        if isinstance(c, (int, float)):
            return Point(p.x*c, p.y*c)
        else: return NotImplemented

    def __truediv__(p, c):
        if isinstance(c, (int, float)):
            return Point(p.x/c, p.y/c)
        else: return NotImplemented

    def __neg__(p):
        return Point(-p.x, -p.y)
    
    def __abs__(p):
        return math.sqrt(p.x*p.x + p.y*p.y)

    def __complex__(p):
        return complex(p.x, p.y)

    def __round__(p, n):
        return Point(round(p.x, n), round(p.y, n))

    @staticmethod
    def polar(dist, angle):
        return Point(dist*math.cos(angle), dist*math.sin(angle))

    def normalize(p):
        return p/abs(p)

def place_graph(graph, position = None, fixed = None,
                iterations = 50, temperature = 1.0):
##  Based on FRUCHTERMAN, Thomas MJ; REINGOLD, Edward M. Graph drawing
##  by force-directed placement. Softw., Pract. Exper., 1991, vol. 21,
##  no 11, p. 1129-1164. http://dx.doi.org/10.1002/spe.4380211102
    vertices, edges = list(graph.vertices()), graph.edges()
    cool_down = temperature/iterations
    shuffle(vertices)
    if position is None:
        pi, n, f = math.pi, len(vertices), Point.polar
        position = {u:f(0.5, (2*pi)*(k/n)) for (k, u) in enumerate(vertices)}
    if fixed:
        fixed = frozenset(fixed)
    else: fixed = tuple()
    for i in range(iterations):
        displacement = {v:Point(0, 0) for v in vertices}
        for (u, v) in combinations(vertices, 2):
            if u in fixed and v in fixed: continue
            difference = position[v] - position[u]
            norm = difference.x**2 + difference.y**2
            if norm < 4:
                vec_repulsion = difference/norm
                displacement[v] += vec_repulsion
                displacement[u] -= vec_repulsion
            if (u, v) in edges:
                vec_atraction = abs(difference)*difference
                displacement[u] += vec_atraction
                displacement[v] -= vec_atraction
        for v in vertices:
            if v in fixed: continue
            distance = abs(displacement[v])
            if distance > temperature:
                displacement[v] *= temperature/distance
            position[v] += displacement[v]
        temperature -= cool_down
    return position

def bigraph_to_tikz(bigraph, position = None):
    if position is None: position = place_graph(bigraph)
    vertex_str = r'    \node (v{u}) at ({x}, {y}) {{$v_{u}$}};'
    edge_str = r'    \draw{options} (v{u}) -- (v{v});'
    format_options = lambda seq: '[{}]'.format(', '.join(seq)) if seq else ''
    lines = [r'\begin{tikzpicture}']
    vertices = bigraph.vertices()
    edges = bigraph.edges()
    orientation = bigraph.orientation()
    line_style = bigraph.line_style()
    for u in vertices:
        x, y = position[u]
        lines.append(vertex_str.format(u = u, x = x, y = y))
    for (u, v) in edges:
        options = []
        if (u, v) in orientation:
            options.append('->')
        elif (v, u) in orientation:
            u, v = v, u
            options.append('->')
        if line_style[u, v] == LineStyle.dotted:
            options.append('dotted')
        lines.append(edge_str.format(options = format_options(options),
                                     u = u, v = v))
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)
    
def draw_bigraph(bigraph, position = None, font_family = 'sans', font_size = 12,
                 font_color = 'black', margin = 8, line_width = 1,
                 dash_pattern = (3, 3), window_title = 'Bigraph',
                 fill_color = 'lightgray', line_color = 'black'):

    def redraw(event):
        canvas.delete('all')
        scale = min((event.width - margin_left - margin_right)/drawing_width,
                (event.height - margin_top - margin_bottom)/drawing_height)
        _position = {u:scale*position[u] + origin for u in vertices}
        for (u, v) in edges:
            if (u, v) in orientation:
                oriented = True
            elif (v, u) in orientation:
                u, v, oriented = v, u, True
            else: oriented = False
            uv = _position[v] - _position[u]
            direction = (uv)/abs(uv)
            tail = _position[u] + radius[u]*direction
            head = _position[v] - radius[v]*direction
            line_options = {'smooth': 1, 'width': line_width}
            if oriented: line_options['arrow'] = tk.LAST
            if line_style[u, v] == LineStyle.dotted:
                line_options['dash'] = dash_pattern
            canvas.create_line(tail.x, tail.y, head.x, head.y, **line_options)
        for u in vertices:
            x, y = _position[u].x, _position[u].y
            canvas.create_oval(x - radius[u], y - radius[u],
                               x + radius[u], y + radius[u],
                               fill = fill_color, outline = line_color)
            canvas.create_text(x, y, text = u, font = (font_family, font_size),
                               fill = font_color)

    vertices, edges = bigraph.vertices(), bigraph.edges()
    orientation, line_style = bigraph.orientation(), bigraph.line_style()
    if not vertices: raise RuntimeError()
    if position is None: position = place_graph(bigraph)
    leftmost = min(vertices, key = lambda u: position[u].x)
    rightmost = max(vertices, key = lambda u: position[u].x)
    topmost = min(vertices, key = lambda u: position[u].y)
    bottommost = max(vertices, key = lambda u: position[u].y)
    displacement = Point(position[leftmost].x, position[topmost].y)
    for u in vertices: position[u] -= displacement
    drawing_width = position[rightmost].x
    drawing_height = position[bottommost].y
    root = tk.Tk()
    
    root.title(window_title)
    root.columnconfigure(0, weight = 1)
    root.rowconfigure(0, weight = 1)
    canvas = tk.Canvas(root, background = 'white')
    canvas.bind('<Configure>', redraw)
    radius = {u:0.0 for u in vertices}
    aux = canvas.create_text(0, 0, fill = 'white', \
                             font = (font_family, font_size))
    for u in vertices:
        canvas.itemconfig(aux, text = u)
        x1, y1, x2, y2 = canvas.bbox(aux)
        radius[u] = 0.5*math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    canvas.delete(aux)
    margins=(radius[u]+margin for u in (leftmost,topmost,rightmost,bottommost))
    margin_left, margin_top, margin_right, margin_bottom = margins
    origin = Point(margin_left, margin_top)
    scale = 5*font_size
    canvas.config(width = scale*drawing_width + margin_left + margin_right,
                  height = scale*drawing_height + margin_top + margin_bottom)
    canvas.grid(column = 0, row = 0, sticky = tk.NSEW)
 
    root.bind('<Return>', lambda e: root.destroy())
    root.attributes('-topmost', True)
    root.focus_set()
    root.mainloop()

if __name__ == '__main__':
    G = new_dynkin(DynkinType.B, 11)
    for t in deflate(G): pass
    print(strmatrix(G.matrix()))
    print(bigraph_to_tikz(G))
    position = place_graph(G)
    draw_bigraph(G, position)
    for s, r in inflate(G):
        print(r'T_{{{}\, {}}}'.format(s, r))
        print(strmatrix(G.matrix()))
        place_graph(G, position, iterations = 50, temperature =0.25)
        print(bigraph_to_tikz(G, position))
        draw_bigraph(G, position)
