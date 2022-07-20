import tkinter as tk
from collections import namedtuple
from collections.abc import Mapping
from itertools import combinations
from math import sqrt, pi, cos, sin
from numbers import Real
from random import uniform

from bigraphs import Bigraph, Dir, Style

__all__ = ['Point2D', 'Embedding2D', 'TkBigraphPlot']


class Point2D(namedtuple('Point2D', ['x', 'y'])):
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Real):
        return Point2D(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __truediv__(self, other: Real):
        return Point2D(self.x / other, self.y / other)

    __rtruediv__ = __truediv__

    def __abs__(self):
        return sqrt(self.x * self.x + self.y * self.y)

    @staticmethod
    def from_polar(length: Real, angle: Real):
        return length * Point2D(cos(angle), sin(angle))


class Embedding2D:
    def __init__(self, graph):
        self._graph = graph
        self._positions = dict()
        self.new_vertex_params = dict(iterations=50, temperature=1.0,
                                      distance_threshold=2)
        self.old_vertex_params = dict(iterations=6, temperature=0.5,
                                      distance_threshold=2)
        self.update()

    @property
    def graph(self) -> Bigraph:
        return self._graph

    @property
    def positions(self):
        return self._positions.copy()

    @positions.setter
    def positions(self, mapping):
        if not issubclass(mapping, Mapping): raise TypeError
        self._positions = {u: Point2D(*p) for (u, p) in mapping.items()}

    def update(self):
        vertices = set(self.graph.vertices)
        positions = self._positions
        for u in set(positions).difference(vertices):
            del positions[u]

        if len(vertices) == 1:
            positions.clear()
            positions[vertices.pop()] = Point2D(0.0, 0.0)
            return
        elif len(positions) <= 1:
            positions.clear()
            n = len(vertices)
            for k, u in enumerate(vertices):
                positions[u] = Point2D.from_polar(0.1, (2 * pi) * (k / n))
            self._fruchterman_reingold(vertices, **self.new_vertex_params)
            return

        new_vertices = vertices.difference(positions)
        if len(new_vertices) > 0:
            top = min(positions[u].y for u in positions)
            left = min(positions[u].x for u in positions)
            bottom = max(positions[u].y for u in positions)
            right = max(positions[u].x for u in positions)
            for u in new_vertices:
                positions[u] = Point2D(uniform(top, bottom),
                                       uniform(left, right))
            self._fruchterman_reingold(new_vertices, **self.new_vertex_params)
        self._fruchterman_reingold(**self.old_vertex_params)

    def _fruchterman_reingold(self, move_only=None, iterations=50,
                              temperature=1.0, distance_threshold=2):

        graph, position = self.graph, self._positions
        squared_threshold = distance_threshold ** 2
        if move_only is None:
            moving_vertices = list(graph.vertices)
        else:
            moving_vertices = list(move_only)
        cooling_value = temperature / iterations
        for i in range(iterations):
            displacement = {u: Point2D(0, 0) for u in graph.vertices}
            for u, v in combinations(graph.vertices, 2):
                directrix = position[v] - position[u]
                squared_distance = directrix.x ** 2 + directrix.y ** 2
                if squared_distance < squared_threshold:
                    repulsion_vector = directrix / squared_distance
                    displacement[v] += repulsion_vector
                    displacement[u] -= repulsion_vector
            for e in graph.edges:
                u, v = e.tail, e.head
                directrix = position[v] - position[u]
                attraction_vector = abs(directrix) * directrix
                displacement[u] += attraction_vector
                displacement[v] -= attraction_vector
            for u in moving_vertices:
                distance = abs(displacement[u])
                if distance == 0.0: continue
                position[u] += (displacement[u] / distance) * min(distance,
                                                                  temperature)
            temperature -= cooling_value
        return position


class TkBigraphPlot:
    def __init__(self, embedding: Embedding2D, **kwargs):
        self.title = 'Bigraph drawing'
        self.background = 'white'
        self.vertex_fill = 'lightgray'
        self.vertex_outline = ''
        self.vertex_text_fill = 'black'
        self.vertex_text_font = ('times', 14)
        self.edge_fill = 'black'
        self.edge_text_fill = 'gray'
        self.dotted_edge_dash = (3, 3)
        self.edge_arrowshape = (8, 10, 3)
        self.edge_width = '1'
        self.edge_length = 92
        self.padding = 8
        self.auto_resize = True
        self.__dict__.update(kwargs)
        self._embedding = embedding

    @property
    def embedding(self):
        return self._embedding

    def show(self):
        window = tk.Tk()
        window.title(self.title)
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        self._canvas = canvas = tk.Canvas(window)
        if len(self.embedding.graph.vertices) > 0:
            self._update_geometry()
            self._compute_initial_canvas_size()
            canvas.bind('<Configure>', lambda e: self._redraw())
            window.bind('<space>', lambda e: self._redraw(update=True))
        canvas.configure(background=self.background)
        canvas.grid(column=0, row=0, sticky=tk.NSEW)
        window.bind('<Return>', lambda e: window.destroy())
        window.mainloop()
        del self._canvas

    def _compute_initial_canvas_size(self):
        position, radius = self._vertex_positions, self._vertex_radiuses
        canvas, scale = self._canvas, self.edge_length
        left = position[self._leftmost_vertex].x
        right = position[self._rightmost_vertex].x
        top = position[self._topmost_vertex].y
        bottom = position[self._bottommost_vertex].y
        const = 2 * self.padding
        width_border = radius[self._leftmost_vertex] + radius[
            self._rightmost_vertex] + const
        height_border = radius[self._topmost_vertex] + radius[
            self._bottommost_vertex] + const

        initial_width = scale * (right - left) + width_border
        initial_height = scale * (bottom - top) + height_border
        canvas.configure(width=initial_width, height=initial_height)

    def _update_geometry(self):
        canvas = self._canvas
        self._vertex_positions = position = dict(self.embedding.positions)
        vertices = position.keys()
        self._edges = list(self.embedding.graph.edges)

        self._topmost_vertex = min(vertices, key=lambda u: position[u].y)
        self._bottommost_vertex = max(vertices, key=lambda u: position[u].y)
        self._leftmost_vertex = min(vertices, key=lambda u: position[u].x)
        self._rightmost_vertex = max(vertices, key=lambda u: position[u].x)

        origin = Point2D(position[self._leftmost_vertex].x,
                         position[self._topmost_vertex].y)
        for u in vertices:
            position[u] -= origin

        self._vertex_radiuses = radius = dict.fromkeys(vertices)
        aux = canvas.create_text(0, 0, fill=self.background,
                                 font=self.vertex_text_font)
        for u in vertices:
            canvas.itemconfig(aux, text=str(u))
            x1, y1, x2, y2 = canvas.bbox(aux)
            radius[u] = max(x2 - x1, y2 - y1) / 2
        canvas.delete(aux)

    def _redraw(self, update=False):
        if update:
            self.embedding.update()
            self._update_geometry()
        if self.auto_resize:
            self._compute_initial_canvas_size()
        canvas = self._canvas
        canvas.delete('all')
        position, radius = self._vertex_positions, self._vertex_radiuses
        vertices = position.keys()
        edges = self._edges
        height = canvas.winfo_height() - 2 * self.padding
        width = canvas.winfo_width() - 2 * self.padding
        origin_x = radius[self._leftmost_vertex] + self.padding
        origin_y = radius[self._topmost_vertex] + self.padding
        scale_x = (width - radius[self._leftmost_vertex] - radius[
            self._rightmost_vertex]) / position[self._rightmost_vertex].x
        scale_y = (height - radius[self._topmost_vertex] - radius[
            self._bottommost_vertex]) / position[self._bottommost_vertex].y

        scaled_position = dict()
        for u in vertices:
            x, y = position[u]
            scaled_position[u] = Point2D(scale_x * x + origin_x,
                                         scale_y * y + origin_y)

        for e in edges:
            u = e.tail
            v = e.head
            self._draw_edge(scaled_position[u], scaled_position[v], e)

        for u in vertices:
            self._draw_vertex(u, scaled_position[u], radius[u])

    def _draw_vertex(self, u: str, position: Point2D, radius: float):
        canvas = self._canvas
        x0, y0 = position - Point2D(radius, radius)
        x1, y1 = position + Point2D(radius, radius)
        canvas.create_oval(x0, y0, x1, y1, fill=self.vertex_fill,
                           outline=self.vertex_outline)
        canvas.create_text(position.x, position.y, text=u,
                           fill=self.vertex_text_fill,
                           font=self.vertex_text_font)

    def _draw_edge(self, tail_position, head_position, edge):
        canvas, radius = self._canvas, self._vertex_radiuses
        directrix = head_position - tail_position
        directrix /= abs(directrix)
        x0, y0 = tail_position + directrix * radius[edge.tail]
        x1, y1 = head_position - directrix * radius[edge.head]
        kwargs = dict(fill=self.edge_fill, width=self.edge_width)
        if edge.dir == Dir.forward:
            kwargs.update(arrow=tk.LAST, arrowshape=self.edge_arrowshape)
        elif edge.dir == Dir.back:
            kwargs.update(arrow=tk.FIRST, arrowshape=self.edge_arrowshape)
        if edge.style == Style.dotted:
            kwargs.update(dash=self.dotted_edge_dash)
        canvas.create_line(x0, y0, x1, y1, **kwargs)
