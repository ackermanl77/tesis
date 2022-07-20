import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple, Iterable

from bigraphs import Bigraph, Style

Vertex = str

__all__ = {'Inflations', 'Deflations'}


class ABCBigraphAlgorithm(ABC):
    def __init__(self, bigraph: Bigraph, stdout=None):
        self.bigraph = bigraph
        self.stdout = stdout

    def run(self):
        raise NotImplementedError()


class ABCFlationsAlgorithm(ABCBigraphAlgorithm):
    def __init__(self, bigraph: Bigraph, stdout=None):
        super().__init__(bigraph, stdout)
        self.flation_queue = deque()
        self.factorization = []

    @abstractmethod
    def select_edges(self) -> Iterable[Tuple[Vertex, Vertex]]:
        raise NotImplementedError()

    def flation_succeeded(self, s, r):
        if len(self.factorization) == 0 or (s, r) != self.factorization[-1]:
            self.factorization.append((s, r))
        else:
            self.factorization.pop()
        if self.stdout is not None:
            print('T', s, r, file=self.stdout)

    def flation_failed(self, s, r):
        pass

    def run(self):
        while True:
            self.flation_queue.extend(self.select_edges())
            if len(self.flation_queue) == 0:
                break
            while len(self.flation_queue) > 0:
                s, r = self.flation_queue.popleft()
                if self.bigraph.flation(s, r):
                    self.flation_succeeded(s, r)
                else:
                    self.flation_failed(s, r)


class Inflations(ABCFlationsAlgorithm):
    def select_edges(self) -> Iterable[Tuple[Vertex, Vertex]]:
        edges = list(e for e in self.bigraph.edges if e.style == Style.dotted)
        if len(edges) > 0:
            e = random.choice(edges)
            s, r = e.tail, e.head
            if random.getrandbits(1):
                s, r = r, s
            yield (s, r)


Inflations.__doc__ = _('''\
Apply flations over dotted edges until no dotted edges remain.
In each step a dotted edge is selected at random.
This method is guaranteed to finish in a finite number of steps
for positive definite bigraphs only.''')


class Deflations(ABCFlationsAlgorithm):
    def select_edges(self) -> Iterable[Tuple[Vertex, Vertex]]:
        edges = list(e for e in self.bigraph.edges if e.style == Style.solid)
        if len(edges) > 0:
            e = random.choice(edges)
            s, r = e.tail, e.head
            if random.getrandbits(1):
                s, r = r, s
            yield (s, r)


Deflations.__doc__ = _('''\
Apply flations over solid edges until no solid edges remain.
In each step a solid edge is selected at random.''')
