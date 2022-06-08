from cmd import Cmd
from bigraph import Stroke
from bigraph_lib import *
import collections
import itertools
import re

Token = collections.namedtuple('Token', ['typ', 'value', 'column'])

def tokenize(code):
    keywords = {'True', 'False', 'None'}
    token_specification = [
        ('DECIMAL', r'\d+\.\d+'),
        ('INTEGER',  r'\d+'),           # Integer
        ('STROKE',  r'\-\-|<[\-\.]|[\-\.]>|\.\.'), #Strokes
        ('ID',  r'[A-Za-z]+'),           # Identifiers
        ('LEFTBRACKET',   r'\['),       # Grouping
        ('RIGHTBRACKET',   r'\]'),      # Grouping
        ('EQUAL',   r'='),              # 
        ('SKIP',    r'[ \t]+'),         # Skip over spaces and tabs
        ('MISMATCH',r'.')               # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group(kind)
        if kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise RuntimeError('%r unexpected on line %d' % (value, line_num))
        else:
            if kind == 'ID' and value in keywords:
                kind = value.upper()
            yield Token(kind, value, mo.start())

def parse_kwargs(arg):
    state, token, args, param = 0, None, dict(), ''
    def action_0_ID(): nonlocal param; param = token.value
    def action_1_ID(): nonlocal param; args[param] = True; param = token.value
    def action_1_EQUAL(): pass
    def action_2_DECIMAL(): nonlocal param; args[param] = float(token.value)
    def action_2_INTEGER(): nonlocal param; args[param] = int(token.value)
    def action_2_TRUE(): nonlocal param; args[param] = True
    def action_2_FALSE(): nonlocal param; args[param] = False
    def action_2_NONE(): nonlocal param; args[param] = None
    one_step = {(0, 'ID'): (1, action_0_ID),
                (1, 'ID'): (0, action_1_ID),
                (1, 'EQUAL'): (2, action_1_EQUAL),
                (2, 'DECIMAL'): (0, action_2_DECIMAL),
                (2, 'INTEGER'): (0, action_2_INTEGER),
                (2, 'TRUE'): (0, action_2_TRUE),
                (2, 'FALSE'): (0, action_2_FALSE),
                (2, 'NONE'): (0, action_2_NONE)}
    for token in tokenize(arg):
        state, action = one_step.get((state, token.typ), (None, None))
        if state is not None:
            action()
        else: break
    else: return args
    raise RuntimeError()

def parse_path(arg):
    path, state = [], 0
    for token in tokenize(arg):
        if state == 0 and token.typ == 'INTEGER':
            path.append(int(token.value))
            state = 1
        elif state == 1 and token.typ == 'STROKE':
            path.append(token.value)
            state = 0
        else: raise RuntimeError()
    if state == 1:
        return path
    else: raise RuntimeError()

def parse_mflation(arg):
    groups, current, state = [], [], 0
    for token in tokenize(arg):
        if state == 0:
            if token.typ == 'INTEGER':
                groups.append((int(token.value),))
            elif token.typ == 'LEFTBRACKET':
                current, state = [], 1
            else: raise RuntimeError()
        elif state == 1:
            if token.typ == 'INTEGER':
                current.append(int(token.value))
            elif token.typ == 'RIGHTBRACKET':
                groups.append(tuple(current))
                state = 0
            else: raise RuntimeError()
    if state == 0:
        return groups
    else: raise RuntimeError()

def parse_cycles(arg):
    cycles, current, state = [], [], 0
    for token in tokenize(arg):
        if state == 0:
            if token.typ == 'LEFTBRACKET':
                current, state = [], 1
            else: raise RuntimeError()
        elif state == 1:
            if token.typ == 'INTEGER':
                current.append(int(token.value))
            elif token.typ == 'RIGHTBRACKET':
                cycles.append(tuple(current))
                state = 0
            else: raise RuntimeError()
    if state == 0:
        return cycles
    else: raise RuntimeError()

def parse_dynkin_type(arg):
    state = 0
    for token in tokenize(arg):
        if state == 0:
            if token.typ == 'ID':
                _type = DynkinType(token.value)
                state = 1
            else: raise RuntimeError()
        elif state == 1:
            if token.typ == 'INTEGER':
                n = int(token.value)
                state = 2
            else: raise RuntimeError()
        else: raise RuntimeError()
    if state == 1:
        raise RuntimeError()
    _type, n = DynkinType(_type), int(n)
    if n <= 0 \
       or _type == DynkinType.E and not 6 <= n <= 8 \
       or _type == DynkinType.F and not n == 4 \
       or _type == DynkinType.G and not n == 2:
        raise ValueError()
    return _type, n

class BigraphShell(Cmd):
    def __init__(self):
        super().__init__()
        self.do_new('')

    def _print(self, *args, **kwargs):
        print(*args, file = self.stdout, **kwargs)

    def _update_position(self, update_all = False):
        bigraph, position, pi = self.bigraph, self.position, math.pi
        vertices = frozenset(bigraph.vertices())

        for u in list(position.keys()):
            if u not in vertices:
                del position[u]

        has_position = frozenset(position.keys())
        need_position = vertices.difference(has_position)

        if need_position:
            n, polar = len(need_position), Point.polar
            for (k, u) in enumerate(need_position):
                position[u] = polar(0.5, (2*pi)*(k/n))
            place_graph(bigraph, position, fixed = has_position)

        if update_all: place_graph(bigraph, position, temperature = 0.25)

    def do_new(self, arg):
        if arg:
            try:
                _type, n = parse_dynkin_type(arg)
            except:
                self._print('Not a valid Dynkin type: {}'.format(arg))
            else: self.bigraph = new_dynkin(_type, n)
        else: self.bigraph = Bigraph()
        self.position = dict()

    def do_add(self, arg):
        try:
            path = parse_path(arg)
        except:
            self._print('Not a valid path: {}'.format(arg))
        else: self.bigraph.add(*path)

    def do_plot(self, arg):
        if arg: print(parse_kwargs(arg))
        self._update_position(update_all = True)
        draw_bigraph(self.bigraph, self.position)

    def do_tikz(self, arg):
        self._update_position()
        self._print(bigraph_to_tikz(self.bigraph, self.position))

    def do_T(self, arg):
        try:
            args = parse_mflation(arg)

        except:
            self._print('Invalid arguments: {}'.format(arg))
        else:
            vertices = self.bigraph.vertices()
            for u in itertools.chain.from_iterable(args):
                if u not in vertices:
                    self._print('***Bigraph is missing vertex {}'.format(u))
                    break
            else: mflation(self.bigraph, *args)

    def do_inflate(self, arg):
        pass


if __name__ == '__main__':
    instance = BigraphShell()
    instance.cmdloop()
