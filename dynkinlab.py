import gettext

from matplotlib.pyplot import plot  # Use pygettext.py and poedit for i18n

try:
    lang = gettext.translation('dynkinlab', localedir='locale')
    lang.install()
except FileNotFoundError:
    import builtins

    builtins.__dict__['_'] = _ = lambda x: x

import os
import re
import sys
from cmd import Cmd
from collections import namedtuple
from enum import Enum

from bigraphs import Bigraph, Edge, Dir, Style, algorithms
from bigraphs.descriptors import MatrixDescriptor, DOTDescriptor, MatrixLang, \
    TikZDescriptor, PolynomialDescriptor, PolynomialLanguage
from bigraphs.drawing import *

vertex_typ = ('id', 'number', 'string')


class Edgeop(Enum):
    dotted_none = '..'
    dotted_forward = '.>'
    dotted_back = '<.'
    solid_none = '--'
    solid_forward = '->'
    solid_back = '<-'

    @property
    def dir(self):
        if '>' in self.value:
            return Dir.forward
        elif '<' in self.value:
            return Dir.back
        return Dir.none

    @property
    def style(self):
        if '.' in self.value:
            return Style.dotted
        return Style.solid


class GraphLang(Enum):
    dot = 'dot'
    tikz = 'tikz'


Token = namedtuple('Token', ['typ', 'value', 'line', 'column'])


def tokenize(code):
    token_specification = [
        ('edgeop', r'\.\.|--|[\.-]>|<[\.-]'),
        ('id', r'[_a-zA-ZÈ-Ź]+[_a-zA-ZÈ-Ź0-9]*'),
        ('number', r'[-]?(\.[0-9]+|[0-9]+(\.[0-9]+)?)'),
        ('string', r'".*?"'),
        ('lparen', r'\('),
        ('rparen', r'\)'),
        ('newline', r'\n'),
        ('skip', r'[ \t]+'),
        ('mismatch', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group(kind)
        if kind == 'newline':
            line_start = mo.end()
            line_num += 1
        elif kind == 'skip':
            pass
        elif kind == 'mismatch':
            raise RuntimeError(_('{value} unexpected on line {line_num}'). \
                               format(value=value, line_num=line_num))
        else:
            column = mo.start() - line_start
            yield Token(kind, value, line_num, column)


def path_parse(code):
    tokens = list(tokenize(code))
    n = len(tokens)
    if not (n % 2 == 1 and all(
                tokens[i].typ in vertex_typ for i in range(0, n, 2)) and all(
            tokens[i].typ == 'edgeop' for i in range(1, n, 2))):
        raise SyntaxError()
    return list(token.value for token in tokens)


def bigraph_add_path(G, path_code):
    path = path_parse(path_code)
    if len(path) < 1: return None
    assert len(path) % 2 == 1
    u = path[0]
    G.add_vertex(u)
    for i in range(1, len(path), 2):
        edgeop = Edgeop(path[i])
        v = path[i + 1]
        G.add_vertex(v)
        label = 1 if edgeop.dir == Dir.none else 2
        G.add_edge(Edge(u, v, edgeop.style, label, edgeop.dir))
        u = v


def mflation_parse(code):
    tokens = list(tokenize(code))
    args = []
    opened_paren = False
    for token in tokens:
        if token.typ == 'lparen':
            opened_paren = True
            args.append([])
        elif token.typ == 'rparen':
            opened_paren = False
        elif token.typ in vertex_typ:
            if opened_paren:
                args[-1].append(token.value)
            else:
                args.append([token.value])
        else:
            raise SyntaxError()
    return args


def mflation(G: Bigraph, code: str):
    args = mflation_parse(code)
    skiped = []
    for i in range(len(args) - 1):
        S, R = args[i], args[i + 1]
        for s in S:
            for r in R:
                if not G.flation(s, r):
                    skiped.append((s, r))
    return skiped


class DynkinLab(Cmd):
    intro = _('''Welcome to DynkinLab 4.0.
Type “help” or “?” to list commands.''')
    ruler = '─'
    prompt = 'dynkinlab> '
    doc_header = _('Documented commands (type help <topic>):')
    misc_header = _('Miscellaneous help topics:')
    undoc_header = _('Undocumented commands:')
    nohelp = _('*** No help on %s')

    def __init__(self):
        super().__init__()
        self.do_restart()

    def default(self, line):
        self.history.pop()
        self.print(_('*** Unknown syntax: {line}').format(line=line))

    def do_add(self, args):
        try:
            bigraph_add_path(self.bigraph, args)
        except:
            return SyntaxError()

    do_add.__doc__ = _('''\
NAME
    add - Add a path to the current bigraph.

SYNOPSIS
    add <vertex>[<edge><vertex>]*

DESCRIPTION
    This command lets you add vertices and edges to the current bigraph in a
    compact manner, one path at a time.

    <vertex> can be any of:
        - A string of alphabetic characters, underscores ('_') or digits (0-9),
          not beginning with a digit;
        - An integer or decimal number;
        - Any double-quoted string ("...").

    <edge> Can be any of
        '--'            a solid undirected edge
        '->' or '<-'    a solid directed edge
        '..'            a dotted undirected edge
        '.>' or '<.'    a dotted directed edge

    This command replaces any edge previously added between the same pair of
    vertices.

EXAMPLE
    add x <- u .. v -> x -- w
    add 1 .> "2x" -- x''')

    def do_apply(self, arg: str):
        args = arg.strip().split()
        if args[0] not in algorithms.__all__:
            return SyntaxError()
        if len(args) > 1 and (len(args) != 2 or args[1] != 'verbose'):
            return SyntaxError()
        algorithm = getattr(algorithms, args[0])(
            bigraph=self.bigraph)
        if 'verbose' in args:
            algorithm.stdout = self.stdout
        algorithm.run()

    def help_apply(self):
        doc = _('''\
NAME
    apply - Apply some algorithm to the current bigraph

SYNOPSIS
    apply <algorithm> [verbose]

DESCRIPTION
    Runs the specified algorithm on the current bigraph. If “verbose” option is
    present then the algorithm is instructed to print any messages to the
    output.

    Current supported algorithms are:
{}''')
        lines = []
        for name in sorted(algorithms.__all__):
            lines.append('')
            lines.append('    ' + name)
            docstring = getattr(algorithms, name).__doc__
            for doc_line in docstring.splitlines():
                lines.append('        ' + doc_line)
        self.print(doc.format('\n'.join(lines)))

    def do_describe(self, args: str):
        args = args.split()
        if not args:
            args = 'as matrix in tabular'
        elif len(args) == 2:
            if args[0] != 'as':
                return SyntaxError()
            if args[1] == 'matrix':
                args.append('in tabular')
            elif args[1] == 'graph':
                args.append('in dot')
            elif args[1] == 'polynomial':
                args.append('in maxima')
        elif len(args) != 4:
            return SyntaxError()
        args = ' '.join(args)
        if args == 'as matrix in tabular':
            descriptor = MatrixDescriptor(bigraph=self.bigraph,
                                          language=MatrixLang.tabular)
        elif args == 'as matrix in python':
            descriptor = MatrixDescriptor(bigraph=self.bigraph,
                                          language=MatrixLang.python)
        elif args == 'as matrix in maxima':
            descriptor = MatrixDescriptor(bigraph=self.bigraph,
                                          language=MatrixLang.maxima)
        elif args == 'as matrix in latex':
            descriptor = MatrixDescriptor(bigraph=self.bigraph,
                                          language=MatrixLang.latex)
        elif args == 'as matrix in csv':
            descriptor = MatrixDescriptor(bigraph=self.bigraph,
                                          language=MatrixLang.csv)
        elif args == 'as graph in dot':
            descriptor = DOTDescriptor(bigraph=self.bigraph)
        elif args == 'as graph in tikz':
            descriptor = TikZDescriptor(embedding=self.embedding)
        elif args == 'as polynomial in latex':
            descriptor = PolynomialDescriptor(self.bigraph,
                                              language=PolynomialLanguage.latex)
        elif args == 'as polynomial in maxima':
            descriptor = PolynomialDescriptor(
                self.bigraph, language=PolynomialLanguage.maxima)
        elif args == 'as polynomial in latex':
            descriptor = PolynomialDescriptor(
                self.bigraph, language=PolynomialLanguage.latex)
        elif args == 'as polynomial in python':
            descriptor = PolynomialDescriptor(
                self.bigraph, language=PolynomialLanguage.python)
        else:
            return SyntaxError()
        self.print(descriptor.describe())

    do_describe.__doc__ = _('''\
NAME
    describe - Print a description of the bigraph.

SYNOPSIS
    describe
    describe as matrix [in <matrix language>]
    describe as graph [in <graph language>]
    describe as polynomial [in <polynomial language>]

DESCRIPTION
    Describe the current bigraph as plain text in some specified language. If
    no argument is given, it defaults to describe as a matrix in tabular form.
    Current supported languages are:

    Matrix languages: tabular, python, maxima, latex, csv
    Graph languages: dot, tikz
    Polynomial languages: python, maxima, latex

EXAMPLES
    describe as graph in tikz
    describe as polynomial in latex''')

    def do_flation(self, args):
        skipped = mflation(self.bigraph, args)
        if len(skipped) > 0:
            for s, r in skipped:
                self.print(
                    _('*** Skipping T {} {}').format(s, r))

    do_flation.__doc__ = _('''\
NAME
    flation - The flation morphism of bigraphs.
    T - Same as flation.

SYNOPSIS
    flation <vertex> <vertex>
    flation (<vertex>*)*

DESCRIPTION
    When given two vertices as arguments, the flation morphism is equivalent to
    aplying an elementary row adding transformation folowed by a column adding
    transformation to the adjacency matrix of the bigraph.
    Vertices may be grouped by parenthesis to abbreviate multiple flations.
    This morphism is guaranted to be closed on bigraphs by skipping those
    transformations that do not yield bigraphs.

EXAMPLE
    T 2 3
        Equivalent to flation 2 3
    T (x y) (z w)
        Abbreviation of T x z; T x w; T y z; T y w''')

    def do_load(self, filename: str):
        try:
            with open(filename, mode='r', encoding='UTF-8') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            self.print(_('*** File not found: {filename}').format(
                filename=filename))
        else:
            self.do_restart()
            self.cmdqueue.extend(lines)

    do_load.__doc__ = _('''\
NAME
    load - Load and execute commands from a file.

EXAMPLE
    load foobar.dyn''')

    def do_plot(self, args):
        self.embedding.update()
        self.plot2D.show()

    do_plot.__doc__ = '''\
NAME
    plot - Draw the current bigraph on the computer screen.

DESCRIPTION
    The plot command opens a plot window which contains a drawing of the current
    bigraph. The layout of vertices is done by the force-directed graph drawing
    algorithm due to Fruchterman & Reingold (1991) and the edges are drawn as
    straight line segments.

    While in the plot window you can press space bar to move the vertices a
    little, or press enter key to close the window and return to the command
    line.'''

    def do_print(self, args):
        self.print(args)

    do_print.__doc__ = '''\
NAME
    print - Print a text.

SYNOPSIS
    print <string>'''

    def do_quit(self, args):
        if args: return SyntaxError()
        return True

    do_quit.__doc__ = _('''\
NAME
    quit - Ends the current session and exit.

SYNOPSIS
    quit''')

    def do_restart(self, args=''):
        args = args.split()
        if len(args) == 1 and args[0] == 'embedding':
            self.embedding = Embedding2D(self.bigraph)
        elif len(args) == 0:
            self.history = []
            self.bigraph = Bigraph()
            self.embedding = Embedding2D(self.bigraph)
            self.plot2D = TkBigraphPlot(self.embedding)
        else:
            return SyntaxError()

    do_restart.__doc__ = _('''\
NAME
    restart - End current session and restart DynkinLab.

SYNOPSIS
    restart [embedding]

DESCRIPTION
    Erases all the data from the current session. This includes the session
    history, the bigraph, and its embedding. Additionally, you can erase the
    embedding alone with “restart embedding”.''')

    def do_save(self, filename: str):
        self.history.pop()
        filename = filename.strip()
        if not filename.endswith('.dyn'):
            filename += '.dyn'
        with open(filename, mode='w', encoding='UTF-8') as file:
            file.write('\n'.join(self.history))

    do_save.__doc__ = _('''\
NAME
    save - Save the commands of current session into a file.

SYNOPSIS
    save <filename>

EXAMPLE
    save foobar.dyn''')

    def help_help(self):
        self.print(_('List available commands with “help” or detailed help \
with “help cmd”.'))

    def postcmd(self, stop, line):
        if isinstance(stop, SyntaxError):
            self.default(line)
            return False
        return stop

    def precmd(self, line: str):
        line = line.strip()
        if line: self.history.append(line)
        return line

    def print(self, *args, **kwargs):
        print(*args, file=self.stdout, **kwargs)

    do_T = do_flation


if __name__ == '__main__':
    app = DynkinLab()
    if os.getenv('TERM') is not None and 'color' in os.getenv('TERM'):
        app.prompt = '\33[92m\33[1m{}\33[0m'.format(app.prompt)
    if len(sys.argv) == 2:
        app.cmdqueue.append('load ' + sys.argv[1])
    elif len(sys.argv) != 1:
        raise SyntaxError()
    app.cmdloop()