import re
from cmd import Cmd
from bigraph import bigraph
from collections import namedtuple
token = namedtuple('token', ['typ', 'val'])

class arg_parser:
    tokens = {'int':    r'\d+',
              'edge':   r'(\-\-)|(\->)|(<\-)|(\.\.)|(\.>)|(<\.)',
              'id':     r'[A-F]',
              'skip':   r'\s+'}

    def __init__(self):
        tok_rex =  '|'.join('(?P<{}>{})'.format(t, r) \
                               for (t, r) in self.tokens.items())
        self.token_match = re.compile(tok_rex).match
        self.whitespace = re.compile('\s+')
    
    def tokenize(self, arg):
        mo = self.token_match(arg)
        pos = 0
        while mo is not None:
            typ = mo.lastgroup
            if typ != 'skip':
                yield token(typ, mo.group(typ))
            pos = mo.end()
            mo = self.token_match(arg, pos)
        if pos != len(arg):
            raise RuntimeError('Unexpected character')

    def intseq(self, arg):
        for tok in self.tokenize(arg):
            if tok.typ != 'int': raise RuntimeError('Unexpected token')
            yield int(tok.val)

    def path(self, arg):
        vert = True
        for tok in self.tokenize(arg):
            if vert and tok.typ == 'int':
                yield int(tok.val)
                
            elif not vert and tok.typ == 'edge':
                yield tok.val
            else: raise RuntimeError('Unexpected token')
            vert = not vert
        if vert: raise RuntimeError('Invalid sintax')

    def splitargs(self, args):
        return self.whitespace.split(args.strip())

    def cartype(self, args):
        toks = list(self.tokenize(args))
        if len(toks) != 2 or toks[0].typ != 'id' or toks[1].typ != 'int':
            raise RuntimeError('Unexpected token')
        return (toks[0].val, int(toks[1].val))

class interactive(Cmd):
    intro = 'interactive.py 3.1 Escriba «?» para ver los comandos disponibles.'
    prompt = '> '
    doc_header = 'Comandos documentados (escriba help <comando>):'
    misc_header = 'Topicos de ayuda misceláneos:'
    undoc_header = 'Comandos sin documentación:'
    nohelp = '* Sin ayuda para %s'
    nosintax = '* Sintaxis desconocida: {}'
    multiedgewarn = 'Aviso: el bigrafo contiene aristas múltiples'
    #do_help.__doc__ = 'Muestra los comandos disponibles con "help" o ayuda detallada con "help cmd"'

    def is_multi_edged(self):
        A = self.B.adj
        n = len(A)
        for u in range(1,n):
            for v in range(u):
                if abs(A[u][v]) > 1 and abs(A[v][u]) > 1:
                    return True
        return False

    def __init__(self):
        super().__init__()
        self.B = bigraph()
        self.pos = None
        self.parse = arg_parser()

    def do_neighbors(self, args):
        '''Sintaxis: neighbors v
Muestra los vecinos del vértice v en el bigrafo.'''
        try:
            u = int(args.strip())
        except RuntimeError: return self.default('neighbors', args)
        if u in self.B.vertices():
            print(' '.join(map(str, self.B.neighbors(u))))
        else: print('{} no pertenece al bigrafo'.format(u))

    def do_new(self, args):
        '''Sintaxis: new [id]
Carga en memoria un bigrafo vacío. Opcionalmente, id puede ser A, B, C, D, E o F
seguido de un entero positivo para cargar en memoria un bigrafo del tipo Dynkin
especificado. Ejemplos:
    new → crea un bigrafo nuevo
    new E6 → crea un bigrafo E6'''
        if args:
            try:
                args = self.parse.cartype(args)
            except: return self.default('new', args)
            cons = {'A':bigraph.finite_A,
                    'B':bigraph.finite_B,
                    'C':bigraph.finite_C,
                    'D':bigraph.finite_D,
                    'E':bigraph.finite_E,
                    'F':bigraph.finite_F}
            self.B = cons[args[0]](args[1])
        else: self.B = bigraph()
        self.pos = None
        
    def do_quit(self, args):
        '''Sintaxis: quit
Salir del programa.'''
        if args == '':
            return True
        else: self.default('quit', args)

    def do_tex(self, args):
        '''Sintaxis: tex
Imprime un código LaTeX que representa al bigrafo en código TikZ'''
        print(self.B.tex(pos = self.pos))

    def do_load(self, args):
        '''Sintaxis: load archivo
Carga en memoria un bigrafo previamente guardado. Ejemplo: load ejemplo'''
        try:
            self.B = bigraph.load(args)
            self.pos = None
        except: print('Error al abrir {}'.format(repr(args)))

    def do_save(self, args):
        '''Sintaxis: save archivo
Guarda en memoria el bigrafo actual.
Ejemplo: save ejemplo'''
        try:
            self.B.save(args)
        except: print('Error al guardar {}'.format(repr(args)))
    
    def do_plot(self, args):
        '''Sintaxis: plot [frame]
Muestra el bigrafo actual en pantalla. Opcionalmente el parámetro “frame”
muestra el marco del bigrafo actual.'''
        if self.B.vertices():                
            args = self.parse.splitargs(args)
            self.pos = self.B.embedding(start_pos = self.pos)
            if args == ['frame']:
                self.B.frame().plot(pos = self.pos)
            else:
                self.B.plot(pos = self.pos)
        else: print('Nada que graficar')

    def do_adj(self, args):
        '''Sintaxis: adj
Muestra la matriz de adyacencia del bigrafo actual.'''
        if args:
            self.default('adj', args)
        else: print(self.B)

    def do_T(self, args):
        '''Sintaxis: T v₁ v₂ v₃ … vₖ
Realiza sucesivamente la operacion de arrastre de v₁ a v₂, de v₂ a v₃, etc.'''
        try:
            v = list(self.parse.intseq(args))
        except RuntimeError: v = []
        if len(v) < 2:
            self.default('T', args)
        else:
            B = self.B
            if any(u not in self.B.vertices() for u in v):
                print('Todos los vertices deben pertenecer al bigrafo.')
                return False
            for k in range(len(v) - 1):
                B.flation(v[k], v[k + 1])
        if self.is_multi_edged(): print(self.multiedgewarn)

    def do_I(self, args):
        '''Sintaxis: I v₁ v₂ v₃ … vₖ
Realiza sucesivamente la operacion de inversion de signo sobre los vertices v₁
v₂, v₃, hasta vₖ.'''
        try:
            v = list(self.parse.intseq(args))
        except RuntimeError: v = []
        if len(v) < 1:
            self.default('I', args)
        else:
            if any(u not in self.B.vertices() for u in v):
                print('Todos los vertices deben pertenecer al bigrafo.')
                return False
            B = self.B
            for u in v:
                B.inversion(u)

    def do_P(self, args):
        '''Sintaxis: P v₁ v₂
Intercambia los vértices v₁ y v₂.'''
        try:
            u, v = self.parse.intseq(args)
        except: return self.default('P', args)
        if not {u, v} <= self.B.vertices():
            print('Los dos vertices deben pertenecer al bigrafo.')
            return False
        else: self.B.permutation(u, v)
        
    def do_S(self, args):
        '''Sintaxis: S v₁ v₂ v₃ … vₖ
Encoge el subgrafo inducido por los vértices
v₁ v₂ v₃ … vₖ en un único vértice con etiqueta vₖ.'''
        try:
            v = list(self.parse.intseq(args))
        except RuntimeError: v = []
        if len(v) < 2:
            self.default('S', args)
        else:
            if any(u not in self.B.vertices() for u in v):
                print('Todos los vertices deben pertenecer al bigrafo.')
            else:
                self.B.shrink(v, v[-1])
                if self.pos is not None:
                    for k in range(len(v) - 1):
                        del self.pos[v[k]]
        if self.is_multi_edged(): print(self.multiedgewarn)

    def do_add(self, args):
        '''Sintaxis: add v₀ e₁ v₁ e₂ v₂ e₃ v₃ … eₖ vₖ
Añade el camino especificado al bigrafo. Los vertices deben ser números
enteros y las aristas pueden ser de cuatro tipos:
    -- : arista sólida
    .. : arista punteada
    -> : arco sólido (también admite <-)
    .> : arco punteado (también admite <.)
Ejemplo: add 1--2..3.>4<-5'''
        if args: 
            try:
                path = list(self.parse.path(args))
            except RuntimeError: return self.default('add', args)
            self.B.add(path)
            self.pos = None
            if self.is_multi_edged(): print(self.multiedgewarn)
        else:
            print('Nada que añadir.')

    def do_del(self, args):
        '''Sintaxis: del v₁ v₂ v₃ … vₖ
Elimina los vértices v₁, v₂, v₃, …, vₖ del bigrafo.'''
        try:
            u = int(args.strip())
        except: return self.default('del', args)
        if u in self.B.vertices():
            self.B.delete(u)
            del self.pos[u]
        else: print('{} no pertenece al bigrafo.'.format(u))

    def default(self, *args):
        print(self.nosintax.format(' '.join(args)))

    def do_reduce(self, args):
        '''Sintaxis: reduce [plot] [adj] [tex]
Aplica el método de las inflaciones al bigrafo actual. Adicionalmente los
parámetros opcionales plot, adj y tex aplican el comando homónimo en cada
bigrafo intermedio obtenido por el algoritmo.'''
        flags = frozenset(self.parse.splitargs(args))
        if not flags <= frozenset(['plot', 'tex', 'adj', '']):
            return self.default('reduce', args)
        bplot = 'plot' in flags
        btex = 'tex' in flags
        badj = 'adj' in flags
        T = iter(self.B.reduce())
        while True:
            if badj: self.do_adj('')
            if bplot: self.do_plot('')
            if btex: self.do_tex('')
            edge = next(T, None)
            if edge == None: break
            print('> T {} {}'.format(*edge))
        if self.is_multi_edged(): print(self.multiedgewarn)

    def do_treeify(self, args):
        '''Sintaxis: treeify [plot] [adj] [tex]
Aplica un método que trata de llevar el bigrafo actual a su forma arbórea.
Adicionalmente los parámetros opcionales plot, adj y tex aplican el comando
homónimo en cada bigrafo intermedio obtenido por el algoritmo.'''
        flags = frozenset(self.parse.splitargs(args))
        if not flags <= frozenset(['plot', 'tex', 'adj', '']):
            return self.default('treeify', args)
        bplot = 'plot' in flags
        btex = 'tex' in flags
        badj = 'adj' in flags
        T = iter(self.B.treeify())
        while True:
            if badj: self.do_adj('')
            if bplot: self.do_plot('')
            if btex: self.do_tex('')
            path = next(T, None)
            if path == None: break
            print('> T ' + ' '.join(map(str, path)))
        if self.is_multi_edged(): print(self.multiedgewarn)

interactive.do_help.__doc__ = 'Muestra los comandos disponibles con "help" o ayuda detallada con "help cmd"'

if __name__ == '__main__':
    interactive().cmdloop()
