��          �      l      �     �                '  �   B  m     (   �  M   �     �  8    G  Q  �  �  �    S   �  E   '  7  m  |   �     "  F   9  %   �  U  �  %   �     "     <      R  �   s  }   o  2   �  O         p  �  �  `  +  �  �  �  F  _   �  I   =  F  �  �   �      _!  N   |!  *   �!             	       
                                                                              *** File not found: {filename} *** No help on %s *** Skipping T {} {} *** Unknown syntax: {line} Apply flations over dotted edges until no dotted edges remain.
In each step a dotted edge is selected at random.
This method is guaranteed to finish in a finite number of steps
for positive definite bigraphs only. Apply flations over solid edges until no solid edges remain.
In each step a solid edge is selected at random. Documented commands (type help <topic>): List available commands with “help” or detailed help with “help cmd”. Miscellaneous help topics: NAME
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
    add 1 .> "2x" -- x NAME
    apply - Apply some algorithm to the current bigraph

SYNOPSIS
    apply <algorithm> [verbose]

DESCRIPTION
    Runs the specified algorithm on the current bigraph. If “verbose” option is
    present then the algorithm is instructed to print any messages to the
    output.

    Current supported algorithms are:
{} NAME
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
    describe as polynomial in latex NAME
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
        Abbreviation of T x z; T x w; T y z; T y w NAME
    load - Load and execute commands from a file.

EXAMPLE
    load foobar.dyn NAME
    quit - Ends the current session and exit.

SYNOPSIS
    quit NAME
    restart - End current session and restart DynkinLab.

SYNOPSIS
    restart [embedding]

DESCRIPTION
    Erases all the data from the current session. This includes the session
    history, the bigraph, and its embedding. Additionally, you can erase the
    embedding alone with “restart embedding”. NAME
    save - Save the commands of current session into a file.

SYNOPSIS
    save <filename>

EXAMPLE
    save foobar.dyn Undocumented commands: Welcome to DynkinLab 4.0.
Type “help” or “?” to list commands. {value} unexpected on line {line_num} Project-Id-Version: 
POT-Creation-Date: 2016-08-24 13:22-0500
PO-Revision-Date: 2016-08-24 13:23-0500
Language-Team: 
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Generated-By: pygettext.py 1.5
X-Generator: Poedit 1.8.7.1
Last-Translator: 
Plural-Forms: nplurals=2; plural=(n != 1);
Language: es
 *** Archivo no encontrado: {filename} *** No hay ayuda sobre %s *** Omitiendo T {} {} *** Sintaxis desconocida: {line} Aplica flaciones sobre las aristas punteadas hasta que no quede ninguna.
En cada paso se selecciona una arista punteada al azar. Sólo se tiene
garantía de que este método termina en un número finito de pasos cuando
el bigrafo es definido positivo. Aplica flaciones sobre las aristas sólidas hasta que no quede ninguna.
En cada paso se selcciona una arista sólida al azar. Órdenes documentadas (escriba “help <tema>”): Lista órdenes disponibles con “help” o ayuda detallada con “help cmd”. Temas de ayuda misceláneos: NOMBRE
    add - Agrega un camino al bigrafo actual.

SINOPSIS
    add <vertex>[<edge><vertex>]*

DESCRIPCIÓN
    Esta orden le permite agregar vértices y aristas al bigrafo actual de forma
    compacta, un camino a la vez.

    <vertex> puede ser cualquiera de lo siguiente:
        - Una cadena de caracteres alfabéticos, guiones bajos ('_') o dígitos
          (0-9) siempre que no inicie en un dígito;
        - Un número entero o decimal;
        - Cualquier cadena entre comillas ("...").

    <edge> Puede ser:
        '--'            una arista sólida no dirigida
        '->' o '<-'     una arista sólida dirigida
        '..'            una arista punteada no dirigida
        '.>' o '<.'     una arista punteada dirigida

    Esta orden reemplaza cualquier arista que haya sido previamente añadida 
    entre la misma pareja dada de vértices.

EJEMPLO
    add x <- u .. v -> x -- w
    add 1 .> "2x" -- x NOMBRE
    apply - Aplica algún algoritmo al bigrafo actual

SINOPSIS
    apply <algorithm> [verbose]

DESCRIPCIÓN
    Ejecuta el algoritmo especificado sobre el bigrafo actual. Si la opción
    “verbose” está presente, entonces el algoritmo es instruido a imprimir
    mensajes en la salida.

    Actualmente los algoritmos disponibles son:
{} NOMBRE
    describe - Imprime una descripción del bigrafo

SINOPSIS
    describe
    describe as matrix [in <matrix language>]
    describe as graph [in <graph language>]
    describe as polynomial [in <polynomial language>]

DESCRIPCIÓN
    Describe el bigrafo actual como texto plano en algún lenguaje especificado
    Si no se proporciona ningún argumento, se describe la matriz en forma
    tabular. Actualmente los lenguajes disponibles son:

    Lenguajes para “matrix”: tabular, python, maxima, latex, csv
    Lenguajes para “graph”: dot, tikz
    Lenguajes para “polynomial”: python, maxima, latex

EJEMPLOS
    describe as graph in tikz
    describe as polynomial in latex NOMBRE
    flation - Aplica el morfismo de flación de bigrafos.
    T - Lo mismo que flation.

SINOPSIS
    flation <vertex> <vertex>
    flation (<vertex>*)*

DESCRIPCIÓN
    Dados dos vértices como argumentos, el morfismo de flación es equivalente a
    la operación de suma de renglones en la matriz de adyacencia, seguida de
    un suma de columnas.
    Es posible agrupar vértices con paréntesis para abreviar flaciones.
    Este morfismo es cerrado en bigrafos al omitir aquellas transformaciones que
    no producen bigrafos.

EJEMPLOS
    T 2 3
        Equivalente a flation 2 3
    T (x y) (z w)
        Abreviación de T x z; T x w; T y z; T y w NOMBRE
    load - Leer y ejecutar ordenes desde un archivo.

EJEMPLO
    load caraculiambro.dyn NOMBRE
    quit - Finalizar la sesión actual y salir.

SINOPSIS
    quit NOMBRE
    restart - Finaliza la sesión actual y reinicia DynkinLab.

SINOPSIS
    restart [embedding]

DESCRIPCIÓN
    Borra todos los datos de la sesión actual. Esto incluye el historial de
    órdenes, el bigrafo y su encaje en el plano. Más aún, se puede borrar
    únicamente el encaje con “restart embedding”. NOMBRE
    save - Guarda las órdenes de la sesión actual en un archivo.

SINOPSIS
    save <nombredearchivo>

EJEMPLO
    save miscomandos.dyn Órdenes sin documentación: Bienvenido a DynkinLab 4.0.
Escriba “help” o “?” para listar órdenes. {value} inesperado en la línea {line_num} 