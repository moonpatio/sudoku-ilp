#!/usr/bin/env python
__version__ = '0.1'
__license__ = 'BSD2'
__version_info__ = (0, 1)
__author__ = 'Matthew Morrow <moonpatio@gmail.com>'
"""
Sudoku solver via the exact set cover problem, which is solved with
0/1-integer programming.

> def sudoku(givens, n=3):
>     ...
>     D = build_digits()
>     R, C, B = build_rows_cols_blocks()
>     RD = intersections(R, D)
>     CD = intersections(C, D)
>     BD = intersections(B, D)
>     Cells = build_cells()
>     Knowns = build_knowns()
>     A = np.vstack((RD.T, CD.T, BD.T, Cells.T, Knowns))
>     b = ones((A.shape[0],1))
>     c = zeros((A.shape[1],1))
>     G = zeros((1,A.shape[1]))
>     h = zeros((1,1))
>     integer = set([])
>     binary = set(range(A.shape[1]))
>     _, x = milp(c, G, h, A=A, b=b, I=integer, B=binary)
>     if x:
>         x = np.squeeze(x).astype(dtype)
>         x = from_indicator(x, nvals)
>     return x

The key technique to highlight here is that of computing the set
of interesections of all pairs of sets in two set covers (which are
not required to cover the entire element set), where the result and
input set covers are represented as incidence matrices (where columns
correspond to the sets and rows correspond to the elements). Note that
I don't bother to remove columns of zeros (i.e. empty interesections)
from the result because in the case of sudoku this never occurs. The
function is repoduced here:

> def intersections(I, J):
>     '''I, J incidence matrices.'''
>     assert(I.shape[0] == J.shape[0])
>     return np.array([u * v for u, v in itertools.product(I.T, J.T)]).T

NOTE[glpk]:
    Some versions of glpk have a bug (which has been reported) where you
    can't prevent it from printing "Long-step dual simplex will be used"
    to stdout. So, if your glpk also has this bug and you want to make use
    of the stdout of this program, you need to do '... | ./sudoku.py |
    grep -vE "^Long"' or equivalent. This is another reminder of one of
    the questions of the ages: Why do scientist-type programmers so love
    printing RANDOM TRASH to the terminal AS THE DEFAULT BEHAVIOR????? We
    simply may never know. :)
"""

import sys
import itertools
import numpy as np
import cvxopt
import cvxopt.glpk

#############################################################################

TEST_PROBLEMS="""
013000002200000480000700019000900800700000020000300000002630900409070600001490008
500070600410053007067120008840000010000010306000700020000000065000000870030000000
000000000008004010004070000800300069007100004095000000400510703070800900500006080
060050030001020050700003400426038000003000000057401200000007000000810007000060001
000600409008000000003009000001705000800000001040300000030000056100000030094008007
046000580718000006500040000100006750080000010020005804000007400069000070200001000
600000001000630000082410009000005010040000000209008005004000080000020460500000000
002805007090027400000010009080000731300000000400071000000000004010900000000000253
400007910000400000200500036503000020000070100070129000000903000305000060000060000
000000004901060500008000090800403706000010000540007180000039007000040050000700810
"""

TEST_SOLUTIONS="""
913584762257169483648723519136942857795816324824357196572638941489271635361495278
529478631418653297367129458846532719275914386193786524982347165654291873731865942
712983645938654217654271398841327569267195834395468172486519723173842956529736481
264159738381724956795683412426538179913276584857491263138947625642815397579362841
512673489978142563463859712321795648859426371746381295237914856185267934694538127
946712583718359246532648197193486752485273619627195834351967428869524371274831965
697852341451639278382417659873265914145973826269148735714396582938521467526784193
142895367893627415756413829685249731371568942429371586238756194514932678967184253
456237918931486572287591436513648729629375184874129653762953841395814267148762395
756981234921364578438572691819453726673218945542697183285139467197846352364725819
"""

def test():
    for x in filter(lambda x:x, TEST_PROBLEMS.split("\n")):
        print(show_sudoku(sudoku(read_sudoku(x))[0]))

#############################################################################

def main():
    if sys.argv[-1] == "--test":
        test()
    else:
        for x in read_sudoku_lines(sys.stdin):
            print(show_sudoku(sudoku(x)[0]))

def read_sudoku(string):
    return np.array(map(int, string))
def show_sudoku(board):
    if type(board) != np.ndarray: return ""
    return "".join(map(str, board))
def read_sudoku_lines(lines):
    def go(line): return [read_sudoku(x) for x in line.rstrip()]
    return (go(x) for x in lines)

#############################################################################

def sudoku(givens, n=3):
    dtype = np.int64
    def ones(shape): return np.ones(shape, dtype=dtype)
    def zeros(shape): return np.zeros(shape, dtype=dtype)
    nvals = n * n
    ncells = nvals * nvals
    allvalsmask = ones(nvals)
    givens = np.array(givens, dtype=dtype).flatten()
    def build_digits():
        icells = np.arange(ncells)
        def go(i): return (
            i * ones(ncells),
            i + nvals * icells)
        digs = zeros((nvals, nvals * ncells))
        digs[zip(*map(go, range(nvals)))] = 1
        D = digs.T
        return D
    def build_rows_cols_blocks():
        I = np.arange(n)
        J = np.hstack([I] * n)
        K = np.arange(n * n)
        Z = np.zeros_like(K)
        dims = [nvals, nvals]
        rows = np.array([np.ravel_multi_index((i + Z, K), dims) for i in K])
        cols = np.array([np.ravel_multi_index((K, j + Z), dims) for j in K])
        blks = np.array([
            np.ravel_multi_index(zip(*itertools.product(i*n+I, j*n+I)), dims)
                for i, j in itertools.product(I,I)])
        def go(X):
            Y = np.array([indices_to_indicator(x, ncells) for x in X])
            return np.array([x * allvalsmask for x in Y.flatten()]).reshape(
                (Y.shape[0], nvals * Y.shape[1]))
        R = go(rows).T
        C = go(cols).T
        B = go(blks).T
        return R, C, B
    def build_cells():
        X = zeros((ncells, nvals * ncells))
        for i in range(ncells):
            j = i * nvals
            X[i,j:j+nvals] = 1
        return X.T
    def build_knowns():
        I = to_indicator(givens, nvals)
        dim = I.shape[0]
        return np.array([indices_to_indicator([x], dim)
            for x in np.nonzero(I)[0]])
    D = build_digits()
    R, C, B = build_rows_cols_blocks()
    RD = intersections(R, D)
    CD = intersections(C, D)
    BD = intersections(B, D)
    Cells = build_cells()
    Knowns = build_knowns()
    A = np.vstack((RD.T, CD.T, BD.T, Cells.T, Knowns))
    b = ones((A.shape[0],1))
    c = zeros((A.shape[1],1))
    G = zeros((1,A.shape[1]))
    h = zeros((1,1))
    integer = set([])
    binary = set(range(A.shape[1]))
    status, x = milp(c, G, h, A=A, b=b, I=integer, B=binary)
    DEBUG = (status, c, G, h, A, b, B)
    if x:
        x = np.squeeze(x).astype(dtype)
        x = from_indicator(x, nvals)
    return x, DEBUG

#############################################################################

def indicator_to_indices(x):
   return np.nonzero(x)[0]
def indices_to_indicator(I, n, dtype=np.int64):
    x = np.zeros(n, dtype=dtype)
    x[I] = 1
    return x
def from_indicator(a, n):
    return np.array([
        (list(1+indicator_to_indices(x))+[0])[0]
            for x in a.flatten().reshape((a.flatten().shape[0]/n,n))])
def to_indicator(a, n):
    return np.array([indices_to_indicator([x], n + 1)[1:]
        for x in a.flatten()]).flatten()
def intersections(I, J):
    """I, J incidence matrices."""
    assert(I.shape[0] == J.shape[0])
    return np.array([u * v for u, v in itertools.product(I.T, J.T)]).T

#############################################################################

#STATUS = (
#'optimal',
#'feasible',
#'undefined',
#'invalid formulation',
#'infeasible problem',
#'LP relaxation is primal infeasible',
#'LP relaxation is dual infeasible',
#'unknown',)
def milp(c, G, h, A=None, b=None, I=None, B=None):
    c, G, h = map(lambda x:cvxopt.matrix(x, x.shape, 'd'), (c, G, h))
    A = A if type(A) == type(None) else cvxopt.matrix(A, A.shape, 'd')
    b = b if type(b) == type(None) else cvxopt.matrix(b, b.shape, 'd')
    return cvxopt.glpk.ilp(
        c, G, h, A=A, b=b, I=I, B=B,
        options={'msg_lev':'GLP_MSG_OFF'})

#############################################################################

if __name__ == '__main__':
    main()

#############################################################################
