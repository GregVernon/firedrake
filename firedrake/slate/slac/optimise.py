from gem.node import MemoizerArg
from functools import singledispatch
from contextlib import contextmanager
import firedrake.slate.slate as sl


def optimise(expression):
    return push_block(expression)


def push_block(expression):
    mapper = MemoizerArg(_push_block)
    ret = mapper(expression, ())
    return ret


@singledispatch
def _push_block(expr, self, indices):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_block.register(sl.Transpose)
def _push_block_transpose(expr, self, indices):
    return sl.Transpose(map(self, expr.children, indices[::-1])) if indices else expr


@_push_block.register(sl.Add)
def _push_block_add(expr, self, indices):
    return sl.Add(map(self, expr.children, indices)) if indices else expr


@_push_block.register(sl.Negative)
def _push_block_neg(expr, self, indices):
    return sl.Negative(map(self, expr.children, indices)) if indices else expr


@_push_block.register(sl.Factorization)
@_push_block.register(sl.Inverse)
@_push_block.register(sl.Solve)
@_push_block.register(sl.Mul)
def _push_block_stop(expr, self, indices):
    return sl.Block(expr, indices) if self.block else expr


@_push_block.register(sl.AssembledVector)
@_push_block.register(sl.Tensor)
def _push_block_terminal(expr, self, indices):
    # FIXME
    assert not isinstance(expr, sl.AssembledVector), "If we do it like this we do need \
                                                 to be able to glue bit of functions back together"
    return type(expr)(self.block.form) if indices else expr


@_push_block.register(sl.Block)
def _push_block_block(expr, self, indices):
    if indices:
        self.block = sl.Block(*expr.children, tuple(big[slice(*small)] for small, big in zip(expr._indices, self.block._indices)))
    else:
        self.block = expr
    return self(*expr.children, indices)
