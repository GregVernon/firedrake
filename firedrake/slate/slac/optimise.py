from gem.node import Memoizer
from functools import singledispatch
from contextlib import contextmanager
import firedrake.slate.slate as sl


def optimise(expression):
    return push_block(expression)


def push_block(expression):
    mapper = Memoizer(_push_block)
    mapper.block = None
    ret = mapper(expression)
    return ret


@singledispatch
def _push_block(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_block.register(sl.Transpose)
def _push_block_transpose(expr, self):
    if self.block:
        with block(self, *expr.children, self.block._indices[::-1]):
            ret = sl.Transpose(self(*expr.children))
    else:
        ret = expr
    return ret


@_push_block.register(sl.Add)
def _push_block_add(expr, self):
    if self.block:
        ops = []
        for op in expr.children:
            with block(self, op, self.block._indices):
                ops += [self(op)]
        ret = sl.Add(*ops)
    else:
        ret = expr
    return ret


@_push_block.register(sl.Negative)
def _push_block_neg(expr, self):
    if self.block:
        with block(self, *expr.operands, self.block._indices):
            ret = sl.Negative(self(*expr.children))
    else:
        ret = expr
    return ret


@_push_block.register(sl.Factorization)
@_push_block.register(sl.Inverse)
@_push_block.register(sl.Solve)
@_push_block.register(sl.Mul)
def _push_block_stop(expr, self):
    return sl.Block(expr, self.block._indices) if self.block else expr


@_push_block.register(sl.AssembledVector)
@_push_block.register(sl.Tensor)
def _push_block_terminal(expr, self):
    # FIXME
    assert not isinstance(expr, sl.AssembledVector), "If we do it like this we do need \
                                                 to be able to glue bit of functions back together"
    return type(expr)(self.block.form) if self.block else expr


@_push_block.register(sl.Block)
def _push_block_block(expr, self):
    if self.block:
        self.block = sl.Block(*expr.children, tuple(big[slice(*small)] for small, big in zip(expr._indices,self.block._indices)))
    else:
        self.block = expr
    return self(*expr.children)


@contextmanager
def block(self, tensor, indices):
    """Provides a context to push blocks inside an expression.
    :arg tensor: tensor of which block is taken of.
    :arg indices: indices to a block of a tensor.
    :returns: the modified code generation context."""
    self.block = sl.Block(tensor, indices)
    yield self