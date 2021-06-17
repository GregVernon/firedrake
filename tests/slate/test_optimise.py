import pytest
from firedrake import *
from firedrake.slate.slac import optimise as optimise_slate
import numpy as np


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    return UnitSquareMesh(2, 2, quadrilateral=request.param)


def test_optimise_tensor_blocks(mesh):
    if mesh.ufl_cell() == quadrilateral:
        U = FunctionSpace(mesh, "RTCF", 1)
    else:
        U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)
    n = FacetNormal(mesh)
    W = U * V * T
    u, p, lambdar = TrialFunctions(W)
    w, q, gammar = TestFunctions(W)

    _A = Tensor(inner(u, w)*dx + p*q*dx - div(w)*p*dx + q*div(u)*dx
               + lambdar('+')*jump(w, n=n)*dS + gammar('+')*jump(u, n=n)*dS
               + lambdar*gammar*ds)

    # Test individual blocks
    expressions = [(_A+_A).blocks[0,0], (_A*_A).blocks[0,0], (-_A).blocks[0,0], (_A.T).blocks[0,0]]
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"optimise_slate": False}).M.values
        opt = assemble(expr, form_compiler_parameters={"optimise_slate": True}).M.values
        assert np.allclose(opt, ref, rtol=1e-14)

    # Test mixed blocks
    expressions = [(_A+_A).blocks[:2, :2], (_A*_A).blocks[:2, :2], (-_A).blocks[:2, :2], (_A.T).blocks[:2, :2],
                   (_A+_A).blocks[1:3, 1:3], (_A*_A).blocks[1:3, 1:3], (-_A).blocks[1:3, 1:3], (_A.T).blocks[1:3, 1:3]]
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"optimise_slate": False}).M.values
        opt = assemble(expr, form_compiler_parameters={"optimise_slate": True}).M.values
        assert np.allclose(opt, ref, rtol=1e-14)

    # Test blocks on blocks
    expressions = [(_A+_A).blocks[:2, :2].blocks[0,0], (_A*_A).blocks[:2, :2].blocks[0,0],
                   (-_A).blocks[:2, :2].blocks[0,0], (_A.T).blocks[:2, :2].blocks[0,0]]
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"optimise_slate": False}).M.values
        opt = assemble(expr, form_compiler_parameters={"optimise_slate": True}).M.values
        assert np.allclose(opt, ref, rtol=1e-14)

# TODO add similar tests for vector blocks