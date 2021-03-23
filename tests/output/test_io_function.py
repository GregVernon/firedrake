from firedrake import *
import functools
import pytest
from pyop2.mpi import COMM_WORLD
from pyop2 import RW
import os
from os.path import abspath, dirname, join
import time
from mpi4py import MPI

cwd = abspath(dirname(__file__))

fs_name = "V"
func_name = "f"


def _get_mesh(cell_type, mesh_name, comm):
    if cell_type == "triangle":
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name, comm=comm)
    elif cell_type == "tetrahedra":
        mesh = Mesh(join(cwd, "..", "meshes", "sphere.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "tetrahedra_large":
        mesh = Mesh(join(cwd, "..", "meshes", "sphere_large.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "quad":
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "triangle_small":
        # Sanity check
        mesh = UnitSquareMesh(2, 1);
        mesh_name = "default"
    elif cell_type == "quad_small":
        # Sanity check
        mesh = UnitSquareMesh(2, 2, quadrilateral=True);
        mesh_name = "default"
    mesh.name = mesh_name
    return mesh


def _get_expr(V):
   mesh = V.mesh()
   dim = mesh.topological_dimension()
   size = V.ufl_element().value_size()
   if dim == 2:
        x, y = SpatialCoordinate(mesh)
        if size == 1:
            return x * y * y
        elif size == 2:
            return as_vector([x * y * y, x * x * y])
        elif size == 3:
            return as_vector([x * y, y, y * y])
        elif size == 4:
            return as_vector([x * y, y, 2 * x, y * y])
        elif size == 5:
            return as_vector([2 + x, 3 + y, 5 + x * y, 7 + x * x, 11 + y * y])
        elif size == 6:
            return as_vector([2 + x, 3 + y, 5 + x * y, x * x, y * y, 7 + x + y ])
        elif size == 7:
            return as_vector([2 + x, 3 + y, 5 + x * y, x * x, y * y, 7 + x * y, x + y])
   elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
        if size == 1:
            return x * y * y * z * z * z
        elif size == 2:
            return as_vector([x * y * y * z * z * z, x * y * z * z])
        elif size == 3:
            return as_vector([x, y, y * z * z])
        elif size == 4:
            return as_vector([x, y, y * z * z, x * x])
        elif size == 5:
            return as_vector([2 + x, 3 + y, 5 + z, 7 + x * y, 11 + y * z])
        elif size == 6:
            return as_vector([x, y, z, x * y, y * z, z * x])
        elif size == 7:
            return as_vector([x, y, z, x * y, y * z, z * x, x * y * z])
   raise ValueError(f"Invalid dim-size pair: dim = {dim}, size = {size}")


def _load_check_save_functions(filename, func_name, comm, method, mesh_name):
    # Load
    start = time.time()
    with CheckpointFile(filename, "r", comm=comm) as afile:
        meshB = afile.load_mesh(mesh_name)
    end = time.time()
    avgtime = comm.allreduce(end - start, MPI.SUM) / comm.size
    if comm.rank == 0:
        print("nproc = ", comm.size, ", mesh load time = ", avgtime, flush=True)
    start = time.time()
    meshB.init()
    end = time.time()
    avgtime = comm.allreduce(end - start, MPI.SUM) / comm.size
    if comm.rank == 0:
        print("nproc = ", comm.size, ", mesh init time = ", avgtime, flush=True)
    start = time.time()
    with CheckpointFile(filename, "r", comm=comm) as afile:
        fB = afile.load_function(func_name, mesh_name=mesh_name)
    end = time.time()
    avgtime = comm.allreduce(end - start, MPI.SUM) / comm.size
    if comm.rank == 0:
        print("nproc = ", comm.size, ", func load time = ", avgtime, flush=True)
    # Check
    VB = fB.function_space()
    expr = _get_expr(VB)
    fBe = getattr(Function(VB), method)(expr)
    assert assemble(inner(fB - fBe, fB - fBe) * dx) < 5.e-15
    # Save
    start = time.time()
    with CheckpointFile(filename, 'w', comm=comm) as afile:
        afile.save_function(fB)
    end = time.time()
    avgtime = comm.allreduce(end - start, MPI.SUM) / comm.size
    if comm.rank == 0:
        print("nproc = ", comm.size, ", func save time = ", avgtime, flush=True)


@pytest.mark.parallel(nprocs=7)
@pytest.mark.parametrize('cell_family_degree', [("triangle_small", "P", 1),
                                                ("triangle_small", "P", 6),
                                                ("triangle_small", "DP", 0),
                                                ("triangle_small", "DP", 1),
                                                ("triangle_small", "DP", 7),
                                                ("quad_small", "Q", 1),
                                                ("quad_small", "Q", 6),
                                                ("quad_small", "DQ", 0),
                                                ("quad_small", "DQ", 1),
                                                ("quad_small", "DQ", 7),
                                                ("triangle", "P", 1),
                                                ("triangle", "P", 2),
                                                ("triangle", "P", 3),#
                                                ("triangle", "P", 4),#
                                                ("triangle", "P", 5),
                                                ("triangle", "DP", 0),
                                                ("triangle", "DP", 1),
                                                ("triangle", "DP", 2),
                                                ("triangle", "DP", 3),#
                                                ("triangle", "DP", 4),#
                                                ("triangle", "DP", 5),#
                                                ("triangle", "DP", 6),
                                                ("tetrahedra", "P", 1),
                                                ("tetrahedra", "P", 2),
                                                ("tetrahedra", "P", 3),
                                                ("tetrahedra", "P", 4),#
                                                ("tetrahedra", "P", 5),#
                                                ("tetrahedra", "P", 6),
                                                ("tetrahedra", "DP", 0),
                                                ("tetrahedra", "DP", 1),
                                                ("tetrahedra", "DP", 2),
                                                ("tetrahedra", "DP", 3),
                                                ("tetrahedra", "DP", 4),#
                                                ("tetrahedra", "DP", 5),#
                                                ("tetrahedra", "DP", 6),#
                                                ("tetrahedra", "DP", 7),
                                                ("quad", "Q", 1),
                                                ("quad", "Q", 2),
                                                ("quad", "Q", 3),#
                                                ("quad", "Q", 4),#
                                                ("quad", "Q", 5),
                                                ("quad", "DQ", 0),
                                                ("quad", "DQ", 1),
                                                ("quad", "DQ", 2),
                                                ("quad", "DQ", 3),#
                                                ("quad", "DQ", 4),#
                                                ("quad", "DQ", 5),#
                                                ("quad", "DQ", 6),])
def test_io_function_cg_dg(cell_family_degree, tmpdir):
    # Parameters
    cell_type, family, degree = cell_family_degree
    method = "interpolate"
    filename = os.path.join(str(tmpdir), "test_io_function_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh_name = "exampleDMPlex"
    meshA = _get_mesh(cell_type, mesh_name, COMM_WORLD)
    mesh_name = meshA.name
    VA = FunctionSpace(meshA, family, degree, name=fs_name)
    expr = _get_expr(VA)
    fA = Function(VA, name=func_name)
    getattr(fA, method)(expr)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)


@pytest.mark.parallel(nprocs=7)
@pytest.mark.parametrize('cell_family_degree_tuples', [("tetrahedra", (("P", 2), ("DP", 1))),
                                                       ("quad", (("Q", 4), ("DQ", 3)))])
def test_io_function_mixed(cell_family_degree_tuples, tmpdir):
    cell_type, family_degree_tuples = cell_family_degree_tuples
    method = "project"
    filename = os.path.join(str(tmpdir), "test_io_function_mixed_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh_name = "exampleDMPlex"
    meshA = _get_mesh(cell_type, mesh_name, COMM_WORLD)
    mesh_name = meshA.name
    VA_list = []
    for i, (family, degree) in enumerate(family_degree_tuples):
        VA_list.append(FunctionSpace(meshA, family, degree, name=fs_name + f"[{str(i)}]"))
    VA = functools.reduce(lambda a, b: a * b, VA_list)
    VA.name = fs_name
    expr = _get_expr(VA)
    fA = getattr(Function(VA, name=func_name), method)(expr)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)


@pytest.mark.parallel(nprocs=7)
@pytest.mark.parametrize('cell_family_degree_dim', [("tetrahedra", "P", 2, 3),
                                                    ("quad", "Q", 4, 5)])
def test_io_function_vector(cell_family_degree_dim, tmpdir):
    cell_type, family, degree, vector_dim = cell_family_degree_dim
    method = "interpolate"
    filename = os.path.join(str(tmpdir), "test_io_function_vector_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh_name = "exampleDMPlex"
    meshA = _get_mesh(cell_type, mesh_name, COMM_WORLD)
    mesh_name = meshA.name
    VA = VectorFunctionSpace(meshA, family, degree, dim=vector_dim, name=fs_name)
    expr = _get_expr(VA)
    fA = getattr(Function(VA, name=func_name), method)(expr)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)


@pytest.mark.parallel(nprocs=7)
@pytest.mark.parametrize('cell_type', ["tetrahedra",
                                       "quad"])
def test_io_function_mixed_vector(cell_type, tmpdir):
    method = "project"
    filename = os.path.join(str(tmpdir), "test_io_function_mixed_vector_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh_name = "exampleDMPlex"
    meshA = _get_mesh(cell_type, mesh_name, COMM_WORLD)
    mesh_name = meshA.name
    if cell_type == "tetrahedra":
        VA0 = VectorFunctionSpace(meshA, "P", 1, dim=2, name=fs_name + "[0]")
        VA1 = FunctionSpace(meshA, "DP", 0, name=fs_name + "[1]")
        VA2 = VectorFunctionSpace(meshA, "DP", 2, dim=4, name=fs_name + "[2]")
        VA = VA0 * VA1 * VA2
    elif cell_type == "quad":
        VA0 = VectorFunctionSpace(meshA, "DQ", 1, dim=2, name=fs_name + "[0]")
        VA1 = FunctionSpace(meshA, "DQ", 0, name=fs_name + "[1]")
        VA2 = VectorFunctionSpace(meshA, "Q", 2, dim=4, name=fs_name + "[2]")
        VA = VA0 * VA1 * VA2
    else:
        raise ValueError("Only testing tetrahedra and quad")
    VA.name = fs_name
    expr = _get_expr(VA)
    fA = getattr(Function(VA, name=func_name), method)(expr)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)


if __name__ == "__main__":
    #test_io_function_cg_dg(("triangle_small", "P", 1), "./")
    #test_io_function_cg_dg(("tetrahedra", "DG", 4), "./")
    #test_io_function_mixed(("tetrahedra", (("P", 2), ("DP", 1))), "./")
    #test_io_function_vector(("tetrahedra", "P", 2, 3), "./")
    test_io_function_mixed_vector("tetrahedra", "./")
