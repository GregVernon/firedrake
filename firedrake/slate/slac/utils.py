from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Action)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import loopy as lp
from loopy.transform.callable import merge
import itertools


def visualise(dag, how = None):
    """
        Visualises a slate dag. Can for example used to show the original expression
        vs the optimised slate expression.

        :arg: a dag with nodes that have shape information
    """
    import tsensor
    from collections import OrderedDict

    # Add tensors as locals to this frame.
    # It's how tsensor acesses shape information and so forth
    from firedrake.slate.slac.utils import traverse_dags
    tensors = OrderedDict()
    for node in traverse_dags([dag]):
        tensors[str(node)] = node
    locals().update(tensors)
    

    code = str(dag)
    # plot expr
    if how == "tree":
        g = tsensor.astviz(code)
        g.view()
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        tsensor.pyviz(code, ax=ax)
        plt.show()


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the
    integrals of forms.
    """
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class SymbolWithFuncallIndexing(ast.Symbol):
    """A functionally equivalent representation of a `coffee.Symbol`,
    with modified output for rank calls. This is syntactically necessary
    when referring to symbols of Eigen::MatrixBase objects.
    """

    def _genpoints(self):
        """Parenthesize indices during loop assignment"""
        pt = lambda p: "%s" % p
        pt_ofs = lambda p, o: "%s*%s+%s" % (p, o[0], o[1])
        pt_ofs_stride = lambda p, o: "%s+%s" % (p, o)
        result = []

        if not self.offset:
            for p in self.rank:
                result.append(pt(p))
        else:
            for p, ofs in zip(self.rank, self.offset):
                if ofs == (1, 0):
                    result.append(pt(p))
                elif ofs[0] == 1:
                    result.append(pt_ofs_stride(p, ofs[1]))
                else:
                    result.append(pt_ofs(p, ofs))
        result = ', '.join(i for i in result)

        return "(%s)" % result


class Transformer(Visitor):
    """Replaces all out-put tensor references with a specified
    name of :type: `Eigen::Matrix` with appropriate shape. This
    class is primarily for COFFEE acrobatics, jumping through
    nodes and redefining where appropriate.

    The default name of :data:`"A"` is assigned, otherwise a
    specified name may be passed as the :data:`name` keyword
    argument when calling the visitor.
    """

    def visit_object(self, o, *args, **kwargs):
        """Visits an object and returns it.

        e.g. string ---> string
        """
        return o

    def visit_list(self, o, *args, **kwargs):
        """Visits an input of COFFEE objects and returns
        the complete list of said objects.
        """
        newlist = [self.visit(e, *args, **kwargs) for e in o]
        if all(newo is e for newo, e in zip(newlist, o)):
            return o

        return newlist

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        """Visits a COFFEE FunDecl object and reconstructs
        the FunDecl body and header to generate
        ``Eigen::MatrixBase`` C++ template functions.

        Creates a template function for each subkernel form.

        .. code-block:: c++

            template <typename Derived>
            static inline void foo(Eigen::MatrixBase<Derived> const & A, ...)
            {
              [Body...]
            }
        """
        name = kwargs.get("name", "A")
        new = self.visit_Node(o, *args, **kwargs)
        ops, okwargs = new.operands()
        if all(new is old for new, old in zip(ops, o.operands()[0])):
            return o

        ret, kernel_name, kernel_args, body, pred, headers, template = ops

        body_statements, _ = body.operands()
        decl_init = "const_cast<Eigen::MatrixBase<Derived> &>(%s_);\n" % name
        new_dec = ast.Decl(typ="Eigen::MatrixBase<Derived> &", sym=name,
                           init=decl_init)
        new_body = [new_dec] + body_statements
        eigen_template = "template <typename Derived>"

        new_ops = (ret, kernel_name, kernel_args,
                   new_body, pred, headers, eigen_template)

        return o.reconstruct(*new_ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        """Visits a declared tensor and changes its type to
        :template: result `Eigen::MatrixBase<Derived>`.

        i.e. double A[n][m] ---> const Eigen::MatrixBase<Derived> &A_
        """
        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        newtype = "const Eigen::MatrixBase<Derived> &"

        return o.reconstruct(newtype, ast.Symbol("%s_" % name))

    def visit_Symbol(self, o, *args, **kwargs):
        """Visits a COFFEE symbol and redefines it as a Symbol with
        FunCall indexing.

        i.e. A[j][k] ---> A(j, k)
        """
        name = kwargs.get("name", "A")
        if o.symbol != name:
            return o

        return SymbolWithFuncallIndexing(o.symbol, o.rank, o.offset)


def slate_to_gem(expression):
    """Convert a slate expression to gem.

    :arg expression: A slate expression.
    :returns: A singleton list of gem expressions and a mapping from
        gem variables to UFL "terminal" forms.
    """

    mapper, var2terminal = slate2gem(expression)
    return mapper, var2terminal


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    name = f"T{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Variable(name, shape)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Block)
def _slate2gem_block(expr, self):
    child, = map(self, expr.children)
    child_shapes = expr.children[0].shapes
    offsets = tuple(sum(shape[:idx]) for shape, (idx, *_)
                    in zip(child_shapes.values(), expr._indices))
    return view(child, *(slice(idx, idx+extent) for idx, extent in zip(offsets, expr.shape)))


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    return Inverse(*map(self, expr.children))

@_slate2gem.register(sl.Action)
def _slate2gem_action(expr, self):
    name = f"A{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Action(*map(self, expr.children), name, expr.pick_op)
    self.var2terminal[var] = expr
    return var

@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    if expr.is_matfree():
        name = f"S{len(self.var2terminal)}"
        assert expr not in self.var2terminal.values()
        var = Solve(*map(self, expr.children), name, expr.is_matfree(), self(expr._Aonx), self(expr._Aonp))
        self.var2terminal[var] = expr
        return var
    else:
        return Solve(*map(self, expr.children))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Product(Literal(-1),
                           Indexed(child, indices)),
                           indices)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.children)
    indices = tuple(make_indices(len(A.shape)))
    var = ComponentTensor(Sum(Indexed(A, indices),
                           Indexed(B, indices)),
                           indices)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.children)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                    Indexed(B, tuple([k] + j)))
    var = ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A



def slate2gem(expression):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    return mapper(expression), mapper.var2terminal


def depth_first_search(graph, node, visited, schedule):
    """A recursive depth-first search (DFS) algorithm for
    traversing a DAG consisting of Slate expressions.

    :arg graph: A DAG whose nodes (vertices) are Slate expressions
                with edges connected to dependent expressions.
    :arg node: A starting vertex.
    :arg visited: A set keeping track of visited nodes.
    :arg schedule: A list of reverse-postordered nodes. This list is
                   used to produce a topologically sorted list of
                   Slate nodes.
    """
    if node not in visited:
        visited.add(node)

        for n in graph[node]:
            depth_first_search(graph, n, visited, schedule)

        schedule.append(node)


def topological_sort(exprs):
    """Topologically sorts a list of Slate expressions. The
    expression graph is constructed by relating each Slate
    node with a list of dependent Slate nodes.

    :arg exprs: A list of Slate expressions.
    """
    graph = OrderedDict((expr, set(traverse_dags([expr])) - {expr})
                        for expr in exprs)

    schedule = []
    visited = set()
    for n in graph:
        depth_first_search(graph, n, visited, schedule)

    return schedule


def merge_loopy(slate_loopy, output_arg, builder, var2terminal, gem2pym, strategy="terminals_first", slate_expr = None, tsfc_parameters=None):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""
    
    if isinstance(slate_loopy, lp.program.Program):
        slate_loopy = slate_loopy.root_kernel

    if strategy == "terminals_first":
        tensor2temp, tsfc_kernels, insns, builder = assemble_terminals_first(builder, var2terminal, slate_loopy)
    elif strategy == "when_needed":
        tensor2temp, tsfc_kernels, insns, builder = assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr, gem2pym)

    # FIXME for some reason the temporaries in the tsfc kernels don't have a shape
    all_kernels = itertools.chain([slate_loopy], tsfc_kernels)
    # Construct args
    import loopy
    args = [output_arg] + builder.generate_wrapper_kernel_args(tensor2temp, list(all_kernels))
    for a in slate_loopy.args:
        if a.name not in [arg.name for arg in args] and a.name.startswith("S"):
            ac = a.copy(address_space=loopy.AddressSpace.LOCAL)
            args.append(ac)

    # Inames come from initialisations + loopyfying kernel args and lhs
    domains = slate_loopy.domains + builder.bag.index_creator.domains

    # The problem here is that some of the actions in the kernel get replaced by multiple tsfc calls.
    # So we need to introduce new ids on those calls to keep them unique.
    # But some the dependencies in the local matfree kernel are hand written and depend on the
    # original action id. At this point all the instructions should be ensured to be sorted, so
    # we remove all existing dependencies and make them sequential instead
    # also help scheduling by setting within_inames_is_final on everything
    insns_new = []
    for i, insn in enumerate(insns):
        if insn:
            insns_new.append(insn.copy(depends_on=frozenset({}),
            priority=len(insns)-i,
            within_inames_is_final=True))

    # Generates the loopy wrapper kernel
    slate_wrapper = lp.make_function(domains, insns_new, args, name="slate_wrapper",
                                     seq_dependencies=True, target=lp.CTarget())

    # Prevent loopy interchange by loopy
    slate_wrapper = lp.prioritize_loops(slate_wrapper, ",".join(builder.bag.index_creator.inames.keys()))

    # Generate program from kernel, so that one can register kernels
    prg = make_program(slate_wrapper)
    loop = itertools.chain(tsfc_kernels, [slate_loopy]) if strategy == "terminals_first" else tsfc_kernels
    for knl in loop:
        if knl:
        # FIXME we might need to inline properly here for inlining the tsfc calls
        # that is so, because we first inline the solve calls with a cg loopy kernel
        # which contains actions and then we inline those actions after that
            if isinstance(knl, lp.program.Program):
                prg = inline_kernel_properly(prg, knl)
            else:
                prg = register_callable_kernel(prg, knl)
                from loopy.transform.callable import _match_caller_callee_argument_dimension_
                prg = _match_caller_callee_argument_dimension_(prg, knl.name)
                prg = inline_callable_kernel(prg, knl.name)
    return prg


def assemble_terminals_first(builder, var2terminal, slate_loopy):
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs, _ = builder.collect_coefficients()
    builder.bag = SlateWrapperBag(coeffs, slate_loopy.name)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), var2terminal.values()))
    tsfc_calls, tsfc_kernels = zip(*itertools.chain.from_iterable(
                                   (builder.generate_tsfc_calls(terminal, tensor2temp[terminal])
                                    for terminal in terminal_tensors)))

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))

    return tensor2temp, tsfc_kernels, insns, builder


def assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr, gem2pym):
    insns = []
    tsfc_knl_list = []
    tensor2temps = OrderedDict()

    # Keeping track off all coefficients upfront
    # saves us the effort of one of those ugly dict comparisons
    coeffs = {}  # all coefficients including the ones for the action
    new_coeffs = {}  # coeffs coming from action
    old_coeffs = {}  # only old coeffs minus the ones replaced by the action coefficients

    # invert dict
    import pymbolic.primitives as pym
    pyms = [pyms.name if isinstance(pyms, pym.Variable) else pyms.assignee_name for pyms in gem2pym.values()]
    pym2gem = OrderedDict(zip(pyms, gem2pym.keys()))
    c = 0 
    for insn in slate_loopy.instructions:
        if isinstance(insn, lp.kernel.instruction.CallInstruction):
            if (insn.expression.function.name.startswith("action") or
                insn.expression.function.name.startswith("solve")):
                c += 1

                # the name of the lhs can change due to inlining,
                # the indirections do only partially contain the right information
                lhs = insn.assignees[0].subscript.aggregate
                gem_action_node = pym2gem[lhs.name]  # we only need this node to the shape
                slate_node = var2terminal[gem_action_node]
                gem_inlined_node = Variable(lhs.name, gem_action_node.shape)
                terminal = slate_node.action()

                # link the coefficient of the action to the right tensor
                coeff = slate_node.ufl_coefficient
                coeff_name = insn.expression.parameters[1].subscript.aggregate.name
                old_coeff, new_coeff = builder.collect_coefficients([coeff],
                                                                    names={coeff._ufl_function_space:coeff_name},
                                                                    action_node=terminal)
                coeffs.update(old_coeff)
                coeffs.update(new_coeff)
                new_coeffs.update(new_coeff)
                old_coeffs.update(old_coeff)

                from firedrake.slate.slac.kernel_builder import SlateWrapperBag
                if not builder.bag:
                    builder.bag = SlateWrapperBag(old_coeffs, "_"+str(c), new_coeff, name)
                    builder.bag.call_name_generator("_"+str(c))
                else:
                    builder.bag.update_coefficients(old_coeffs, "_"+str(c), new_coeff)

                if terminal not in tensor2temps.keys():
                    inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: terminal}, builder.bag.coefficients)
                    tensor2temps.update(tensor2temp)

                    # temporaries that are filled with calls, which get inlined later,
                    # need to be initialised
                    for init in inits:
                        insns.append(init)
                
                # local assembly of the action or the matrix for the solve
                tsfc_calls, tsfc_knls = zip(*builder.generate_tsfc_calls(terminal, tensor2temps[terminal]))
                if tsfc_calls[0] and tsfc_knls[0]:
                    tsfc_knl_list.extend(tsfc_knls)

                    if isinstance(slate_node, sl.Action):
                        # substitute action call with the generated tsfc call for that action
                        # but keep the lhs so that the following instructions still act on the right temporaries
                        for i, tsfc_call in enumerate(tsfc_calls):
                            insns.append(lp.kernel.instruction.CallInstruction(insn.assignees,
                                                                            tsfc_call.expression,
                                                                            id=insn.id+"_inlined_tsfc_"+str(i),
                                                                            within_inames=insn.within_inames,
                                                                            predicates=tsfc_call.predicates))
                    else:
                        # TODO we need to be able to this in case someone wants a matrix explicit solve
                        # If we want an explicit solve, we need to assemble matrix first
                        # FIXME in fact this does not work yet, we got trouble with instructions not containing the right temps
                        insns.append(tsfc_calls[0])
                        insns.append(insn)
                else:
                    insns.append(lp.kernel.instruction.Assignment(insn.assignees[0].subscript, insn.expression.parameters[1].subscript))

        else:
            insns.append(insn)

    # Initialise the very first temporary
    # For that we need to get the temporary which
    # links to the same coefficient as the rhs of this node and init it              
    init_coeffs,_ = builder.collect_coefficients()
    var2terminal_vectors = {v:t for (v,t) in var2terminal.items()
                                for (cv,ct) in init_coeffs.items()
                                if isinstance(t, sl.AssembledVector)
                                and t._function==cv}
    inits, tensor2temp = builder.initialise_terminals(var2terminal_vectors, init_coeffs)            
    tensor2temps.update(tensor2temp)
    for i in inits:
        insns.insert(0, i)

    # Get all coeffs into the wrapper kernel
    # so that we can generate the right wrapper kernel args of it
    builder.bag.update_coefficients(init_coeffs, "_"+str(c), new_coeffs)

    return tensor2temps, tsfc_knl_list, insns, builder

def inline_kernel_properly(wrapper, kernel):

    from loopy.transform.callable import _match_caller_callee_argument_dimension_

    # Generate program from kernel, so that one can register kernels
    from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
    from loopy.kernel.function_interface import CallableKernel

    for tsfc_loopy in tsfc_kernels:
        slate_wrapper = merge([slate_wrapper, tsfc_loopy])
        names = tsfc_loopy.callables_table
        for name in names:
            if isinstance(slate_wrapper.callables_table[name], CallableKernel):
                slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)
    slate_wrapper = merge([slate_wrapper, slate_loopy])
    names = slate_loopy.callables_table
    for name in names:
        if isinstance(slate_wrapper.callables_table[name], CallableKernel):
            slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)

    return slate_wrapper
