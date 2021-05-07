# Some generic python utilities not really specific to our work.
from decorator import decorator
import functools

from pyop2.utils import cached_property  # noqa: F401
from pyop2.datatypes import ScalarType, as_cstr
from pyop2.datatypes import RealType     # noqa: F401
from pyop2.datatypes import IntType      # noqa: F401
from pyop2.datatypes import as_ctypes    # noqa: F401
from firedrake_configuration import get_config
from firedrake.petsc import get_petsc_variables

_current_uid = 0

ScalarType_c = as_cstr(ScalarType)
IntType_c = as_cstr(IntType)

complex_mode = get_config()["options"].get("complex", False)

# Remove this (and update test suite) when Slate supports complex mode.
SLATE_SUPPORTS_COMPLEX = False


def _new_uid():
    global _current_uid
    _current_uid += 1
    return _current_uid


def _init():
    """Cause :func:`pyop2.init` to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    :func:`pyop2.init` if she wants to set a non-default option, for example
    to switch the debug or log level."""
    from pyop2 import op2
    from firedrake.parameters import parameters
    if not op2.initialised():
        op2.init(**parameters["pyop2_options"])


def unique_name(name, nameset):
    """Return name if name is not in nameset, or a deterministic
    uniquified name if name is in nameset. The new name is inserted into
    nameset to prevent further name clashes."""

    if name not in nameset:
        nameset.add(name)
        return name

    idx = 0
    while True:
        newname = "%s_%d" % (name, idx)
        if newname in nameset:
            idx += 1
        else:
            nameset.add(name)
            return newname


def known_pyop2_safe(f):
    """Decorator to mark a function as being PyOP2 type-safe.

    This switches the current PyOP2 type checking mode to the value
    given by the parameter "type_check_safe_par_loops", and restores
    it after the function completes."""
    from firedrake.parameters import parameters

    def wrapper(f, *args, **kwargs):
        opts = parameters["pyop2_options"]
        check = opts["type_check"]
        safe = parameters["type_check_safe_par_loops"]
        if check == safe:
            return f(*args, **kwargs)
        opts["type_check"] = safe
        try:
            return f(*args, **kwargs)
        finally:
            opts["type_check"] = check
    return decorator(wrapper, f)

@functools.lru_cache(maxsize=None)
def get_eigen_include_dir():
    """Return the include directory for Eigen.
    
    Depending on how Eigen was installed this will either be defined in
    petscvariables or the Firedrake configuration file.

    Returns ``None`` if not found.
    """
    try:
        return get_petsc_variables()["EIGEN_INCLUDE_DIR"].lstrip("-I")
    except KeyError:
        try:
            return get_config()["libraries"]["EIGEN_INCLUDE_DIR"]
        except KeyError:
            return None
