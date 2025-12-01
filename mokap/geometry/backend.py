USE_JAX = False


if USE_JAX:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.typing import ArrayLike

    # JAX (float32 default)
    # Threshold to switch to Taylor series (float32 needs this earlier than float64)
    _eps = 1e-4

    # A very tiny value to avoid division by zero
    _tiny = 1e-12

    # Or match precision:
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    xp = jnp
    jit = jax.jit
    vmap = jax.vmap

    def set_at(arr, indices, values, inplace=True):
        return arr.at[indices].set(values)

    print(f'Mokap math backend: JAX')

else:
    from itertools import repeat
    import numpy as np
    from numpy.typing import ArrayLike

    # NumPy (float64 default)
    # Threshold can be much smaller due to higher precision
    _eps = 1e-8

    # A very tiny value to avoid division by zero
    _tiny = 1e-12

    xp = np

    def set_at(arr, indices, values, inplace=True):
        if not inplace:
            arr_new = arr.copy()
        else:
            arr_new = arr
        arr_new[indices] = values
        return arr_new

    # No-op JIT
    def jit(fun, static_argnums=None, static_argnames=None, **kwargs):
        return fun

    # Minimal LAX for explicit loops (scan/while), not for conditionals
    class LaxShim:
        @staticmethod
        def fori_loop(lower, upper, body_fun, init_val):
            val = init_val
            for i in range(int(lower), int(upper)):
                val = body_fun(i, val)
            return val

        @staticmethod
        def while_loop(cond_fun, body_fun, init_val):
            val = init_val
            while cond_fun(val):
                val = body_fun(val)
            return val
    lax = LaxShim()

    # vmap shim
    def vmap(fun, in_axes=0, out_axes=0):

        def wrapper(*args, **kwargs):
            # Normalise in_axes
            if isinstance(in_axes, int) or in_axes is None:
                axes = [in_axes] * len(args)
            else:
                axes = in_axes

            if len(axes) != len(args):
                raise ValueError("vmap: in_axes length must match args length")

            # Prepare iterators for zipping
            iterators = []
            for arg, axis in zip(args, axes):
                if axis is None:
                    # Broadcast: repeat the constant infinitely
                    iterators.append(repeat(arg))
                elif axis == 0:
                    # Standard iteration
                    iterators.append(arg)
                else:
                    # Move target axis to front to iterate over it
                    iterators.append(np.moveaxis(arg, axis, 0))

            # We assume the non-repeated args determine the batch size implicitly
            results = [fun(*x, **kwargs) for x in zip(*iterators)]

            if not results:
                return np.array([])

            first_res = results[0]

            # Stack results (handle multiple outputs)
            if isinstance(first_res, (tuple, list)):
                transposed = zip(*results)

                if isinstance(out_axes, int):
                    o_axes = [out_axes] * len(first_res)
                else:
                    o_axes = out_axes

                stacked = []
                for res_col, ax in zip(transposed, o_axes):
                    stack_axis = ax if ax is not None else 0
                    stacked.append(np.stack(list(res_col), axis=stack_axis))

                return type(first_res)(stacked)
            else:
                stack_axis = out_axes if out_axes is not None else 0
                return np.stack(results, axis=stack_axis)

        return wrapper

    print(f'Mokap math backend: NumPy')


def align_batch_dims(target_ndim: int, arr: xp.ndarray, data_dims: int = 1) -> xp.ndarray:
    """
    Helper to ensure 'arr' broadcasts correctly against a data array with 'target_ndim' batch dimensions.
    It inserts singleton dimensions between the array's batch dims and its data dims.

    Args:
        target_ndim: The number of batch dimensions in the reference data (e.g. points)
        arr: The parameter array (e.g. K, D, rvec)
        data_dims: How many dimensions at the end of 'arr' are data (1 for vec, 2 for matrix)
    """
    arr = xp.asarray(arr)
    arr_batch_ndim = arr.ndim - data_dims
    pad_needed = target_ndim - arr_batch_ndim
    if pad_needed > 0:
        # Handle data_dims=0 case where slicing with [:-0] returns empty tuple
        if data_dims == 0:
            # For scalars/1D arrays, we append 1s at the end
            # e.g. (5,) -> (5, 1) to match (5, 6)
            new_shape = arr.shape + (1,) * pad_needed
        else:
            # Insert 1s before the data dimensions
            # e.g. (5, 3, 3) -> (5, 1, 3, 3) to match (5, 6, ...)
            new_shape = arr.shape[:-data_dims] + (1,) * pad_needed + arr.shape[-data_dims:]

        return arr.reshape(new_shape)
    return arr
