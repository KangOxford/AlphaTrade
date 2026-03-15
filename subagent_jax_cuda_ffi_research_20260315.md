# JAX CUDA FFI Integration Research (JAX 0.9.0.1)

Date: 2026-03-15

## Executive Summary

JAX 0.9.0.1 provides a **Foreign Function Interface (FFI)** via `jax.ffi` for integrating custom CUDA kernels.
The modern API supersedes the old `xla_client.register_custom_call_target` / `jax.interpreters.mlir.custom_call` approach.

There are **two methods** to expose CUDA code to JAX:
1. **ctypes + `XLA_FFI_DEFINE_HANDLER_SYMBOL`** -- simpler, no Python binding library needed
2. **nanobind + `XLA_FFI_DEFINE_HANDLER`** -- more flexible, needed for stateful ops or complex registration

---

## 1. Architecture Overview

```
+------------------+     +-------------------+     +------------------+
| Python (JAX)     |     | Shared Library    |     | CUDA Kernel      |
|                  |     | (.so)             |     | (.cu)            |
| jax.ffi.ffi_call |--->| XLA FFI Handler   |--->| __global__ void  |
|   "target_name"  |     | (CPU function     |     |   kernel(...)    |
|                  |     |  that launches    |     |                  |
|                  |     |  CUDA kernels)    |     |                  |
+------------------+     +-------------------+     +------------------+
        |                        ^
        | register_ffi_target    | ctypes.cdll.LoadLibrary
        +------------------------+   or nanobind module
```

Key insight: The FFI handler is a **CPU function** that receives a `cudaStream_t` and
**enqueues GPU work** by launching CUDA kernels on that stream. Buffer pointers
are **device pointers** (GPU memory), not host memory.

---

## 2. Complete Python API (verified on JAX 0.9.0.1)

### 2.1 `jax.ffi.ffi_call`

```python
jax.ffi.ffi_call(
    target_name: str,                              # Registered FFI target name
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],  # Output specs
    *,
    has_side_effect: bool = False,                 # Force execution even if outputs unused
    vmap_method: str | None = None,                # "sequential", "broadcast_all", "expand_dims"
    input_layouts: Sequence[FfiLayoutOptions] | None = None,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = None,
    input_output_aliases: dict[int, int] | None = None,  # In-place operations
    custom_call_api_version: int = 4,              # 4 = typed FFI (default)
    legacy_backend_config: str | None = None,      # For api_version < 4 only
    vectorized: bool | None | DeprecatedArg = Deprecated  # DEPRECATED
) -> Callable[..., Array | Sequence[Array]]
```

**Returns a callable**. Positional args = input arrays. Keyword args = FFI attributes (must be static Python values, NOT jax arrays).

```python
# Single output
result = jax.ffi.ffi_call("my_op", jax.ShapeDtypeStruct(shape, dtype))(x, y, eps=np.float32(1e-5))

# Multiple outputs
out1, out2 = jax.ffi.ffi_call("my_op", (
    jax.ShapeDtypeStruct(shape1, dtype1),
    jax.ShapeDtypeStruct(shape2, dtype2),
))(x, y, n=np.uint64(42))
```

### 2.2 `jax.ffi.register_ffi_target`

```python
jax.ffi.register_ffi_target(
    name: str,           # Target name (must match ffi_call's target_name)
    fn: Any,             # PyCapsule or dict of PyCapsules for multi-stage
    platform: str = 'cpu',   # "cpu", "CUDA", "ROCM"
    api_version: int = 1,    # 1 = typed FFI (default, modern), 0 = legacy
    **kwargs: Any
) -> None
```

### 2.3 `jax.ffi.pycapsule`

```python
jax.ffi.pycapsule(funcptr) -> PyCapsule
```

Wraps a ctypes function pointer into a PyCapsule for registration.

### 2.4 `jax.ffi.include_dir`

```python
jax.ffi.include_dir() -> str
# Returns: /path/to/jaxlib/include
# On this system: /projects/s5e/quant/miniforge3/lib/python3.12/site-packages/jaxlib/include
```

### 2.5 Other APIs

```python
# Mark an FFI target as supporting batch partitioning
jax.ffi.register_ffi_target_as_batch_partitionable(name: str) -> None

# Register custom types for stateful FFI ops
jax.ffi.register_ffi_type(name, type_registration, platform='cpu') -> None
jax.ffi.register_ffi_type_id(name, obj, platform='cpu') -> None
```

---

## 3. vmap_method Options (Batching Support)

| Method | Behavior | When to Use |
|--------|----------|-------------|
| `"sequential"` | Rewrites vmapped call as `jax.lax.scan()` | Safe default; works always but no parallelism |
| `"broadcast_all"` | Broadcasts all inputs to match batch dimensions | When kernel handles leading batch dims natively |
| `"expand_dims"` | Adds size-1 leading dimension to unbatched inputs | When kernel handles arbitrary leading dims |
| `None` | Will raise `NotImplementedError` under vmap (future default) | When vmap is not needed |

**Critical**: If your CUDA kernel handles leading batch dimensions (i.e., it loops over
`totalSize / lastDim` batches internally), use `vmap_method="broadcast_all"`.
This is the recommended approach for most kernels.

---

## 4. C++ XLA FFI API

### 4.1 Header

```cpp
#include "xla/ffi/api/ffi.h"    // Main header (header-only)
// Located at: $(python3 -c "from jax import ffi; print(ffi.include_dir())")/xla/ffi/api/ffi.h
```

Available on this system at:
```
/projects/s5e/quant/miniforge3/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/ffi.h
```

### 4.2 Buffer Types

```cpp
namespace ffi = xla::ffi;

// Input buffers (read-only device pointers)
ffi::Buffer<ffi::F32>         // Typed, any rank
ffi::BufferR0<ffi::F32>       // Typed, rank 0 (scalar)
ffi::BufferR2<ffi::F32>       // Typed, rank 2
ffi::AnyBuffer                // Any type, any rank

// Output buffers (writable device pointers)
ffi::ResultBuffer<ffi::F32>   // -> access via operator->
ffi::Result<ffi::AnyBuffer>   // -> access via operator->

// Accessing data
buffer.typed_data()           // const T* (input)
buffer.typed_data<T>()        // const T* (AnyBuffer, explicit type)
result->typed_data()          // T* (output, note -> not .)
buffer.dimensions()           // span of int64_t
buffer.element_count()        // total number of elements
buffer.element_type()         // ffi::DataType enum
```

### 4.3 Data Types

```
ffi::PRED, ffi::S8, ffi::S16, ffi::S32, ffi::S64
ffi::U8, ffi::U16, ffi::U32, ffi::U64
ffi::F16, ffi::BF16, ffi::F32, ffi::F64
ffi::C64, ffi::C128
```

### 4.4 Handler Definition

Two macros, choose based on how you expose to Python:

#### Method A: `XLA_FFI_DEFINE_HANDLER_SYMBOL` (for ctypes)

Creates a symbol with **C linkage** loadable via `ctypes.cdll.LoadLibrary`.

```cpp
// Declaration (optional, for headers):
XLA_FFI_DECLARE_HANDLER_SYMBOL(MyHandler);

// Definition:
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MyHandler,          // Symbol name (C linkage)
    MyImplFunction,     // C++ implementation function
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // GPU stream
        .Arg<ffi::Buffer<ffi::F32>>()              // input 0
        .Arg<ffi::Buffer<ffi::F32>>()              // input 1
        .Ret<ffi::Buffer<ffi::F32>>()              // output 0
        .Attr<float>("eps")                        // named attribute
        .Attr<size_t>("n"),                        // named attribute
    {xla::ffi::Traits::kCmdBufferCompatible}       // optional: enable cudaGraph
);
```

#### Method B: `XLA_FFI_DEFINE_HANDLER` (for nanobind)

Creates a handler variable (not C-linkage symbol). Wrap in nanobind to expose.

```cpp
XLA_FFI_DEFINE_HANDLER(
    kMyHandler,         // Variable name
    MyImplFunction,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

// Then in nanobind:
NB_MODULE(_my_module, m) {
    m.def("handler", []() {
        return nb::capsule(reinterpret_cast<void*>(kMyHandler));
    });
}
```

### 4.5 Error Handling

```cpp
ffi::Error MyImpl(...) {
    // ... launch kernel ...
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// Other error types:
ffi::Error::InvalidArgument("message")
ffi::Error::Internal("message")
```

### 4.6 Binding Chain Reference

```cpp
ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream context
    .Ctx<ffi::State<MyType>>()                 // Custom state (stateful ops)
    .Arg<ffi::Buffer<ffi::F32>>()              // Typed input buffer
    .Arg<ffi::AnyBuffer>()                     // Any-type input buffer
    .Ret<ffi::Buffer<ffi::F32>>()              // Typed output buffer
    .Ret<ffi::AnyBuffer>()                     // Any-type output buffer
    .Attr<float>("name")                       // Named attribute (float)
    .Attr<int64_t>("name")                     // Named attribute (int64)
    .Attr<size_t>("name")                      // Named attribute (size_t)
    .RemainingArgs()                           // Variadic inputs
    .RemainingRets()                           // Variadic outputs
```

---

## 5. Complete End-to-End Example: CUDA Kernel with JAX FFI

### 5.1 Directory Structure

```
my_cuda_op/
  CMakeLists.txt
  pyproject.toml              # optional, for pip install
  src/
    my_cuda_op/
      __init__.py
      cuda_kernel.cu          # CUDA kernel + FFI handler
      wrapper.py              # Python wrapper with jax.ffi
```

### 5.2 CUDA Kernel (`cuda_kernel.cu`)

```cuda
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ============ CUDA Kernel ============
__global__ void MyKernel(const float* input, float* output,
                         int64_t n, float scale) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = tid; i < n; i += grid_stride) {
        output[i] = input[i] * scale;
    }
}

// ============ FFI Host Function ============
// This runs on CPU but launches GPU work via the stream
ffi::Error MyOpHost(
    cudaStream_t stream,                    // from .Ctx<PlatformStream>
    float scale,                            // from .Attr<float>("scale")
    ffi::Buffer<ffi::F32> input,           // from .Arg<Buffer<F32>>()
    ffi::ResultBuffer<ffi::F32> output     // from .Ret<Buffer<F32>>()
) {
    int64_t n = input.element_count();

    const int block_dim = 256;
    const int grid_dim = std::min<int>(1024, (n + block_dim - 1) / block_dim);

    MyKernel<<<grid_dim, block_dim, 0, stream>>>(
        input.typed_data(),      // const float* (device pointer)
        output->typed_data(),    // float* (device pointer, note ->)
        n,
        scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// ============ Handler Registration ============
// XLA_FFI_DEFINE_HANDLER_SYMBOL creates C-linkage symbol for ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MyOp,                                               // Symbol name
    MyOpHost,                                           // Implementation
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()      // CUDA stream
        .Attr<float>("scale")                           // Attribute
        .Arg<ffi::Buffer<ffi::F32>>()                  // Input
        .Ret<ffi::Buffer<ffi::F32>>(),                 // Output
    {xla::ffi::Traits::kCmdBufferCompatible}            // Enable cudaGraph
);
```

### 5.3 Python Wrapper (`wrapper.py`)

```python
import os
import ctypes
import numpy as np
import jax
import jax.numpy as jnp

# ============ Load and Register ============
_SHARED_LIB = os.path.join(os.path.dirname(__file__), "lib_cuda_kernel.so")
_library = ctypes.cdll.LoadLibrary(_SHARED_LIB)

jax.ffi.register_ffi_target(
    "my_op",                                    # target name
    jax.ffi.pycapsule(_library.MyOp),           # PyCapsule from C-linkage symbol
    platform="CUDA",                            # GPU platform
)

# ============ Python Function ============
def my_op(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """Apply custom CUDA operation: output = input * scale."""
    assert x.dtype == jnp.float32, f"Expected float32, got {x.dtype}"

    out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    return jax.ffi.ffi_call(
        "my_op",
        out_type,
        vmap_method="broadcast_all",    # Kernel handles batch dims
    )(x, scale=np.float32(scale))       # Attrs must be numpy, NOT jax arrays

# ============ Usage ============
# x = jnp.ones((8, 4), dtype=jnp.float32)
# result = my_op(x, scale=2.0)           # Works with jit
# result = jax.jit(my_op)(x, scale=2.0)  # Also works
# result = jax.vmap(my_op)(x)            # Works because vmap_method="broadcast_all"
```

### 5.4 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15...3.30)
project(my_cuda_op LANGUAGES CXX CUDA)

# Find Python and get XLA include directory
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "-c" "from jax import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(CUDAToolkit REQUIRED)

# Build CUDA shared library
add_library(lib_cuda_kernel SHARED src/my_cuda_op/cuda_kernel.cu)
set_target_properties(lib_cuda_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
)
target_include_directories(lib_cuda_kernel PUBLIC ${XLA_DIR})
install(TARGETS lib_cuda_kernel LIBRARY DESTINATION my_cuda_op)
```

### 5.5 Manual Build (without CMake)

```bash
# Get include directory
XLA_DIR=$(python3 -c "from jax import ffi; print(ffi.include_dir())")

# Compile CUDA kernel to .so
nvcc -shared -o lib_cuda_kernel.so \
    -std=c++17 \
    -Xcompiler -fPIC \
    -I${XLA_DIR} \
    cuda_kernel.cu
```

---

## 6. Advanced: Custom VJP (Backward Pass)

### 6.1 CUDA Side (Forward + Backward Kernels)

```cuda
// Forward: c = a * (b + 1), also save intermediate b_plus_1
__global__ void FwdKernel(const float* a, const float* b,
                          float* c, float* b_plus_1, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += grid_stride) {
        b_plus_1[i] = b[i] + 1.0f;
        c[i] = a[i] * b_plus_1[i];
    }
}

// Backward: da = dc * (b+1), db = dc * a
__global__ void BwdKernel(const float* c_grad, const float* a,
                          const float* b_plus_1,
                          float* a_grad, float* b_grad, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += grid_stride) {
        a_grad[i] = c_grad[i] * b_plus_1[i];
        b_grad[i] = c_grad[i] * a[i];
    }
}

ffi::Error FwdHost(cudaStream_t stream,
                   ffi::Buffer<ffi::F32> a, ffi::Buffer<ffi::F32> b,
                   ffi::ResultBuffer<ffi::F32> c,
                   ffi::ResultBuffer<ffi::F32> b_plus_1,
                   size_t n) {
    const int block_dim = 128;
    const int grid_dim = 1;
    FwdKernel<<<grid_dim, block_dim, 0, stream>>>(
        a.typed_data(), b.typed_data(),
        c->typed_data(), b_plus_1->typed_data(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error BwdHost(cudaStream_t stream,
                   ffi::Buffer<ffi::F32> c_grad,
                   ffi::Buffer<ffi::F32> a,
                   ffi::Buffer<ffi::F32> b_plus_1,
                   ffi::ResultBuffer<ffi::F32> a_grad,
                   ffi::ResultBuffer<ffi::F32> b_grad,
                   size_t n) {
    const int block_dim = 128;
    const int grid_dim = 1;
    BwdKernel<<<grid_dim, block_dim, 0, stream>>>(
        c_grad.typed_data(), a.typed_data(), b_plus_1->typed_data(),
        a_grad->typed_data(), b_grad->typed_data(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MyFwd, FwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()     // a
        .Arg<ffi::Buffer<ffi::F32>>()     // b
        .Ret<ffi::Buffer<ffi::F32>>()     // c
        .Ret<ffi::Buffer<ffi::F32>>()     // b_plus_1 (residual)
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(MyBwd, BwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()     // c_grad
        .Arg<ffi::Buffer<ffi::F32>>()     // a
        .Arg<ffi::Buffer<ffi::F32>>()     // b_plus_1
        .Ret<ffi::Buffer<ffi::F32>>()     // a_grad
        .Ret<ffi::Buffer<ffi::F32>>()     // b_grad
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});
```

### 6.2 Python Side (custom_vjp)

```python
import os, ctypes
import numpy as np
import jax
import jax.numpy as jnp

lib = ctypes.cdll.LoadLibrary("lib_my_op.so")
jax.ffi.register_ffi_target("my_fwd", jax.ffi.pycapsule(lib.MyFwd), platform="CUDA")
jax.ffi.register_ffi_target("my_bwd", jax.ffi.pycapsule(lib.MyBwd), platform="CUDA")

def my_fwd(a, b):
    n = np.prod(a.shape).astype(np.uint64)
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    c, b_plus_1 = jax.ffi.ffi_call("my_fwd", (out_type, out_type))(a, b, n=n)
    return c, (a, b_plus_1)  # (primal_out, residuals)

def my_bwd(res, c_grad):
    a, b_plus_1 = res
    n = np.prod(a.shape).astype(np.uint64)
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    return jax.ffi.ffi_call("my_bwd", (out_type, out_type))(c_grad, a, b_plus_1, n=n)

@jax.custom_vjp
def my_op(a, b):
    c, _ = my_fwd(a, b)
    return c

my_op.defvjp(my_fwd, my_bwd)

# Now jax.grad(my_op) works!
```

---

## 7. Advanced: Sharding / Multi-GPU

### 7.1 shard_map

```python
from functools import partial
from jax.sharding import PartitionSpec as P

mesh = jax.make_mesh((4,), ("x",))

@partial(jax.shard_map, mesh=mesh,
         in_specs=P("x", None), out_specs=P("x", None))
def my_op_sharded(x):
    return my_op(x)
```

### 7.2 custom_partitioning

```python
from jax.experimental.custom_partitioning import custom_partitioning

@partial(custom_partitioning, static_argnums=(1,))
def my_op_partitioned(x, scale=1.0):
    return my_op(x, scale=scale)

def infer_sharding(scale, mesh, args_info, result_info):
    return result_info.sharding

def partition(scale, mesh, args_info, result_info):
    return mesh, lambda x: my_op(x, scale), result_info.sharding, (args_info[0].sharding,)

my_op_partitioned.def_partition(
    infer_sharding_from_operands=infer_sharding,
    partition=partition,
)
```

---

## 8. Key Constraints & Gotchas

### 8.1 Attributes Must Be Static

```python
# WRONG: JAX arrays as attributes
jax.ffi.ffi_call("op", out)(x, eps=jnp.float32(1e-5))  # ERROR

# CORRECT: numpy scalars as attributes
jax.ffi.ffi_call("op", out)(x, eps=np.float32(1e-5))    # OK
```

### 8.2 Buffer Pointers Are Device Pointers

The FFI handler runs on CPU but buffer data is on GPU. You CANNOT dereference
buffer pointers on the host. You must launch CUDA kernels or use `cudaMemcpy`.

### 8.3 cudaStream Synchronization

Do NOT call `cudaStreamSynchronize` in the handler unless absolutely necessary.
XLA manages its own synchronization. Synchronizing will serialize execution and
kill performance.

### 8.4 Error Checking

Always check `cudaGetLastError()` after kernel launch. Note this may return
errors from PREVIOUS asynchronous launches by XLA.

### 8.5 input_output_aliases (In-Place Operations)

```python
# Tell XLA that input 0 can be aliased with output 0 (in-place)
result = jax.ffi.ffi_call(
    "my_op",
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    input_output_aliases={0: 0},  # input[0] -> output[0]
)(x)
```

### 8.6 Platform String

For GPU registration, use `platform="CUDA"` (uppercase). The platform strings are:
- `"cpu"` (lowercase, default)
- `"CUDA"` (uppercase)
- `"ROCM"` (uppercase)

### 8.7 vmap_method Default Change

The `vectorized` parameter is **deprecated**. Use `vmap_method` instead.
Currently the default is `"sequential"` (fallback) but this will change to
`None` (raise error) in a future version. **Always explicitly set `vmap_method`**.

---

## 9. Build System Options

### 9.1 Option A: Simple nvcc (Recommended for Quick Integration)

```bash
XLA_DIR=$(python3 -c "from jax import ffi; print(ffi.include_dir())")

nvcc -shared -o lib_my_kernel.so \
    -std=c++17 \
    -Xcompiler -fPIC \
    -I${XLA_DIR} \
    my_kernel.cu

# For ARM (GH200):
nvcc -shared -o lib_my_kernel.so \
    -std=c++17 \
    -Xcompiler -fPIC \
    -I${XLA_DIR} \
    --gpu-architecture=sm_90 \
    my_kernel.cu
```

### 9.2 Option B: CMake + scikit-build-core

**pyproject.toml:**
```toml
[build-system]
requires = ["scikit-build-core", "nanobind", "jax>=0.4.31"]
build-backend = "scikit_build_core.build"

[project]
name = "my_cuda_op"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = ["jax"]
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15...3.30)
project(my_cuda_op LANGUAGES CXX CUDA)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "-c" "from jax import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)

find_package(CUDAToolkit REQUIRED)

# For ctypes approach: plain shared library
add_library(lib_cuda_kernel SHARED src/my_cuda_op/cuda_kernel.cu)
set_target_properties(lib_cuda_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
)
target_include_directories(lib_cuda_kernel PUBLIC ${XLA_DIR})
install(TARGETS lib_cuda_kernel LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# For nanobind approach: Python extension module
# find_package(nanobind CONFIG REQUIRED)
# nanobind_add_module(_my_module NB_STATIC src/my_cuda_op/bindings.cc)
# target_include_directories(_my_module PUBLIC ${XLA_DIR})
# target_link_libraries(_my_module PRIVATE CUDA::cudart)
# install(TARGETS _my_module LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
```

### 9.3 Option C: nanobind Module (for Complex Registration)

Used when you need:
- Stateful ops (`ffi::State<T>`)
- Multiple handlers bundled
- Complex type registration

```cpp
// bindings.cc
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

// Declare extern handlers defined in .cu file
extern XLA_FFI_Handler* MyOp;  // from XLA_FFI_DEFINE_HANDLER

NB_MODULE(_my_module, m) {
    m.def("registrations", []() {
        nb::dict d;
        d["my_op"] = nb::capsule(reinterpret_cast<void*>(MyOp));
        return d;
    });
}
```

```python
# Python side
from my_cuda_op import _my_module

for name, target in _my_module.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform="CUDA")
```

---

## 10. Comparison: Old API vs New FFI API

| Feature | Old API (deprecated) | New FFI API (current) |
|---------|---------------------|----------------------|
| Registration | `xla_client.register_custom_call_target()` | `jax.ffi.register_ffi_target()` |
| Call | `xla_client.ops.CustomCallWithLayout()` | `jax.ffi.ffi_call()` |
| Shape inference | `primitive.def_abstract_eval()` | Automatic from `result_shape_dtypes` |
| Batching | Manual `primitive.batching_rule` | `vmap_method=` parameter |
| CUDA signature | `void fn(CUstream, void**, char*, size_t)` | `ffi::Error fn(cudaStream_t, Buffer<T>, ResultBuffer<T>)` |
| Params passing | Packed opaque bytes | Named `.Attr<T>("name")` |
| Type safety | None (void** casting) | Compile-time type checking |
| API version | `api_version=0` | `api_version=1` (default) |

The old API required:
1. Defining a `core.Primitive`
2. Implementing `abstract_eval`
3. Implementing `mlir_lowering` or `xla_translation`
4. Implementing `batching_rule` for vmap
5. Packing config into opaque bytes

The new FFI API handles all of this automatically via `ffi_call` parameters.

---

## 11. Reference: Official JAX FFI Example Files

From the JAX repository (`jax-ml/jax/examples/ffi/`):

| File | Purpose |
|------|---------|
| `cuda_examples.cu` | Complete CUDA kernel with fwd/bwd, `XLA_FFI_DEFINE_HANDLER_SYMBOL` |
| `cuda_examples.py` | Python wrapper with `custom_vjp`, ctypes loading |
| `gpu_examples.cc` | Stateful GPU op with nanobind, `ffi::State<T>` |
| `gpu_examples.py` | Python wrapper for stateful op |
| `rms_norm.cc` | CPU example with `AnyBuffer`, multiple dtypes |
| `rms_norm.py` | Python wrapper with `custom_vjp` |
| `CMakeLists.txt` | Build system with CUDA support |
| `pyproject.toml` | Package config for scikit-build-core |

---

## 12. Relevant to Our Project: Order Book Matching Kernel

For integrating a CUDA order book matching kernel into JAX:

1. **Kernel**: CUDA kernel that operates on order book arrays
2. **Handler**: CPU function receiving `cudaStream_t`, launching kernel
3. **Registration**: `jax.ffi.register_ffi_target("lob_match", ..., platform="CUDA")`
4. **Wrapper**: `jax.ffi.ffi_call("lob_match", result_shapes, vmap_method="broadcast_all")`
5. **Build**: `nvcc -shared` or CMake
6. **No need for**: manual abstract_eval, manual batching rules, opaque bytes packing
7. **Config via**: `.Attr<int32_t>("max_orders")` etc. as named attributes

Key consideration: The kernel must be compiled for the correct GPU architecture
(sm_90 for GH200). Compilation should happen on a compute node or with the correct
CUDA cross-compilation flags.

---

## Sources

- [JAX FFI Documentation](https://docs.jax.dev/en/latest/ffi.html)
- [JAX FFI Examples (GitHub)](https://github.com/jax-ml/jax/tree/main/examples/ffi)
- [XLA Custom Calls (OpenXLA)](https://openxla.org/xla/custom_call)
- [jax.ffi.ffi_call API Reference](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html)
- [jax.ffi.register_ffi_target API Reference](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.register_ffi_target.html)
- [Extending JAX with C++ and CUDA (Dan Foreman-Mackey)](https://dfm.io/posts/extending-jax/)
- [Custom Operations for GPUs](https://docs.jax.dev/en/latest/Custom_Operation_for_GPUs.html)
- [vmap_method deprecation PR](https://github.com/jax-ml/jax/pull/23881)
