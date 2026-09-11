"""
HIP kernel compilation utilities for rocm-qlora.

Compiles .hip files using hipcc at import time if not already compiled.
Compiled shared libraries cached to avoid recompilation.
Falls back gracefully to Triton kernels if hipcc not available.

Public API:
    find_hipcc() -> str | None
    compile_hip_kernel(hip_file, output_so, arch="gfx942") -> bool
    load_hip_kernel(so_path) -> ctypes.CDLL | None
    get_or_compile_int4_kernel() -> object | None
"""

import os
import subprocess
import ctypes
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -- Cache directory for compiled .so files --
# Default: ~/.cache/rocm_qlora/hip_kernels
# Override via ROCM_QLORA_CACHE_DIR env var
_CACHE_DIR = os.environ.get(
    "ROCM_QLORA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "rocm_qlora", "hip_kernels")
)


def find_hipcc() -> Optional[str]:
    """
    Locate the hipcc compiler binary.

    Search order:
      1. ROCM_HOME/bin/hipcc
      2. /opt/rocm/bin/hipcc
      3. which hipcc (PATH lookup)

    Returns:
        Absolute path to hipcc as string, or None if not found.
    """
    # Check ROCM_HOME first
    rocm_home = os.environ.get("ROCM_HOME")
    if rocm_home:
        candidate = os.path.join(rocm_home, "bin", "hipcc")
        if os.path.isfile(candidate):
            return candidate
        # Also try hipcc.exe on Windows
        candidate_exe = candidate + ".exe"
        if os.path.isfile(candidate_exe):
            return candidate_exe

    # Check /opt/rocm (Linux convention)
    candidate = "/opt/rocm/bin/hipcc"
    if os.path.isfile(candidate):
        return candidate

    # Check /opt/rocm-<version>/bin/hipcc
    import glob
    for path in glob.glob("/opt/rocm-*/bin/hipcc"):
        if os.path.isfile(path):
            return path

    # PATH lookup via 'which'/'where'
    try:
        # On Windows, use 'where'; on Linux/Mac, use 'which'
        cmd = "where hipcc" if sys.platform == "win32" else "which hipcc"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            path = result.stdout.strip().split("\n")[0]
            if os.path.isfile(path):
                return path
    except Exception:
        pass

    return None


def compile_hip_kernel(
    hip_file: str,
    output_so: str,
    arch: str = "gfx942"
) -> bool:
    """
    Compile a .hip source file into a shared library using hipcc.

    Uses: hipcc -O3 -fPIC -shared --offload-arch={arch} {hip_file} -o {output_so}
    # NOTE: -O3 is critical — HIP kernels without optimization are slower than Triton.

    Args:
        hip_file: Path to the .hip source file.
        output_so: Path where the compiled .so should be written.
        arch: GPU architecture target (default gfx942 for MI300X).

    Returns:
        True if compilation succeeded, False otherwise.
    """
    hipcc_path = find_hipcc()
    if hipcc_path is None:
        logger.warning("hipcc not found. Cannot compile HIP kernels.")
        return False

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(output_so) or ".", exist_ok=True)

    # Build compiler command
    cmd = [
        hipcc_path,
        "-O3",
        "-fPIC",
        "-shared",
        f"--offload-arch={arch}",
        hip_file,
        "-o", output_so,
    ]

    # On Windows, hipcc might need additional flags
    if sys.platform == "win32":
        # hipcc on Windows might produce .dll or .so depending on configuration
        cmd.append("-shared")

    logger.info(f"Compiling HIP kernel: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes should be enough
        )
        if result.returncode == 0:
            if os.path.isfile(output_so):
                logger.info(f"HIP kernel compiled successfully: {output_so}")
                return True
            else:
                logger.warning(
                    f"hipcc returned 0 but output file missing: {output_so}"
                )
                return False
        else:
            logger.warning(
                f"hipcc compilation failed (rc={result.returncode}):\n"
                f"STDERR: {result.stderr[:500]}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.warning("hipcc compilation timed out after 120s.")
        return False
    except FileNotFoundError:
        logger.warning(f"hipcc binary not found at '{hipcc_path}'.")
        return False
    except Exception as e:
        logger.warning(f"hipcc compilation error: {e}")
        return False


def load_hip_kernel(so_path: str) -> Optional[ctypes.CDLL]:
    """
    Load a compiled HIP kernel shared library via ctypes.

    Args:
        so_path: Path to the compiled .so file.

    Returns:
        ctypes.CDLL handle on success, None on failure.
    """
    if not os.path.isfile(so_path):
        logger.warning(f"Shared library not found: {so_path}")
        return None

    try:
        lib = ctypes.CDLL(so_path)
        logger.info(f"Loaded HIP kernel library: {so_path}")
        return lib
    except OSError as e:
        logger.warning(f"Failed to load HIP kernel library '{so_path}': {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading '{so_path}': {e}")
        return None


def get_or_compile_int4_kernel() -> Optional[object]:
    """
    Get or compile the INT4/INT8 HIP kernel shared library.

    Checks if compiled .so exists and is newer than the .hip source.
    If stale or missing, recompiles via compile_hip_kernel().
    Loads and returns the library, or returns None if compilation fails.

    # NOTE: returns None silently — caller falls back to Triton.

    Returns:
        ctypes.CDLL handle for the compiled kernel, or None if unavailable.
    """
    # Paths
    package_dir = os.path.dirname(os.path.abspath(__file__))
    hip_file = os.path.join(package_dir, "int4_matmul.hip")
    cache_so = os.path.join(_CACHE_DIR, "int4_matmul.so")

    if not os.path.isfile(hip_file):
        logger.warning(f"HIP source not found: {hip_file}")
        return None

    # Check if cached .so exists and is up-to-date
    need_compile = True
    if os.path.isfile(cache_so):
        hip_mtime = os.path.getmtime(hip_file)
        so_mtime = os.path.getmtime(cache_so)
        if so_mtime >= hip_mtime:
            need_compile = False
            logger.info(f"Using cached HIP kernel: {cache_so}")

    if need_compile:
        logger.info(f"Compiling HIP kernel (source newer than cache)...")
        success = compile_hip_kernel(hip_file, cache_so, arch="gfx942")
        if not success:
            # Try gfx90a (CDNA2) as fallback for older MI200 hardware
            logger.info("Trying gfx90a target as fallback...")
            success = compile_hip_kernel(hip_file, cache_so, arch="gfx90a")
        if not success:
            logger.warning(
                "HIP kernel compilation failed. Falling back to Triton kernels."
            )
            return None

    # Load the compiled library
    lib = load_hip_kernel(cache_so)
    if lib is not None:
        # Configure ctypes argument/return types for the launch functions
        # launch_int4_dequant_matmul(stream, x, w_packed, scales, c, M, N, K, block_size)
        lib.launch_int4_dequant_matmul.argtypes = [
            ctypes.c_void_p,  # hipStream_t
            ctypes.c_void_p,  # x (fp16*)
            ctypes.c_void_p,  # w_packed (uint8*)
            ctypes.c_void_p,  # scales (fp16*)
            ctypes.c_void_p,  # c (fp16*)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # block_size
        ]
        lib.launch_int4_dequant_matmul.restype = None

        # launch_int8_dequant_matmul(stream, x, w_int8, scales, c, M, N, K, block_size)
        lib.launch_int8_dequant_matmul.argtypes = [
            ctypes.c_void_p,  # hipStream_t
            ctypes.c_void_p,  # x (fp16*)
            ctypes.c_void_p,  # w_int8 (int8*)
            ctypes.c_void_p,  # scales (fp16*)
            ctypes.c_void_p,  # c (fp16*)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # block_size
        ]
        lib.launch_int8_dequant_matmul.restype = None

        logger.info("HIP kernel library loaded and configured.")

    return lib


# -- Module-level lazy cache --
# Compiled kernel library, loaded once at first request (import-safe).
_kernel_lib_cache: Optional[object] = None
# Sentinel: None = not tried yet, True = loaded, False = tried and failed
_kernel_lib_state: Optional[bool] = None  # start as "not tried"


def _ensure_kernel_lib() -> Optional[object]:
    """
    Internal: ensure the HIP kernel library is loaded (or cached as failed).

    Uses module-level sentinel to avoid repeated compilation attempts.
    """
    global _kernel_lib_cache, _kernel_lib_state

    # Already tried (True = loaded, False = failed)
    if _kernel_lib_state is not None:
        return _kernel_lib_cache

    # First attempt
    _kernel_lib_cache = get_or_compile_int4_kernel()
    _kernel_lib_state = _kernel_lib_cache is not None  # True if loaded, False if failed
    return _kernel_lib_cache


def is_hipcc_available() -> bool:
    """
    Check whether hipcc is available on this system.
    Does NOT attempt compilation — just path lookup.
    """
    return find_hipcc() is not None