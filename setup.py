from pathlib import Path
import os
import subprocess
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = ROOT_DIR / "python"
CPP_DIR = ROOT_DIR / "src" / "cpp"
CUDA_DIR = ROOT_DIR / "src" / "cuda"
THIRD_PARTY_DIR = ROOT_DIR / "src" / "third_party"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"The {name} environment variable is not set.")
    return value


eigen_dir = require_env("EIGEN_DIR")
boost_dir = require_env("BOOST_DIR")

if sys.platform.startswith("win"):
    extra_compile_args = ["/std:c++17"]
    cuda_path = os.environ.get(
        "CUDA_PATH",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
    )
    lib_dir = "lib/x64"
    kernel_suffix = ".obj"
else:
    extra_compile_args = ["-std=c++17", "-O3", "-ffast-math", "-march=native"]
    cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    lib_dir = "lib64"
    kernel_suffix = ".o"


class CustomBuildExt(build_ext):
    def _get_inplace_equivalent(self, build_py, ext):
        inplace_file, regular_file = super()._get_inplace_equivalent(build_py, ext)
        return str(PYTHON_DIR / Path(inplace_file).name), regular_file

    def build_extensions(self):
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        kernel_cu = CUDA_DIR / "distance_kernel.cu"
        kernel_obj = build_temp / f"distance_kernel{kernel_suffix}"

        if sys.platform.startswith("win"):
            nvcc_command = [
                "nvcc",
                "-c",
                str(kernel_cu),
                "-o",
                str(kernel_obj),
                "--compiler-options",
                "/MD",
                "-arch=sm_60",
            ]
        else:
            nvcc_command = [
                "nvcc",
                "-c",
                str(kernel_cu),
                "-o",
                str(kernel_obj),
                "-Xcompiler",
                "-fPIC",
                "-arch=sm_60",
            ]

        print(f"Running command: {' '.join(nvcc_command)}")
        subprocess.check_call(nvcc_command)

        for extension in self.extensions:
            extension.extra_objects = list(extension.extra_objects) + [str(kernel_obj)]

        super().build_extensions()


ext_modules = [
    Extension(
        "ActSeekLib",
        sources=[str(CPP_DIR / "ActSeekLib.cpp")],
        include_dirs=[
            pybind11.get_include(),
            str(Path(cuda_path) / "include"),
            eigen_dir,
            boost_dir,
            str(CPP_DIR),
            str(THIRD_PARTY_DIR),
        ],
        library_dirs=[str(Path(cuda_path) / lib_dir)],
        libraries=["cudart_static"],
        extra_compile_args=extra_compile_args,
        extra_objects=[],
    ),
]


setup(
    name="ActSeekLib",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
