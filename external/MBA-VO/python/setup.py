from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path

__version__ = '0.1.2'

script_dir = os.path.dirname(os.path.abspath(__file__))
third_party_dir = os.path.normpath(os.path.join(script_dir, "..", "third_party"))

ext_modules=[
    CUDAExtension(
        name='pymbavo_lib',
        sources=[
            'pymbavo_lib/ba_tracker.cc',
            'pymbavo_lib/factory.cc',
            'pymbavo_lib/python_bind.cc'
        ],
        extra_compile_args={'cxx': ['-O2', '-std=c++20', '-DCOMPILE_WITH_CUDA'],
                            'nvcc': ['-O2', '-std=c++20']},
        extra_link_args=['-L/usr/local/lib',
                         '-lslam_core',
                         '-lslam_utils',
                         '-lslam_blur_tracker'],
        include_dirs=[
            "/usr/include/eigen3/",
            "/usr/include/opencv4/",
            "/usr/local/include/opencv4/",
            os.path.join(third_party_dir, "Sophus"),
            os.path.join(third_party_dir, "fmt", "include")]
    ),
]

INSTALL_REQUIREMENTS = ['torch', 'eigen', 'kornia', 'lightglue']

if __name__ == "__main__":
    setup(
        description='pymbavo',
        author='Lingzhe Zhao',
        author_email='zhaolingzhe@westlake.edu.cn',
        license='MIT License',
        version=__version__,
        name='pymbavo',
        packages=['pymbavo', 'pymbavo_lib'],
        install_requires=INSTALL_REQUIREMENTS,
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
    )
