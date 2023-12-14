
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ('debug', None, 'Specify debug build')
    ]

    def initialize_options(self):
        super().initialize_options()
        self.debug = None

    def finalize_options(self):
        super().finalize_options()
        if self.debug is not None:
            self.debug = bool(self.debug)

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'

        # specify the build output directory
        build_temp = os.path.join(extdir, f"build_ext-{ext.name}")

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=' + cfg
        ]

        build_args = ['--config', cfg]

        os.makedirs(build_temp, exist_ok=True)
        os.chdir(build_temp)
        self.spawn(['cmake', ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)


setup(
    name='bliss',
    version='0.0.1',
    packages=['bliss'],
    ext_modules=[CMakeExtension('bliss/bliss')],
    package_data={
        'bliss': ['bliss/*.so'],  # use wildcard to match any .so file
    },
    py_modules=['bliss.pybliss'],
    cmdclass=dict(build_ext=CMakeBuild),
)