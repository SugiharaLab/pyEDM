#--------------------------------------------------------------------------
# Build the pyEDM Python binding to cppEDM.
#
# The core/extension library is the C++ cppEDM libEDM.a
# NOTE: libEDM.a needs to be built with -fPIC. 
#
# NOTE: win32 builds are of course, problematic.
#       ****** Force the use of mingw32. Do not use msvc ******
#
# Bindings to cppEDM are provided via pybind11 in src/bindings/PyBind.cpp
# and are built as an Extension module into a platform-specific shared
# library (EDM_pybind) as defined in Extension_modules(). EDM_pybind
# bindings return Python dictionaries analogous to the cppEDM and Pandas
# DataFrame (or nested dictionaries in the case of SMap & Multiview). 
#
# The EDM_pybind module is wrapped with the EDM module to convert the
# Python dictionaries into Pandas DataFrames and provide rudimentary
# plotting via matplotlib.pyplot.  
#
# Some of this setup is cloned from pybind11 example setup.py
# https://github.com/pybind/python_example
#--------------------------------------------------------------------------

DEBUG_BUILD_pyEDM = False

import sys
import os
import re
import subprocess
import setuptools
from   setuptools import setup, Extension
from   setuptools.command.build_ext import build_ext

# Clear msvc runtime dll for windog build
# CygwinCCompiler class is a subclass of UnixCCompiler that handles
# the Cygwin port of the GNU C compiler to Windows. It also contains
# the Mingw32CCompiler class which handles the mingw32 port of GCC
import distutils.cygwinccompiler
distutils.cygwinccompiler.get_msvcr = lambda:[]

# Package paths e.g. /tmp/pip-req-build-9ljrp27z/
tmpInstallPath = os.path.dirname( os.path.abspath( __file__ ) )
EDM_Lib_Path   = os.path.join( tmpInstallPath, "cppEDM/lib" )
EDM_H_Path     = os.path.join( tmpInstallPath, "cppEDM/src" )
Bindings_Path  = os.path.join( tmpInstallPath, "src/bindings/" )

# Build libEDM.a 
cppLibName = "libEDM.a"

if not os.path.exists(EDM_Lib_Path): # in case of sdist build, mkdir lib
    os.makedirs(EDM_Lib_Path)

build_libEDM = subprocess.Popen(["make", "-C", "./cppEDM/src"], 
                                stderr=subprocess.STDOUT)
build_libEDM.wait()

# Check that libEDM exists
if not os.path.isfile( os.path.join( EDM_Lib_Path, cppLibName ) ) :
    errStr = "Error: " + os.path.join( "./cppEDM/src/lib", cppLibName ) +\
             " does not exist.\n\nYou can install cppEDM manually per: " +\
             "https://github.com/SugiharaLab/pyEDM#manual-install."
    raise Exception( errStr )

# Transfer the README.md to the package decsription
with open(os.path.join(tmpInstallPath, 'README.md')) as f:
    long_description = f.read()

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
def read_version(*file_paths):
    '''Read __init__.py to get __version__'''
    with open(os.path.join( tmpInstallPath, *file_paths ), 'r') as fp:
        version_file =  fp.read()

    # Why don't we just use re syntax as unbreakable crypto-keys?
    version_re = r'^__version__ = "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"$'
    
    version_match = re.search( version_re, version_file, re.MULTILINE )
    if version_match:
        # version_match.group() is: '__version__ = "1.0.0.0"'
        # isolate just the numeric part between " "
        version = version_match.group().split('"')[1]
        return version
    raise RuntimeError("find_version(): Unable to find version string.")

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
class get_pybind_include( object ):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

#----------------------------------------------------------------------
# 
#----------------------------------------------------------------------
def has_flag( compiler, flagname ):
    """Return a boolean indicating whether a flag is supported on
    the specified compiler."""
    
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
def cpp_flag( compiler ):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11."""
    
    if has_flag( compiler, '-std=c++17' ):
        return '-std=c++17'
    elif has_flag( compiler, '-std=c++14' ):
        return '-std=c++14'
    elif has_flag( compiler, '-std=c++11' ):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler: '
                           'C++11 minimum standard required.')

#----------------------------------------------------------------------
# Note:
# >>> distutils.ccompiler.show_compilers()
# List of available compilers:
#   --compiler=mingw32  Mingw32 port of GNU C Compiler for Win32
#   --compiler=msvc     Microsoft Visual C++
#   --compiler=unix     standard UNIX-style compiler
#----------------------------------------------------------------------
class BuildExt( build_ext ):
    
    c_opts = {
        'msvc'    : [],
        'unix'    : ['-llapack'],
        'mingw32' : ['-DMS_WIN64']
    }

    if DEBUG_BUILD_pyEDM :
        print( ">>>>>>>>>>> sys.platform: ", sys.platform )

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct   = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        if DEBUG_BUILD_pyEDM :
            print( '>>>>>>>>>>> compiler_type: ', self.compiler.compiler_type )
        
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append( cpp_flag(self.compiler) )
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args = opts

        build_ext.build_extensions(self)

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
Extension_modules = [
    Extension(
        name = 'pyBindEDM',

        sources = [ Bindings_Path + 'PyBind.cpp' ],
        
        include_dirs = [
            get_pybind_include(), # Path to pybind11 headers
            get_pybind_include( user = True ),
            EDM_H_Path            # Path to cppEDM headers
        ],

        language           = 'c++',
        extra_compile_args = ['-std=c++11'],
        library_dirs       = [ EDM_Lib_Path, '/usr/lib/' ],
        
        # Note PEP 308: <expression1> if <condition> else <expression2>
        libraries = ['EDM','openblas','gfortran','pthread','m','quadmath'] \
                    if sys.platform.startswith('win') else ['EDM','lapack'],
        
        extra_link_args = ["-static", "-static-libgfortran", "-static-libgcc"] \
                          if sys.platform.startswith('win') else [],
    ),
]

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
setup(
    name             = 'pyEDM', # name of *-version.dist-info directory
    version          = read_version( 'pyEDM', '__init__.py' ),
    author           = 'Joseph Park & Cameron Smith',
    author_email     = 'Sugihara.Lab@gmail.com',
    url              = 'https://github.com/SugiharaLab/pyEDM',
    description      = 'Python wrapper for cppEDM using pybind11',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license          = 'Copyright 2019 The Regents of the University ' +\
                       'of California.',
    packages         = setuptools.find_packages(), # Enable ./EDM Python module
    ext_modules      = Extension_modules,
    package_data     = { 'pyEDM' : ['data/*.csv', 'tests/*.py' ]},
    #test_suite      = "tests", # ??? [1]
    install_requires = ['pybind11>=2.3', 'pandas>=1.1', 'matplotlib>=2.2'],
    python_requires  = '>=3.6',
    cmdclass         = { 'build_ext' : BuildExt }, # Command/class to build .so
    zip_safe         = False,
)
#----------------------------------------------------------------------
# [1] This test_suite doesn't seem terribly useful here in that its use
#     seems to be to enable "python setup.py test" as a way to test
#     functionality prior to deployment, or perhaps from a source
#     distribution (sdist command) build/test. See:
#     https://setuptools.readthedocs.io/en/latest/setuptools.html
#           #test-build-package-and-run-a-unittest-suite
#
#     One can run the tests in EDM/tests: python -m unittest discover
#----------------------------------------------------------------------
