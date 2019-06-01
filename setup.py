#
# Build the EDM Python binding to cppEDM.
#
# The core library is the C++ cppEDM libEDM.a
# libEDM.a needs to be built with -fPIC
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
# Most of this setup is cloned from pybind11 example setup.py
# https://github.com/pybind/python_example
#
# This only works if the directory structure for the module to
# be installed (EDM) is:
#
# pyEDM/  # Any name
#   setup.py
#   EDM/
#     __init__.py # import ./EDM
#     EDM.py
#     data/
#     tests/
#
# Getting this to work reasonably is just mind-numbing and insensible. 
# There seem to be many implicit requirements such as the directory 
# stucture, name conventions, apparent incongruitues between using
# package_data, MANIFEST.in, etc...
#
import sys, os
import setuptools
from   setuptools import setup, Extension
from   setuptools.command.build_ext import build_ext

__version__ = '0.1.21'  # Get version from cppEDM Parameter.cc ?

# e.g. /tmp/pip-req-build-9ljrp27z/
tmpInstallPath = os.path.dirname( os.path.abspath( __file__ ) )
EDM_Lib_Path   = os.path.join( tmpInstallPath, "lib" )
EDM_H_Path     = os.path.join( tmpInstallPath, "src/cppEDM/" )

# Transfer the README.md to the package decsription
with open(os.path.join(tmpInstallPath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
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
# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
#----------------------------------------------------------------------
def has_flag( compiler, flagname ):
    """Return a boolean indicating whether a flag name is supported on
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
    The c++14 is prefered over c++11 (when it is available)."""
    
    if has_flag( compiler, '-std=c++14' ):
        return '-std=c++14'
    elif has_flag( compiler, '-std=c++11' ):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
class BuildExt( build_ext ):
    """A custom build extension for adding compiler-specific options."""
    
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
Extension_modules = [
    Extension(
        name = 'EDM_pybind',

        sources = [ 'src/bindings/PyBind.cpp' ],
        
        include_dirs = [
            get_pybind_include(), # Path to pybind11 headers
            get_pybind_include( user = True ),
            EDM_H_Path # Path to cppEDM headers
        ],
        
        language     = 'c++',
        library_dirs = [ EDM_Lib_Path ], # cppEDM libEDM.a -fPIC
        libraries    = ['EDM'],          # cppEDM libEDM.a -fPIC
    ),
]

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------
setup(
    # name of the *-version.dist-info directory
    name             = 'EDM-SugiharaLab', 
    version          = __version__,
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
    package_data     = { 'EDM' : ['data/*.csv', 'tests/*.py'] },
    #test_suite      = "tests", # ??? [1]
    install_requires = ['pybind11>=2.2', 'pandas>=0.20.3', 'matplotlib>=2.2'],
    python_requires  = '>=3',
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
