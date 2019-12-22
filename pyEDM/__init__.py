'''Python interface to cppEDM github.com/SugiharaLab/cppEDM'''

import os, sys

dir_path     = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
dependencies = ["libquadmath-0","libgfortran-3",
                "libgcc_s_seh-1","libopenblas"]

# load in dependencies dlls if on win

if sys.platform.startswith('win') :

    from ctypes import WinDLL

    for dependency in dependencies :

        WinDLL( dir_path+"win_64_dependencies"+os.path.sep+dependency+".dll" )

# export all edm functions

from pyEDM.CoreEDM import *
from pyEDM.AuxFunc import *
