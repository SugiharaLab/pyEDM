#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import Multiview

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    Multiview( args, 'Python' )
