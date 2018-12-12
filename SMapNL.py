#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import SMapNL

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    SMapNL( args, 'Python' )
