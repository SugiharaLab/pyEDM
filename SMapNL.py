#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import SMapNL, Source

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    
    SMapNL( args, data = None, colNames = None, target = None, thetas = None,
            source = Source.Python )
