#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import EmbedDimensions

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    EmbedDimensions( args, 'Python' )
