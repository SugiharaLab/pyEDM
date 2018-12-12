#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import Embed

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
if __name__ == "__main__":

    args = ParseCmdLine()
    Embed( args, 'Python' )
