#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import Embed, Source

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
if __name__ == "__main__":

    args = ParseCmdLine()
    Embed( args, data = None, colNames = None, source = Source.Python )
