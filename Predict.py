#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import Predict, Source

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    Predict( args, data = None, colNames = None, target = None,
             source = Source.Python )
