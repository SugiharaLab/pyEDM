#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import Predict

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    Predict( args, 'Python' )
