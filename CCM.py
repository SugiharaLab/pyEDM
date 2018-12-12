#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import CCM

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    CCM( args, 'Python' )
