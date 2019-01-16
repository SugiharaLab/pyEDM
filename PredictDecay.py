#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from Methods  import PredictDecays

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = ParseCmdLine()
    PredictDecays( args, 'Python' )
