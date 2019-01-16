#! /usr/bin/env python3

from subprocess import run, TimeoutExpired
from argparse   import ArgumentParser

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Test():

    args = ParseCmdLine()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Embedding dimension (E) vs prediction skill (ρ) ---------------" )
    if args.embedded :
        s = './EmbedDimension.py -e -i Embed10_TentMap_rEDM.csv ' +\
            '-l 1 100 -p 201 500 -T 1 -P'
    else:
        s = './EmbedDimension.py -i TentMap_rEDM.csv -c TentMap ' +\
            '-l 1 100 -p 201 500 -T 1 -P'
    try:
        run( s.split(), timeout = args.timeout )
    except TimeoutExpired:
        pass
    print()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Forecast interval (Tp) vs prediction skill (ρ) ----------------" )
    if args.embedded :
        s = './PredictDecay.py -e -i Embed10_TentMap_rEDM.csv ' +\
            '-l 1 100 -p 201 500 -E 2 -P'
    else:
        s = './PredictDecay.py -i TentMap_rEDM.csv -c TentMap ' +\
            '-l 1 100 -p 201 500 -E 2 -P'
    try:
        run( s.split(), timeout = args.timeout )
    except TimeoutExpired:
        pass
    print()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    if False:  # Not run
        print("SMap localization (θ) vs prediction skill (ρ) -----------------")
        if args.embedded :
            s = './SMapNL.py -e -i Embed10_TentMap_rEDM.csv ' +\
                '-l 1 100 -p 201 500 -T 1 -E 2 -P'
        else:
            s = './SMapNL.py -i TentMap_rEDM.csv -c TentMap ' +\
                '-l 1 100 -p 201 500 -T 1 -E 2 -P'
        try:
            run( s.split(), timeout = args.timeout )
        except TimeoutExpired:
            pass
        print()
    
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "SMap localization (θ) vs prediction skill (ρ) with noise ------" )
    if args.embedded :
        s = './SMapNL.py -e -i Embed10_TentMapErr_rEDM.csv ' +\
            '-l 1 100 -p 201 500 -T 1 -E 2 -P'        
    else:
        s = './SMapNL.py -i TentMapErr_rEDM.csv -c TentMap ' +\
            '-l 1 100 -p 201 500 -T 1 -E 2 -P'
    try:
        run( s.split(), timeout = args.timeout )
    except TimeoutExpired:
        pass
    print()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Multivariable SMap prediction ---------------------------------" )
    # Note Multivariable SMap should always use -e (embedded) and -c (columns)
    # to ensure that library and prediction matrix columns correspond to the
    # E dimensions used in the linear decomposition and projection.
    # Note that one can limit the k_NN to allow improved prediction dynamic
    # range and accuracy. 
    s = './Predict.py -e -i block_3sp.csv -m smap -r x_t -c x_t y_t z_t '+\
        '-l 1 99 -p 100 198 -T 1 -t 2 -P'
    try:
        run( s.split(), timeout = args.timeout )
    except TimeoutExpired:
        pass
    print()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Multivariable SMap prediction ---------------------------------" )
    # Note Multivariable SMap should always use -e (embedded) and -c (columns)
    # to ensure that library and prediction matrix columns correspond to the
    # E dimensions used in the linear decomposition and projection.
    # This example uses V1 embedded to E = 10
    s = './Predict.py -e -i Embed10_V1_Lorenz.csv -m smap ' +\
        '-E 10 -l 1 300 -p 301 990 -T 1 -t 3 -k 200 -P'
    # This example uses an explicit multivariable column:dimension mapping
    s = './Predict.py -e -i LorenzData1000.csv -m smap ' +\
        '-r V1 -c V1 V2 V3 V4 V5 -l 1 300 -p 301 990 -T 1 -t 3 -k 200 -P'
    try:
        run( s.split(), timeout = args.timeout )
    except TimeoutExpired:
        pass
    print()

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Multiview ensemble simplex prediction -------------------------" )
    if args.embedded :
        print( "Multiview requires dynamic run-time embedding." )
    else:
        s = './Multiview.py -i block_3sp.csv -E 3 -r x_t -c x_t y_t z_t ' +\
            '-l 1 100 -p 101 200 -T 1 -P'
        try:
            run( s.split(), timeout = args.timeout )
        except TimeoutExpired:
            pass
        print()
        

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    print( "Convergent cross mapping --------------------------------------" )
    if args.embedded :
        print( "CCM requires dynamic run-time embedding." )
    else:
        # These values correspond to the rEDM test
        s = './CCM.py -i sardine_anchovy_sst.csv -c anchovy -r np_sst ' +\
            '-E 3 -s 100 -L 10 80 10 -R -P'
        try:
            run( s.split(), timeout = args.timeout  )
        except TimeoutExpired:
            pass
        print()
        

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'Test' )
    
    parser.add_argument('-e', '--embedded',
                        dest   = 'embedded', 
                        action = 'store_true', default = False,
                        help = 'Data input is an embbedding.')
    
    parser.add_argument('-t', '--timeout',
                        dest   = 'timeout', type = int,
                        action = 'store', default = 10,
                        help = 'Subprprocess timeout in seconds.')
    
    args = parser.parse_args()

    return args
    
#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    Test()
