
# python modules

# package modules

# local modules
import pyEDM.API as API
from .AuxFunc import ComputeError

#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )
    return df

#----------------------------------------------------
# Function to evaluate combo rank (rho)
#----------------------------------------------------
def MultiviewSimplexRho( combo, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#----------------------------------------------------
# Function to evaluate Simplex in EmbedDimension Pool
#----------------------------------------------------
def EmbedDimSimplexFunc( E, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = E, 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#-----------------------------------------------------
# Function to evaluate Simplex in PredictInterval Pool
#-----------------------------------------------------
def PredictIntervalSimplexFunc( Tp, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = Tp,
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#-----------------------------------------------------
# Function to evaluate SMap in PredictNonlinear Pool
#-----------------------------------------------------
def PredictNLSMapFunc( theta, data, args ) :

    S = API.SMap( dataFrame       = data,
                  columns         = args['columns'],
                  target          = args['target'], 
                  lib             = args['lib'],
                  pred            = args['pred'],
                  E               = args['E'], 
                  Tp              = args['Tp'],
                  tau             = args['tau'],
                  theta           = theta,
                  exclusionRadius = args['exclusionRadius'],
                  solver          = args['solver'],
                  embedded        = args['embedded'],
                  validLib        = args['validLib'],
                  noTime          = args['noTime'],
                  ignoreNan       = args['ignoreNan'] )

    df = S['predictions']
    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']
