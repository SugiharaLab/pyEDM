#! /usr/bin/env python3

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import SGDRegressor, Lars, LassoLars
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

import pyEDM as EDM

#------------------------------------------------------------
#------------------------------------------------------------
def main():
    '''Test sklearn.linear_model solvers'''

    circle = EDM.sampleData['circle']

    lmSolvers = {
        'SVD'              : None,
        'LinearRegression' : LinearRegression(),
        'SGDRegressor'     : SGDRegressor( alpha = 0.005 ),
        'Ridge'            : Ridge( alpha = 0.05 ),
        'Lasso'            : Lasso( alpha = 0.005 ),
        'Lars'             : Lars(),
        'LassoLars'        : LassoLars( alpha = 0.005 ),        
        'ElasticNet'       : ElasticNet( alpha = 0.001, l1_ratio = 0.001 ),
        'RidgeCV'          : RidgeCV(),
        'LassoCV'          : LassoCV( cv = 5 ),
        'ElasticNetCV'     : ElasticNetCV(l1_ratio = [.05,0.2,.5,.9,1], cv = 5)
    }

    smapResults = {}

    for solverName in lmSolvers.keys() :
        print( solverName )
        result = EDM.SMap( dataFrame = circle, columns = "x y", target = "x",
                           lib = [1, 100], pred = [101, 198],
                           embedded = True, E = 2, theta = 3.14,
                           solver = lmSolvers[ solverName ], showPlot = True )
        smapResults[ solverName ] =  result

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
