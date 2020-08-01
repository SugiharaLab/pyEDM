#! /usr/bin/env python3

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import pyEDM

#------------------------------------------------------------
#------------------------------------------------------------
def main():
    '''Test sklearn.linear_model solvers'''
    
    circle = pyEDM.sampleData['circle']
    
    lmSolvers = {
        'SVD'          : None, 
        'Ridge'        : Ridge( alpha = 0.05 ),
        'Lasso'        : Lasso( alpha = 0.005 ),
        'ElasticNet'   : ElasticNet( alpha = 0.001, l1_ratio = 0.001 ),
        'RidgeCV'      : RidgeCV(),
        'LassoCV'      : LassoCV( cv = 5 ),
        'ElasticNetCV' : ElasticNetCV( l1_ratio = [.05,.1,.5,.7,.9,.95,1],
                                       cv = 5 )
    }
    
    smapResults = {}
    
    for solverName in lmSolvers.keys() :
        print( solverName )
        result = pyEDM.SMap( dataFrame = circle,
                             lib = "1 100", pred = "101 198",
                             embedded = True, E = 2, theta = 3.14,
                             columns = "x y", target = "x", showPlot = True,
                             solver = lmSolvers[ solverName ] )
        smapResults[ solverName ] =  result

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
