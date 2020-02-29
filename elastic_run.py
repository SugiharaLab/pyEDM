from sklearn.linear_model import Ridge, Lasso, ElasticNet
import pyEDM

circle = pyEDM.sampleData['circle']

lmSolvers = {
    'SVD'          : None, 
    'lmRidge'      : Ridge( alpha = 0.05 ),
    'lmLasso'      : Lasso( alpha = 0.005 ),
    'lmElasticNet' : ElasticNet( alpha = 0.001, l1_ratio = 0.001 )
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

