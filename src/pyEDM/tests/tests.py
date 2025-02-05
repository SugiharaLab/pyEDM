
import sys
import unittest
from   datetime import datetime

from numpy  import nan
from pandas import read_csv
import pyEDM as EDM

#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_EDM( unittest.TestCase ):
    '''The examples.py and smapSolverTest.py must also run.

    NOTE: Bizarre default of unittest class presumes
          methods names to be run begin with "test_" 
    '''
    # JP How to pass command line arg to class? verbose = True
    def __init__( self, *args, **kwargs):
        super( test_EDM, self ).__init__( *args, **kwargs )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def setUpClass( self ):
        self.verbose = False
        self.GetValidFiles( self )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    def GetValidFiles( self ):
        '''Create dictionary of DataFrame values from file name keys'''
        self.ValidFiles = {}

        validFiles = [ 'CCM_anch_sst_valid.csv',
                       'CCM_Lorenz5D_MV_Space_valid.csv',
                       'CCM_Lorenz5D_MV_valid.csv',
                       'CCM_nan_valid.csv',
                       'EmbedDim_valid.csv',
                       'Multiview_combos_valid.csv',
                       'Multiview_pred_valid.csv',
                       'PredictInterval_valid.csv',
                       'PredictNonlinear_valid.csv',
                       'SMap_circle_E2_embd_valid.csv',
                       'SMap_circle_E4_valid.csv',
                       'SMap_nan_valid.csv',
                       'SMap_noTime_valid.csv',
                       'Smplx_DateTime_valid.csv',
                       'Smplx_disjointLib_valid.csv',
                       'Smplx_disjointPred_nan_valid.csv',
                       'Smplx_E3_block_3sp_valid.csv',
                       'Smplx_E3_embd_block_3sp_valid.csv',
                       'Smplx_exclRadius_valid.csv',
                       'Smplx_nan2_valid.csv',
                       'Smplx_nan_valid.csv',
                       'Smplx_negTp_block_3sp_valid.csv',
                       'Smplx_validLib_valid.csv' ]

        # Create map of module validFiles pathnames in validFiles
        for file in validFiles:
            filename = "validation/" + file
            self.ValidFiles[ file ] = read_csv( filename )

    #------------------------------------------------------------
    # API
    #------------------------------------------------------------
    def test_API_1( self ):
        '''API 1'''
        if self.verbose : print ( " --- API 1 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_, columns = 'V1', target = 'V5',
                           lib = '1 300', pred = '301 310', E = 5 )

    def test_API_2( self ):
        '''API 2'''
        if self.verbose : print ( "--- API 2 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_, columns = ['V1'], target = 'V5',
                           lib = [1, 300], pred = [301, 310], E = 5 )

    def test_API_3( self ):
        '''API 3'''
        if self.verbose : print ( "--- API 3 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_, columns = ['V1','V3'], target = 'V5',
                           lib = [1, 300], pred = [301, 310], E = 5 )

    def test_API_4( self ):
        '''API 4'''
        if self.verbose : print ( "--- API 4 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_,
                           columns = ['V1','V3'], target = ['V5','V2'], 
                           lib = [1, 300], pred = [301, 310], E = 5 )

    def test_API_5( self ):
        '''API 5'''
        if self.verbose : print ( "--- API 5 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_, columns = 'V1', target = 'V5',
                           lib = [1, 300], pred = [301, 310], E = 5, knn = 0 )

    def test_API_6( self ):
        '''API 6'''
        if self.verbose : print ( "--- API 6 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.Simplex( dataFrame = df_, columns = 'V1', target = 'V5',
                           lib = [1, 300], pred = [301, 310], E = 5, tau = -2 )

    def test_API_7( self ):
        '''API 7'''
        if self.verbose : print ( "--- API 7 Column names with space ---" )
        df_ = EDM.sampleData["columnNameSpace"]
        df = EDM.Simplex( df_, ['Var 1','Var 2'], ["Var 5 1"],
                          [1, 80], [81, 85], 5, 1, 0, -1, 0,
                          False, [], False, 0, False, False, False )

    #------------------------------------------------------------
    # Embed
    #------------------------------------------------------------
    def test_embed( self ):
        '''Embed'''
        if self.verbose : print ( "--- Embed ---" )
        df_ = EDM.sampleData['circle']
        df  = EDM.Embed( df_, 3, -1, "x", False )

    def test_embed2( self ):
        '''Embed multivariate'''
        if self.verbose : print ( "--- Embed multivariate ---" )
        df_ = EDM.sampleData['circle']
        df  = EDM.Embed( df_, 3, -1, ['x', 'y'], False )

    def test_embed2( self ):
        '''Embed multivariate'''
        if self.verbose : print ( "--- Embed includeTime ---" )
        df_ = EDM.sampleData['circle']
        df  = EDM.Embed( df_, 3, -1, ['x', 'y'], True )

    def test_embed3( self ):
        '''Embed from file'''
        if self.verbose : print ( "--- Embed from file ---" )
        df  = EDM.Embed( pathIn = '../data/', dataFile = 'circle.csv',
                         E = 3, tau = -1, columns = ['x', 'y'] )

    #------------------------------------------------------------
    # Simplex
    #------------------------------------------------------------
    def test_simplex( self ):
        '''embedded = False'''
        if self.verbose : print ( "--- Simplex embedded = False ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.Simplex( df_, "x_t", "x_t",
                          [1, 100], [101, 195], 3, 1, 0, -1, 0,
                          False, [], False, 0, False, False, False )

        dfv = self.ValidFiles["Smplx_E3_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:95], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:95], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex2( self ):
        '''embedded = True'''
        if self.verbose : print ( "--- Simplex embedded = True ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.Simplex( df_, "x_t y_t z_t", "x_t",
                          [1, 99], [100, 198], 3, 1, 0, -1, 0,
                          True, [], False, 0, False, False, False )

        dfv = self.ValidFiles["Smplx_E3_embd_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:98], 6 )      # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex3( self ):
        '''negative Tp'''
        if self.verbose : print ( "--- negative Tp ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.Simplex( df_, "x_t", "y_t",
                          [1, 100], [50, 80], 3, -2, 0, -1, 0,
                          False, [], False, 0, False, False, False )

        dfv = self.ValidFiles["Smplx_negTp_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex4( self ):
        '''validLib'''
        if self.verbose : print ( "--- validLib ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.Simplex( dataFrame = df_, columns = 'x', target = 'x',
                          lib = [1,200], pred = [1,200], E = 2, Tp = 1,
                          validLib = df_.eval('x > 0.5 | x < -0.5') )

        dfv = self.ValidFiles["Smplx_validLib_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex5( self ):
        '''disjoint lib'''
        if self.verbose : print ( "--- disjoint lib ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.Simplex( dataFrame = df_, columns = 'x', target = 'x',
                          lib = [1,40, 50,130], pred = [80,170],
                          E = 2, Tp = 1, tau = -3 )

        dfv = self.ValidFiles["Smplx_disjointLib_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex6( self ):
        '''disjoint pred w/ nan'''
        if self.verbose : print ( "--- disjoint pred w/ nan ---" )
        df_ = EDM.sampleData["Lorenz5D"]
        df_.iloc[ [8,50,501], [1,2] ] = nan

        df = EDM.Simplex( dataFrame = df_, columns='V1', target = 'V2',
                          E = 5, Tp = 2, lib = [1,50,101,200,251,500],
                          pred = [1,10,151,155,551,555,881,885,991,1000] )

        dfv = self.ValidFiles["Smplx_disjointPred_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 5 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 5 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex7( self ):
        '''exclusion radius'''
        if self.verbose : print ( "--- exclusion radius ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.Simplex( dataFrame = df_, columns = 'x', target = 'y',
                          lib = [1,100], pred = [21,81], E = 2, Tp = 1,
                          exclusionRadius = 5 )

        dfv = self.ValidFiles["Smplx_exclRadius_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:60], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:60], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex8( self ):
        '''nan'''
        if self.verbose : print ( "--- nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        df = EDM.Simplex( dataFrame = dfn, columns = 'x', target = 'y',
                          lib = [1,100], pred = [1,95], E = 2, Tp = 1 )

        dfv = self.ValidFiles["Smplx_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:90], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:90], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex9( self ):
        '''nan'''
        if self.verbose : print ( "--- nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        df = EDM.Simplex( dataFrame = dfn, columns = 'y', target = 'x',
                          lib = [1,200], pred = [1,195], E = 2, Tp = 1 )

        dfv = self.ValidFiles["Smplx_nan2_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:190], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:190], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex10( self ):
        '''DateTime'''
        if self.verbose : print ( "--- DateTime ---" )
        df_ = EDM.sampleData["SumFlow_1980-2005"]

        df = EDM.Simplex( dataFrame = df_,
                          columns = 'S12.C.D.S333', target = 'S12.C.D.S333',
                          lib = [1,800], pred = [801,1001], E = 3, Tp = 1 )

        self.assertTrue( isinstance( df['Time'][0],  datetime ) )

        dfv = self.ValidFiles["Smplx_DateTime_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:200], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:200], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # S-map
    #------------------------------------------------------------
    def test_smap( self ):
        '''SMap'''
        if self.verbose : print ( "--- SMap ---" )
        df_ = EDM.sampleData["circle"]
        S = EDM.SMap( dataFrame = df_, columns = 'x', target = 'x',
                      lib = [1,100], pred = [110,160], E = 4, Tp = 1,
                      tau = -1, theta = 3. )

        dfv = self.ValidFiles["SMap_circle_E4_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_smap2( self ):
        '''SMap embedded = True'''
        if self.verbose : print ( "--- SMap embedded = True ---" )
        df_ = EDM.sampleData["circle"]
        S = EDM.SMap( dataFrame = df_, columns = ['x', 'y'], target = 'x',
                      lib = [1,200], pred = [1,200], E = 2, Tp = 1,
                      tau = -1, embedded = True, theta = 3. )

        dfv  = self.ValidFiles["SMap_circle_E2_embd_valid.csv"]
        
        df  = S['predictions']
        dfc = S['coefficients']

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

        self.assertTrue( dfc['∂x/∂x'].mean().round(5) == 0.99801 )
        self.assertTrue( dfc['∂x/∂y'].mean().round(5) == 0.06311 )

    #------------------------------------------------------------
    def test_smap3( self ):
        '''SMap nan'''
        if self.verbose : print ( "--- SMap nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        S = EDM.SMap( dataFrame = dfn, columns = 'x', target = 'y',
                      lib = [1,50], pred = [1,50], E = 2, Tp = 1,
                      tau = -1, theta = 3. )

        dfv = self.ValidFiles["SMap_nan_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_smap4( self ):
        '''DateTime'''
        if self.verbose : print ( "--- noTime ---" )
        df_ = EDM.sampleData["circle_noTime"]

        S = EDM.SMap( dataFrame = df_, columns = 'x', target = 'y',
                      lib = [1,100], pred = [101,150], E = 2,
                      theta = 3, noTime = True )

        dfv = self.ValidFiles["SMap_noTime_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # CCM
    #------------------------------------------------------------
    def test_ccm( self ):
        if self.verbose : print ( "--- CCM ---" )
        df_ = EDM.sampleData['sardine_anchovy_sst']
        df = EDM.CCM( dataFrame = df_, columns = 'anchovy', target = 'np_sst',
                      libSizes = [10,20,30,40,50,60,70,75], sample = 100,
                      E = 3, Tp = 0, tau = -1, seed = 777 )

        dfv = round( self.ValidFiles["CCM_anch_sst_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    def test_ccm2( self ):
        '''CCM Multivariate'''
        if self.verbose : print ( "--- CCM multivariate ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df = EDM.CCM( dataFrame = df_, columns = 'V3 V5', target = 'V1',
                      libSizes = [20, 200, 500, 950], sample = 30, E = 5,
                      Tp = 10, tau = -5, seed = 777 )

        dfv = round( self.ValidFiles["CCM_Lorenz5D_MV_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    def test_ccm3( self ):
        '''CCM nan'''
        if self.verbose : print ( "--- CCM nan ---" )
        df_ = EDM.sampleData['circle']
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        df = EDM.CCM( dataFrame = dfn, columns = 'x', target = 'y',
                      libSizes = [10,190,10], sample = 20, E = 2,
                      Tp = 5, tau = -1, seed = 777 )

        dfv = round( self.ValidFiles["CCM_nan_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    def test_ccm4( self ):
        '''CCM Multivariate names with spaces'''
        if self.verbose : print ( "--- CCM multivariate name spaces ---" )
        df_ = EDM.sampleData['columnNameSpace']
        df = EDM.CCM( dataFrame = df_,
                      columns = ['Var 1','Var3','Var 5 1'],
                      target = ['Var 2','Var 4 A'],
                      libSizes = [20, 50, 90], sample = 1,
                      E = 5, Tp = 0, tau = -1, seed = 777 )

        dfv = round( self.ValidFiles["CCM_Lorenz5D_MV_Space_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    # Multiview
    #------------------------------------------------------------
    def test_multiview( self ):
        if self.verbose : print ( "--- Multiview ---" )
        df_ = EDM.sampleData['block_3sp']
        M = EDM.Multiview( dataFrame = df_,
                           columns = "x_t y_t z_t", target = "x_t",
                           lib = [1, 100], pred = [101, 198],
                           D = 0, E = 3, Tp = 1, knn = 0, tau = -1,
                           multiview = 0, exclusionRadius = 0,
                           trainLib = False, excludeTarget = False,
                           numProcess = 4, showPlot = False )

        df_pred  = M['Predictions']
        df_combo = M['View'][ ['rho', 'MAE', 'RMSE'] ]

        # Validate predictions
        dfvp      = self.ValidFiles["Multiview_pred_valid.csv"]
        predValid = round( dfvp.get('Predictions'), 4 )
        pred      = round( df_pred.get('Predictions'), 4 )
        self.assertTrue( predValid.equals( pred ) )

        # Validate combinations
        dfvc = round( self.ValidFiles['Multiview_combos_valid.csv'], 4 )
        dfvc = dfvc[ ['rho', 'MAE', 'RMSE'] ]

        self.assertTrue( dfvc.equals( round( df_combo, 4 ) ) )

    #------------------------------------------------------------
    # EmbedDimension
    #------------------------------------------------------------
    def test_embedDimension( self ):
        if self.verbose : print ( "--- EmbedDimension ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df = EDM.EmbedDimension( dataFrame = df_, columns = 'V1', target = 'V1',
                                 maxE = 12, lib = [1, 500], pred = [501, 800],
                                 Tp = 15, tau = -5, exclusionRadius = 20,
                                 numProcess = 10, showPlot = False )

        dfv = round( self.ValidFiles["EmbedDim_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # PredictInterval
    #------------------------------------------------------------
    def test_PredictInterval( self ):
        if self.verbose : print ( "--- PredictInterval ---" )
        df_ = EDM.sampleData['block_3sp']
        df = EDM.PredictInterval( dataFrame = df_,
                                  columns = 'x_t', target = 'x_t', maxTp = 15,
                                  lib = [1, 150], pred = [151, 198], E = 3,
                                  tau = -1, numProcess = 10, showPlot = False )

        dfv = round( self.ValidFiles["PredictInterval_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # PredictNonlinear
    #------------------------------------------------------------
    def test_PredictNonlinear( self ):
        if self.verbose : print ( "--- Predict ---" )
        df_ = EDM.sampleData['TentMapNoise']
        df = EDM.PredictNonlinear( dataFrame = df_,
                                   columns = 'TentMap', target = 'TentMap',
                                   lib = [1, 500], pred = [501,800], E = 4,
                                   Tp = 1, tau = -1, numProcess = 10,
                                   theta = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5,
                                            2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20 ],
                                   showPlot = False )

        dfv = round( self.ValidFiles["PredictNonlinear_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # Generative mode
    #------------------------------------------------------------
    def test_generate__simplex1( self ):
        '''Simplex Generate 1'''
        if self.verbose : print ( "--- Simplex Generate 1 ---" )
        df_ = EDM.sampleData["circle"]

        df = EDM.Simplex( dataFrame = df_,
                          columns = 'x', target = 'x',
                          lib = [1,200], pred = [1,2], E = 2,
                          generateSteps = 100, generateConcat = True )

        self.assertTrue( df.shape == (300,4) )

    #------------------------------------------------------------
    def test_generate_simplex2( self ):
        '''Simplex generateSteps 2'''
        if self.verbose : print ( "--- Simplex generateSteps 2 ---" )
        df_ = EDM.sampleData["Lorenz5D"]

        df = EDM.Simplex( df_, "V1", "V1",
                          [1, 1000], [1, 2], 5, 1, 0, -1, 0,
                          False, [], False, 100, False, False, False )

        self.assertTrue( df.shape == (100,4) )

    #------------------------------------------------------------
    def test_generate_smap1( self ):
        '''DateTime'''
        if self.verbose : print ( "--- SMap Generate ---" )
        df_ = EDM.sampleData["circle"]

        S = EDM.SMap( dataFrame = df_,
                      columns = 'x', target = 'x', theta = 3.,
                      lib = [1,200], pred = [1,2], E = 2,
                      generateSteps = 100, generateConcat = True )

        self.assertTrue( S['predictions'].shape == (300,4) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
