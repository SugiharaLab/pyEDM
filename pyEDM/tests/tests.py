
import pyEDM as EDM
import unittest
import pkg_resources # Get data file names from EDM package

from pandas import read_csv

#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_EDM( unittest.TestCase ):

    #def __init__(self, *args, **kwargs):
    #    super( test_EDM, self ).__init__( *args, **kwargs )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def setUpClass( self ):
        self.GetDataFiles( self )
        
    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    def GetDataFiles( self ):
        self.Files = {}

        dataFiles = [ "block_3sp.csv",
                      "Smplx_E3_block_3sp_pyEDM.csv",
                      "Smplx_embd_block_3sp_pyEDM.csv",
                      "circle.csv",
                      "Smap_circle_pyEDM.csv",
                      "Smap_embd_block_3sp_pyEDM.csv",
                      "Multiview_pred_valid.csv",
                      "Multiview_combos_valid.csv",
                      "sardine_anchovy_sst.csv",
                      "CCM_anch_sst_cppEDM_valid.csv" ]
        
        # Create map of module dataFiles pathnames in Files
        for file in dataFiles:
            filename = "data/" + file
            if pkg_resources.resource_exists( EDM.__name__, filename ):
                self.Files[ file ] = \
                    pkg_resources.resource_filename( EDM.__name__, filename )
            else :
                raise Exception( "tests.py: Failed to find data file " + \
                                 file + " in EDM package" )

    #------------------------------------------------------------
    # Simplex
    #------------------------------------------------------------
    def test_simplex( self ):
        #--------------------------------------------------------
        # embedded = False
        #--------------------------------------------------------
        print ( "--- Simplex embedded = False ---" )
        df = EDM.Simplex( "", self.Files[ "block_3sp.csv" ], None, "./", "", 
                          "1 100", "101 195", 3, 1, 0, -1, 0,
                          "x_t", "x_t",
                          False, False, True, False, [], 0, False )

        # cppEDM and devEDM outputs are rounded to os.precision( 4 );
        dfv = EDM.ReadDataFrame( "",
                                 self.Files[ "Smplx_E3_block_3sp_pyEDM.csv" ] )
        
        S1 =       dfv.get('Prediction_t(+1)')[1:95] # Skip row 0 Nan
        S2 = round( df.get('Predictions'), 4 )[1:95] # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex2( self ):
        #--------------------------------------------------------
        # embedded = True
        #--------------------------------------------------------
        print ( "--- Simplex embedded = True ---" )
        df = EDM.Simplex( "", self.Files[ "block_3sp.csv" ], None, "./", "", 
                          "1 99", "100 198", 3, 1, 0, -1, 0,
                          "x_t y_t z_t", "x_t",
                          True, False, True, False, [], 0, False )

        # This has been rounded to os.precision( 4 );
        dfv = EDM.ReadDataFrame( "",
                                 self.Files["Smplx_embd_block_3sp_pyEDM.csv"] )
        
        S1 =       dfv.get('Prediction_t(+1)')[1:98] # Skip row 0 Nan
        S2 = round( df.get('Predictions'), 4 )[1:98] # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )
        
    #------------------------------------------------------------
    # S-map
    #------------------------------------------------------------
    def test_smap( self ):
        #--------------------------------------------------------
        # circle test embedded = True
        #--------------------------------------------------------
        print ( "--- S-map circle embedded = True ---" )
        dfc = EDM.ReadDataFrame( "", self.Files[ "circle.csv" ] )

        # Passing Pandas DataFrame input to Smap rather than path/file
        SM = EDM.SMap( "", "", dfc, "", "",
                       "1 100", "101 198", 2, 1, 0, -1, 4, 0,
                       "x y", "x", "", "",
                       None, True, False, False, False, [], 0, False )
        
        df = SM['predictions']
        
        dfv = EDM.ReadDataFrame( "", self.Files[ "Smap_circle_pyEDM.csv" ] )
        
        S1 =       dfv.get('Prediction_t(+1)')
        S2 = round( df.get('Predictions'), 4 ) 
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_smap2( self ):
        #--------------------------------------------------------
        # block_3sp embedded = True
        #--------------------------------------------------------
        print ( "--- S-map block_3sp embedded = True ---" )
        SM = EDM.SMap( "", self.Files[ "block_3sp.csv" ], None, "./", "", 
                       "1 99", "100 198", 3, 1, 0, -1, 2, 0,
                       "x_t y_t z_t", "x_t", "", "",
                       None, True, False, False, False, [], 0, False )

        df = SM['predictions']

        dfv = EDM.ReadDataFrame( "",
                                 self.Files[ "Smap_embd_block_3sp_pyEDM.csv" ] )

        S1 =       dfv.get('Predictions')
        S2 = round( df.get('Predictions'), 4 ) 
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # Multiview
    #------------------------------------------------------------
    def test_multiview( self ):
        print ( "--- Multiview ---" )
        M = EDM.Multiview( "", self.Files[ "block_3sp.csv" ], None, "./", "", 
                           "1 100", "101 198", 0, 3, 1, 0, -1,
                           "x_t y_t z_t", "x_t", 0, 0,
                           True, False, False, False, 4 )

        df_pred  = M['Predictions']
        df_combo = round( M['View'], 4 )

        # Validate predictions
        dfv = EDM.ReadDataFrame( "", self.Files[ "Multiview_pred_valid.csv" ],
                                 noTime = True )

        # cppEDM and devPDM outputs are rounded to os.precision( 4 );
        M1 = dfv.get('Predictions')
        M2 = round( df_pred.get('Predictions'), 4 ) 
        self.assertTrue( M1.equals( M2 ) )

        # Validate combinations
        dfc = read_csv('../data/Multiview_combos_valid.csv')
        self.assertTrue( dfc.equals( df_combo ) )

    #------------------------------------------------------------
    # CCM
    #------------------------------------------------------------
    def test_ccm( self ):
        print ( "--- CCM ---" )
        df = EDM.CCM( "", self.Files[ "sardine_anchovy_sst.csv" ],
                      None, "./", "", 
                      3, 0, 0, -1, 0, "anchovy", "np_sst",
                      "10 75 5", 1, False, False, 0, False, False, False )

        dfv = EDM.ReadDataFrame( "",
                                 self.Files[ "CCM_anch_sst_cppEDM_valid.csv" ],
                                 noTime = True )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
