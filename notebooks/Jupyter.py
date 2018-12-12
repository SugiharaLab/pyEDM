# Python distribution modules
import sys
from   collections     import OrderedDict
from   multiprocessing import Pool
from   copy            import deepcopy

# Community modules
import ipywidgets as widgets
from   IPython.display import display
import matplotlib.pyplot as plt
from   matplotlib.dates import num2date

# Local modules
# Patch sys.path so local modules are found in ../
# Alternative is to set the JUPYTER_PATH environment variable 
sys.path.append("../")
import Methods
from   ArgParse import AdjustArgs, ParseCmdLine
from   EDM      import ReadEmbeddedData, EmbedData, Prediction, \
                       ComputeError, nCol, nRow

# Monkey patch sys.argv so that parser.parse_args() doesn't choke
# on invalid arguments from the IPython/Jupyter invocation
sys.argv = ['Jupyter Module from pyEDM']

# Globals 
args    = ParseCmdLine()
Widgets = OrderedDict() # Dictionary of arg names and widgets

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def GetArgs( arguments ):
    '''Get args parameters from GUI/widgets or passed in arguments'''
    global args
    
    if not arguments:
        UpdateArgs()       # Update global args from GUI/widgets
    else:
        args = arguments   # Replace global args with arguments
        AdjustArgs( args ) # Index offsets and parameter validation.
        
    return args

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Embed( arguments = None ):
    '''Interface for Embed() in Methods'''
    args = GetArgs( arguments )
    
    D = Methods.Embed( args, Methods.Source.Jupyter )
    return D # Dictionary { header, embedding, target }

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Predict( arguments = None ):
    '''Interface for Predict() in Methods'''
    args = GetArgs( arguments )
    
    D = Methods.Predict( args, Methods.Source.Jupyter )
    return D # Dictionary { rho, RMSE, MAE, header, prediction, S-map }

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def EmbedDimension( arguments = None ):
    '''Interface for EmbedDimension() in Methods'''
    args = GetArgs( arguments )

    E_rho = Methods.EmbedDimensions( args, Methods.Source.Jupyter )
    return E_rho # Dictionary { E:rho }
    
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def PredictDecay( arguments = None ):
    '''Interface for PredictDecays() in Methods'''
    args = GetArgs( arguments )

    Tp_rho = Methods.PredictDecays( args, Methods.Source.Jupyter )
    return Tp_rho # Dictionary { Tp:rho }
    
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def SMapNL( arguments = None ):
    '''Interface for SMapNL() in Methods'''
    args = GetArgs( arguments )

    theta_rho = Methods.SMapNL( args, Methods.Source.Jupyter )
    return theta_rho # Dictionary { theta:rho }

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def Multiview( arguments = None ):
    '''Interface for Multiview() in Methods'''
    args = GetArgs( arguments )

    D = Methods.Multiview( args, Methods.Source.Jupyter )
    return D # Dictionary { header, multiview, rho, r, RMSE, MAE }

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def CCM( arguments = None ):
    '''Interface for CCM() in Methods'''
    args = GetArgs( arguments )

    D = Methods.CCM( args, Methods.Source.Jupyter )
    return D # Dictionary { lib_sizes, column_target_rho, target_column_rho }

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def init():
    '''Create notebook widgets to hold args parameters'''
    
    method = widgets.Dropdown( options=['Simplex','SMap'], value='Simplex',
                               description='method:' )
    
    lib = widgets.IntRangeSlider( value=[1,100], min=1, max=1000, step=1,
                                  description='lib:' )

    pred = widgets.IntRangeSlider( value=[201,500], min=1, max=1000, step=1,
                                   description='pred:' )
    
    E = widgets.IntSlider( value=-1, min=-1, max=12, step=1, description='E:' )
    
    knn = widgets.IntSlider( value=-1, min=-1, max=100, step=1,
                             description='knn:' )
    
    noNeighborLimit = widgets.Checkbox( value=False,
                                        description='no Neighbor Limit' )
    
    Tp = widgets.IntSlider( value=0, min=0, max=20, step=1, description='Tp:' )
    
    theta = widgets.FloatText( value=0, description='theta:',
                               layout = widgets.Layout(width='70%') )
    
    # jacobians is a list of int arguments, converted in UpdateArgs()
    jacobians = widgets.Text( value='', description='Jacobians:')
    
    svd = widgets.Checkbox( value=False, description='SVD',
                            layout = widgets.Layout(width='50%') )
    
    TikhonovAlpha = widgets.FloatText(value=None, description='Tikhonov Alpha:')
    
    multiview = widgets.IntSlider( value=0, min=0, max=20, step=1,
                                   description='multiview:')
    
    tau = widgets.IntSlider( value=1, min=1, max=20, step=1, description='tau:')
    
    forwardTau = widgets.Checkbox( value=False, description='forward Tau:' )
    
    columns = widgets.Text( value='1', description='columns:')
    
    target = widgets.Text( value='', description='target:')
    
    embedded = widgets.Checkbox( value=False, description='embedded' )
    
    libsize = widgets.Text( value='', description='libsize:')
    
    subsample = widgets.IntSlider( value=100, min=10, max=200, step=1,
                                   description='subsample:' )
    
    randomLib = widgets.Checkbox( value=False, description='randomLib',
                                  layout = widgets.Layout(width='70%') )
    
    seed = widgets.IntText( value=None, description='seed:' )
    
    path = widgets.Text( value='../data/', description='path:')
    
    inputFile = widgets.Text( value='TentMap_rEDM.csv',
                              description='input file:')
    
    outputFile = widgets.Text( value='', description='output file:')
    
    outputSmapFile = widgets.Text( value='', description='output S-Map:',
                                   style = {'description_width': 'initial'} )
    
    outputEmbed = widgets.Text( value='', description='output embedded:',
                                style = {'description_width': 'initial'} )
    
    plot = widgets.Checkbox( value=False, description='plot',
                             layout = widgets.Layout(width='50%') )

    verbose = widgets.Checkbox( value=False, description='verbose',
                                layout = widgets.Layout(width='70%') )

    warnings = widgets.Checkbox( value=False, description='warnings',
                                 layout = widgets.Layout(width='70%') )

    Debug = widgets.Checkbox( value=False, description='Debug',
                              layout = widgets.Layout(width='70%') )

    # Populate global dictionary of widgets
    Widgets['method']          = method
    Widgets['library']         = lib
    Widgets['prediction']      = pred
    Widgets['E']               = E
    Widgets['knn']             = knn
    Widgets['noNeighborLimit'] = noNeighborLimit
    Widgets['Tp']              = Tp
    Widgets['theta']           = theta
    Widgets['jacobians']       = jacobians
    Widgets['svd']             = svd
    Widgets['TikhonovAlpha']   = TikhonovAlpha
    Widgets['tau']             = tau
    Widgets['columns']         = columns
    Widgets['target']          = target
    Widgets['embedded']        = embedded
    Widgets['libsize']         = libsize
    Widgets['subsample']       = subsample
    Widgets['randomLib']       = randomLib
    Widgets['path']            = path
    Widgets['inputFile']       = inputFile
    Widgets['outputFile']      = outputFile
    Widgets['outputSmapFile']  = outputSmapFile
    Widgets['outputEmbed']     = outputEmbed
    Widgets['plot']            = plot
    Widgets['verbose']         = verbose

    # Organize widgets 
    left_box  = widgets.VBox( [ method, pred, lib, columns, target,
                                libsize, widgets.HBox( [ plot, verbose ] ) ] )
    mid_box   = widgets.VBox( [ E, knn, Tp, theta, tau, subsample,
                                widgets.HBox( [ randomLib, svd ] ) ] )
    right_box = widgets.VBox( [ path, inputFile, outputFile, outputSmapFile,
                                outputEmbed, jacobians, embedded ] )
    # Display widgets on notebook
    display( widgets.HBox( [ left_box, mid_box, right_box ] ) )

    UpdateArgs()

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def UpdateArgs():
    '''Update EDM args parameters from notebook widgets'''

    args.method          = Widgets['method'].value
    args.prediction      = Widgets['prediction'].value
    args.library         = Widgets['library'].value
    args.E               = Widgets['E'].value
    args.k_NN            = Widgets['knn'].value
    args.noNeighborLimit = Widgets['noNeighborLimit'].value
    args.Tp              = Widgets['Tp'].value
    args.theta           = Widgets['theta'].value
    args.jacobians       = [ int(x) for x in Widgets['jacobians'].value.split()]
    args.TikhonovAlpha   = Widgets['TikhonovAlpha'].value
    args.tau             = Widgets['tau'].value
    args.columns         = Widgets['columns'].value.split()
    args.target          = Widgets['target'].value
    args.embedded        = Widgets['embedded'].value
    args.libsize         = [ int(x) for x in Widgets['libsize'].value.split() ]
    args.subsample       = Widgets['subsample'].value
    args.randomLib       = Widgets['randomLib'].value
    args.path            = Widgets['path'].value
    args.inputFile       = Widgets['inputFile'].value
    args.outputFile      = Widgets['outputFile'].value
    args.outputSmapFile  = Widgets['outputSmapFile'].value
    args.outputEmbed     = Widgets['outputEmbed'].value
    args.plot            = Widgets['plot'].value
    args.verbose         = Widgets['verbose'].value

    AdjustArgs( args ) # Index offsets and parameter validation. 
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Notebook():
    '''Determine if Jupyter notebook is the parent process'''
    try:
        ipython = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in ipython:
            return True   # Jupyter notebook or qtconsole
        elif 'TerminalInteractiveShell' in ipython:
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def IPythonVersion():
    import IPython
    return IPython.__version__
