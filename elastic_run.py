import pyEDM
df = pyEDM.sampleData['TentMap']

embedded = pyEDM.Embed( dataFrame=df, E=2, columns="TentMap" )

target   = 2*df.iloc[1:,1]

sample = embedded[:10]

from sklearn.linear_model import ElasticNet

pyEDM.SMap(dataFrame=df, columns="TentMap", target="TentMap", E=4,
            lib="1 200",pred="201 202",showPlot=True)
pyEDM.SMap(dataFrame=df, columns="TentMap", target="TentMap", E=4,
            lib="1 200",pred="201 202",showPlot=True,elasticNet=ElasticNet())
