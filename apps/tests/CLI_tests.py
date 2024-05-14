
# CLI Tests
# Default -d Lorenz5D

./Embedding.py -i ../src/pyEDM/data/LorenzData1000.csv -c V1 V3 -v
./Embedding.py -i ../src/pyEDM/data/LorenzData1000.csv -c V1 V3 -p -tau 2 -v

./SMap_Multiprocess_Tp.py -v -P
./SMap_Multiprocess_Tp.py -th 3.3 -T 0 1 2 3 4 5 6 7 8 9 10 -tau -5 -v -P

./SMap_Multiprocess_theta.py -P
./SMap_Multiprocess_theta.py -th 0.01 0.05 0.08 0.1 0.5 0.75 1 2 3 4 5 6 7 8 9 -v -P

./CrossMap_Multiprocess_Columns.py -t V1 -E 5 -l 1 500 -p 501 800 -v -P

./CrossMap_Multiprocess_Matrix.py -E 5 -v -P
./CrossMap_Multiprocess_Matrix.py -i ~/Research/ZebrafishDynamics/data/FishDyn50.csv -E 7 -C 5 -P
