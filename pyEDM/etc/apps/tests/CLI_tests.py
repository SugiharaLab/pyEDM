
# CLI Tests

./Embedding.py -i ../../data/LorenzData1000.csv -c V1 V3
./Embedding.py -i ../../data/LorenzData1000.csv -c V1 V3 -v -p -tau 2

./EmbedDim_Multiprocess_Columns.py -v -P
./EmbedDim_Multiprocess_Columns.py -x 20 -T 10 -P

./CCM_Multiprocess_Columns.py -P

./CCM_Multiprocess_LibSizes.py -P
./CCM_Multiprocess_LibSizes.py -P -C 6 -l "40 80 100" "120 150" "200 400" "600" "800" "980"

./SMap_Multiprocess_Tp.py -P
./SMap_Multiprocess_Tp.py -th 2 -x 20 -tau -5 -P

./PredictNL_Multiprocess_Columns.py -P
./PredictNL_Multiprocess_Columns.py -x 20 -T 5 -P
