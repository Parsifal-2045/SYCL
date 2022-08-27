DEV_ID=$1
# EXEC=./main
USE_GPU="${2:-1}"
EXEC="${3:-./main}"

echo "Launch Params"
echo "- USE_GPU: $USE_GPU"
echo "- EXEC: $EXEC"

# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_1000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_2000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_3000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_4000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_5000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_6000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_7000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_8000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_9000.csv 20 25 2 $USE_GPU 10 0
# CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/toyDetector_10000.csv 20 25 2 $USE_GPU 10 0

FILES="toyDetector_1000.csv toyDetector_2000.csv toyDetector_3000.csv toyDetector_4000.csv \
toyDetector_5000.csv toyDetector_6000.csv toyDetector_7000.csv toyDetector_8000.csv \
toyDetector_9000.csv toyDetector_10000.csv"
touch results_file.csv
for FileName in $FILES
do
    echo "Processing $FileName..."
    echo "$FileName" | sed 's/....$/,/' >> .mytemp.csv # Column Header 
    CUDA_VISIBLE_DEVICES=$DEV_ID $EXEC data/input/$FileName 20 25 2 $USE_GPU 10 0 | tail -12 | head -10 | cut -d' ' -f 3 | sed 's/..$/,/' >> .mytemp.csv
    paste -d '' results_file.csv .mytemp.csv > new_results_file.csv && mv new_results_file.csv results_file.csv
    rm .mytemp.csv
done


# Remove last , of each line
cat results_file.csv | sed 's/.$//' > new_results_file.csv && mv new_results_file.csv results_file.csv