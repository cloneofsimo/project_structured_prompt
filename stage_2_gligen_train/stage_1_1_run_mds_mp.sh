#!/bin/bash

# # Launch the Python3.10 script on different GPUs
CUDA_VISIBLE_DEVICES=0 python3.10 stage_1_preprocess_to_streaming.py --device cuda --file_index 370 &
# CUDA_VISIBLE_DEVICES=1 python3.10 stage_1_preprocess_to_streaming.py --device cuda --file_index 1 &
# CUDA_VISIBLE_DEVICES=2 python3.10 stage_1_preprocess_to_streaming.py --device cuda --file_index 2 &
# CUDA_VISIBLE_DEVICES=3 python3.10 stage_1_preprocess_to_streaming.py --device cuda --file_index 3 &

# wait

# #!/bin/bash

# # # Total number of GPUs available
# NUM_GPUS=8

# # Start and end file indices
# START_INDEX=40
# END_INDEX=369

# # Loop through each file index
# for ((i=START_INDEX; i<=END_INDEX; i++)); do
#     # Calculate GPU index to use
#     GPU_INDEX=$((i % NUM_GPUS))

#     # Run the Python3.10 script on the calculated GPU
#     CUDA_VISIBLE_DEVICES=$GPU_INDEX python3.10 stage_1_preprocess_to_streaming.py --device cuda --file_index $i &

#     # If we've reached the max number of concurrent jobs equal to NUM_GPUS, wait for them to finish
#     if [ $((i % NUM_GPUS)) -eq 0 ]; then
#         wait
#     fi
# done

# # Wait for the last set of jobs to finish
# wait
