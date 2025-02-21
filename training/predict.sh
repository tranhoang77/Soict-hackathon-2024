#!/bin/bash

GPU_LIST=("$@")

if [ ${#GPU_LIST[@]} -eq 0 ]; then
    echo "missing GPU list"
    exit 1
fi

CHECKPOINTS=(
    "weight/weight_15e_withDat/epoch_15.pth"
    "weight/weight_4e_pseudo/epoch_4.pth"
    "weight/weight_15e_base/epoch_15.pth"
)

OUTPUTS=(
    "../../inference/results/weight_15e_withDat.txt"
    "../../inference/results/weight_4e_pseudo.txt"
    "../../inference/results/weight_15e_base.txt"
)

if [ ${#CHECKPOINTS[@]} -ne ${#OUTPUTS[@]} ]; then
    echo "Eror: Unmatch between OUTPUTS and CHECKPOINTS"
    exit 1
fi

NUM_GPUS=${#GPU_LIST[@]}
NUM_TASKS=${#CHECKPOINTS[@]}

run_task() {
    local task_index="$1"
    local gpu_id="$2"
    local checkpoint_file="$3"
    local output_file="$4"

    mkdir -p "$(dirname "$output_file")"

    CUDA_VISIBLE_DEVICES="$gpu_id" python3 predict.py \
        --checkpoint_file "$checkpoint_file" \
        --input_dir ../../data/public_test \
        --output_file "$output_file"
}

declare -a RUNNING_PIDS
declare -a GPU_STATUS


for ((i=0; i<NUM_GPUS; i++)); do
    GPU_STATUS[$i]=0
done

find_free_gpu() {
    local i
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ ${GPU_STATUS[$i]} -eq 0 ]; then
            echo "$i"
            return
        fi
    done
    echo "-1"
}


manage_processes() {
    local i pid gpu_index

    for i in "${!RUNNING_PIDS[@]}"; do
        if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
            # Nếu tiến trình không còn tồn tại, giải phóng GPU và loại bỏ PID khỏi danh sách
            gpu_index=$((i % NUM_GPUS))
            GPU_STATUS[$gpu_index]=0
            unset 'RUNNING_PIDS[$i]'
        fi
    done


    RUNNING_PIDS=("${RUNNING_PIDS[@]}")  # Remove gaps in the array
}

# Phân phối và chạy các task
task_index=0
while [ $task_index -lt $NUM_TASKS ]; do
    manage_processes

    # Tìm GPU rảnh
    GPU_INDEX=$(find_free_gpu)
    
    if [ "$GPU_INDEX" -ne -1 ]; then
        # Lấy ID GPU
        GPU_ID=${GPU_LIST[$GPU_INDEX]}
        
        # Đánh dấu GPU đang bận
        GPU_STATUS[$GPU_INDEX]=1
        
        # Chạy task trong background
        run_task "$task_index" "$GPU_ID" "${CHECKPOINTS[$task_index]}" "${OUTPUTS[$task_index]}" &
        PID=$!
        
        # Lưu PID 
        RUNNING_PIDS+=("$PID")
        
        echo "task $task_index GPU $GPU_ID checkpoint ${CHECKPOINTS[$task_index]} (PID: $PID)"
        
        ((task_index++))
    else
        # Chờ nếu không có GPU rảnh
        sleep 1
    fi
done

# Chờ tất cả các task hoàn thành
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
    manage_processes
    sleep 1
done


