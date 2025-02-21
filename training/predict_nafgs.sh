#!/bin/bash

# Kiểm tra và chuyển đổi các đối số được truyền vào thành mảng GPU
GPU_LIST=("$@")

# Kiểm tra xem có GPU được chỉ định không
if [ ${#GPU_LIST[@]} -eq 0 ]; then
    echo "missing GPU list"
    exit 1
fi

# Danh sách các checkpoint
CHECKPOINTS=(
    "weight/weight_2e_nafgs_pseudo/epoch_2.pth"
    "weight/weight_3e_nafgs_8e/epoch_3.pth"
    "weight/weight_15e_nafgs_withDat/epoch_15.pth"
    "weight/weight_15e_nafgs_nocam05/epoch_15.pth"
)
# Danh sách các file output
OUTPUTS=(
    "../../inference/results/weight_2e_nafgs_pseudo.txt"
    "../../inference/results/weight_3e_nafgs_8e.txt"
    "../../inference/results/weight_15e_nafgs_withDat.txt"
    "../../inference/results/weight_15e_nafgs_nocam05.txt"
)

# Kiểm tra tính nhất quán của mảng
if [ ${#CHECKPOINTS[@]} -ne ${#OUTPUTS[@]} ]; then
    echo "Error: checkpoints and outputs not equal"
    exit 1
fi

# Số lượng GPU và task
NUM_GPUS=${#GPU_LIST[@]}
NUM_TASKS=${#CHECKPOINTS[@]}

# Hàm để chạy task trên GPU
run_task() {
    local task_index="$1"
    local gpu_id="$2"
    local checkpoint_file="$3"
    local output_file="$4"

    # Kiểm tra tồn tại của file checkpoint

    # Tạo thư mục output nếu chưa tồn tại
    mkdir -p "$(dirname "$output_file")"

    # Chạy predict với GPU được chỉ định
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 predict.py \
        --checkpoint_file "$checkpoint_file" \
        --input_dir ../../data/public_test_nafnet_gsad \
        --output_file "$output_file"
}

# Khởi tạo mảng để quản lý trạng thái
declare -a RUNNING_PIDS
declare -a GPU_STATUS

# Khởi tạo trạng thái GPU
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_STATUS[$i]=0
done

# Hàm tìm GPU rảnh
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

# Hàm kiểm tra và quản lý các tiến trình
manage_processes() {
    local i pid

    # Kiểm tra và loại bỏ các tiến trình đã hoàn thành
    for i in "${!RUNNING_PIDS[@]}"; do
        if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
            # Nếu tiến trình không còn tồn tại, giải phóng GPU
            gpu_index=$((i % NUM_GPUS))
            GPU_STATUS[$gpu_index]=0
            unset 'RUNNING_PIDS[$i]'
        fi
    done
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
        
        echo "Đang chạy task $task_index trên GPU $GPU_ID với checkpoint ${CHECKPOINTS[$task_index]} (PID: $PID)"
        
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

echo "Tất cả các task đã hoàn thành."
