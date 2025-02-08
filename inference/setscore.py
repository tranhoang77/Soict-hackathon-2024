import argparse

def filter_detections(input_file_cam11, input_file_cam13, output_file):
    # Dictionary mapping class ids to their minimum score thresholds
    score_thresholds = {
        0: 0.1,  # motorcycle
        1: 0.21,  # car
        2: 0.1,  # bus
        3: 0.1  # truck
    }

    score_thresholds2 = {
        0: 0.5,  # motorcycle
        1: 0.1,  # carss
        2: 0.1,  # bus
        3: 0.1  # truck
    }

    # Read and process the file
    filtered_lines = []

    with open(input_file_cam11, 'r') as f:
        for line in f:
            # Split the line into components
            parts = line.strip().split()

            # Extract relevant information
            image_name = parts[0]
            class_id = int(parts[1])
            score = float(parts[6])  # Score is at index 6

            if "cam_11" in image_name:
                if class_id in score_thresholds and score >= score_thresholds[class_id]:
                    filtered_lines.append(line)

    # Second file hardcoded path
    with open(input_file_cam13, 'r') as f: 
        for line in f:
            # Split the line into components
            parts = line.strip().split()

            # Extract relevant information
            image_name = parts[0]
            class_id = int(parts[1])
            score = float(parts[6])  # Score is at index 6

            if "cam_13" in image_name:
                if class_id in score_thresholds and score >= score_thresholds2[class_id]:
                    filtered_lines.append(line)

    # Write filtered results to new file
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter detections based on score thresholds.")
    parser.add_argument("--input_file_cam11", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--input_file_cam13", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")

    args = parser.parse_args()

    try:
        filter_detections(args.input_file_cam11, args.input_file_cam13, args.output_file)
        print(f"Successfully filtered detections. Results saved to {args.output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
