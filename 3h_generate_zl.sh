#!/bin/bash

# Check for the correct number of input arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 cpu_cores"
    exit 1
fi

# Read the argument for the number of CPU cores
cpu_cores=$1

# Function to process each subject folder
process_folder() {
    folder=$1
    cd "$folder" || exit
    echo "Processing: $folder"

    # Extract the subject ID from the folder name
    subject_id=$(basename "$folder")

    # Check if the subject ID is a number
    if ! [[ $subject_id =~ ^[0-9]+$ ]]; then
        echo "Skipping: $folder is not a valid subject ID"
        return
    fi


    # Run gyralnet.py
    python3 /home/lab/Documents/HCP_data/gyralnet.py --root_dir="$PWD" --subject_list_start_id="$subject_id" --subject_list_end_id="$((subject_id + 1))" --input_dir="${subject_id}_recon" --out_dir='gyralnet_island_164k_flip'

    echo "Completed processing: $subject_id" 
}

# Export function for parallel to use
export -f process_folder

# Get all subject folders in the current directory
subject_folders=($(find . -maxdepth 1 -type d -name "*" | sed 's|./||'))

# Process each directory using GNU Parallel
parallel -j $cpu_cores process_folder ::: "${subject_folders[@]}"

echo "All processing complete."
