#!/bin/bash

# Usage help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Convert JSON configuration to MEEPSAT Python script"
    echo ""
    echo "Options:"
    echo "  -f FILE    Path to JSON file (required)"
    echo "  -o DIR     Output directory (default: current directory)"
    echo "  -n NAME    Output script name (default: derived from JSON filename)"
    echo "  -p PATH    Path to MEEPSAT installation (default: /home/ashesh_ak/Phd_work/MEEPSAT_WFH)"
    echo "  -h         Display this help message"
    exit 1
}

# Initialize variables
JSON_FILE=""
OUTPUT_DIR="$(pwd)"
OUTPUT_NAME=""
MEEPSAT_PATH="/home/ashesh_ak/Phd_work/MEEPSAT_WFH"  # Default path

# Parse command line options
while getopts "f:o:n:p:h" opt; do
    case ${opt} in
        f )
            JSON_FILE=$OPTARG
            ;;
        o )
            OUTPUT_DIR=$OPTARG
            ;;
        n )
            OUTPUT_NAME=$OPTARG
            ;;
        p )
            MEEPSAT_PATH=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if JSON file is provided
if [ -z "$JSON_FILE" ]; then
    echo "Error: JSON file path is required"
    usage
fi

# Check if the file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File '$JSON_FILE' not found"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory '$OUTPUT_DIR' does not exist"
    echo "Would you like to create it? (y/n)"
    read create_dir
    if [ "$create_dir" = "y" ]; then
        mkdir -p "$OUTPUT_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create directory '$OUTPUT_DIR'"
            exit 1
        fi
        echo "Directory created: $OUTPUT_DIR"
    else
        exit 1
    fi
fi

# Check if the directory is writable
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "Error: Directory '$OUTPUT_DIR' is not writable"
    exit 1
fi

# Check if the file has a .json extension
if [[ "$JSON_FILE" != *.json ]]; then
    echo "Warning: File '$JSON_FILE' does not have a .json extension"
    echo "Continue anyway? (y/n)"
    read answer
    if [ "$answer" != "y" ]; then
        exit 0
    fi
fi

# Execute Python script
echo "Converting $JSON_FILE to Python script..."
echo "Using MEEPSAT path: $MEEPSAT_PATH"

# Use the same function for both cases, with or without a custom name
if [ -z "$OUTPUT_NAME" ]; then
    python -c "import sys; sys.path.append('$MEEPSAT_PATH'); from meepsat.json_to_script import json_to_pyscript; json_to_pyscript('$JSON_FILE', output_dir='$OUTPUT_DIR')"
else
    python -c "import sys; sys.path.append('$MEEPSAT_PATH'); from meepsat.json_to_script import json_to_pyscript; json_to_pyscript('$JSON_FILE', output_dir='$OUTPUT_DIR', output_name='$OUTPUT_NAME')"
fi

# Check if the conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully"
    if [ -z "$OUTPUT_NAME" ]; then
        echo "Output saved to: $OUTPUT_DIR"
    else
        echo "Output saved as: $OUTPUT_DIR/$OUTPUT_NAME"
    fi
else
    echo "Error: Conversion failed"
    exit 1
fi