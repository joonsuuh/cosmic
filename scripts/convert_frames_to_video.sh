#!/bin/bash

# Define possible frame directories to check (in order of priority)
POSSIBLE_DIRS=(
  "../frames"
  "../build/frames"
  "./frames"
  "frames"
)

# Find the first valid frames directory
FRAMES_DIR=""
for dir in "${POSSIBLE_DIRS[@]}"; do
  if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
    FRAMES_DIR="$dir"
    break
  fi
done

# Exit if no valid directory found
if [ -z "$FRAMES_DIR" ]; then
    echo "Error: No frames directory found. Please check that one of these exists:"
    for dir in "${POSSIBLE_DIRS[@]}"; do
        echo "  - $dir"
    done
    exit 1
fi

# Count the number of frame files
FRAME_COUNT=$(ls -1 $FRAMES_DIR/frame_*.ppm 2>/dev/null | wc -l)
if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "Error: No frame files found in $FRAMES_DIR"
    exit 1
fi

echo "Found $FRAME_COUNT frames in $FRAMES_DIR"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it first."
    echo "On Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "On Fedora: sudo dnf install ffmpeg"
    echo "On macOS with Homebrew: brew install ffmpeg"
    exit 1
fi

# Create output directory if it doesn't exist
VIDEOS_DIR="../assets/videos"
mkdir -p $VIDEOS_DIR

# Get current timestamp for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$VIDEOS_DIR/black_hole_${TIMESTAMP}.mp4"

# Ask for framerate
read -p "Enter framerate for the video (recommended: 30): " FRAMERATE
FRAMERATE=${FRAMERATE:-30}

# Show debug info
echo "Input path: $FRAMES_DIR/frame_%06d.ppm"
echo "Output file: $OUTPUT_FILE"
echo "Framerate: $FRAMERATE fps"

# Verify input files exist
FIRST_FRAME=$(ls $FRAMES_DIR/frame_*.ppm 2>/dev/null | head -n 1)
if [ ! -f "$FIRST_FRAME" ]; then
    echo "Error: Cannot access frame files. Check permissions."
    exit 1
else
    echo "First frame found: $FIRST_FRAME"
fi

# Convert frames to video with detailed output
echo "Converting frames to video with framerate: $FRAMERATE fps..."
ffmpeg -v info -framerate $FRAMERATE -i "$FRAMES_DIR/frame_%06d.ppm" -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium "$OUTPUT_FILE"

# Check if video was created successfully and has content
if [ $? -eq 0 ] && [ -s "$OUTPUT_FILE" ]; then
    echo "Video created successfully: $OUTPUT_FILE"
    
    # Display video information
    echo "Video information:"
    ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_FILE" | grep -E 'width|height|duration|bit_rate'
    
    # Optional: Ask if user wants to delete the frames to save space
    read -p "Do you want to delete the PPM frames to save space? (y/n): " choice
    case "$choice" in 
        y|Y ) 
            rm -f $FRAMES_DIR/frame_*.ppm
            echo "Frames deleted."
            ;;
        * ) 
            echo "Frames preserved."
            ;;
    esac
else
    echo "Error: Video creation failed or output is empty."
    echo "Try running the following command manually to see detailed errors:"
    echo "ffmpeg -v debug -framerate $FRAMERATE -i \"$FRAMES_DIR/frame_%06d.ppm\" -c:v libx264 -pix_fmt yuv420p \"$OUTPUT_FILE\""
fi
