#!/bin/bash

# Example script to process multiple audio files
# Usage: ./examples/batch_process.sh /path/to/audio/files/

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_with_audio_files>"
    exit 1
fi

AUDIO_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/../transcribe_audio.py"

echo "Processing all audio files in: $AUDIO_DIR"

# Process all common audio file formats
for file in "$AUDIO_DIR"/*.{wav,mp3,mp4,m4a,flac,aac}; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        python "$MAIN_SCRIPT" "$file"
        
        # Rename output to include source filename
        filename=$(basename "$file")
        name="${filename%.*}"
        mv transcription.txt "transcription_${name}.txt"
        
        echo "Completed: transcription_${name}.txt"
        echo "---"
    fi
done

echo "Batch processing complete!"
