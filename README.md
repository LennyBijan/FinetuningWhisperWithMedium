# Audio Preprocessing Toolkit for Whisper Fine-Tuning

## Introduction

The Audio Preprocessing Toolkit is designed to facilitate the preprocessing of audio files for the purpose of fine-tuning OpenAI's Whisper model. It automates the task of audio segmentation to create high-quality datasets necessary for effective machine learning training.

## Features

- **Dynamic Thresholding**: Adaptive calculation of audio energy and ZCR to segment audio based on its content.
- **Adjustable Segment Sizes**: Customizable minimum and maximum chunk lengths and frame length.
- **Concurrency for Efficiency**: Utilizes Python's `concurrent.futures` to process multiple files concurrently.
- **Customizable Sample Rates**: Supports various audio sampling rates to accommodate different types of audio inputs.
- **Organized Output**: Automatically sorts output audio chunks into dedicated directories per input file.

## Intended Use
This toolkit is particularly useful for researchers and developers working on speech recognition and processing tasks who need to preprocess large audio datasets into smaller, more manageable units without losing contextual integrity. It is also an essential step for those preparing datasets for training or fine-tuning the Whisper model.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Librosa
- SoundFile
- NumPy

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/audiopreprocessing-toolkit.git
cd audiopreprocessing-toolkit
pip install -r requirements.txt
```

## Execution

To start processing your audio files with the Audio Preprocessing Toolkit, simply run the main script. Ensure that you have the necessary directory paths and parameters set up in the script before executing.

Here are the steps to run the script:

  1. Modify the Script (if needed):

Open the main script file and modify the source_dir and output_dir variables to point to your audio files directory and the desired output directory, respectively. You can also adjust other parameters such as frame_length_ms, min_chunk_length_ms, and max_chunk_length_ms according to your requirements.

  2. Execute the Script:

Navigate to the directory containing your script in your terminal or command prompt, and run the following command:
    
    python your_script_name.py
    
  Replace your_script_name.py with the name of your Python script file.

## Example Commmand

    python preprocess_audio.py

This command will process all .wav files in the specified source directory, segmenting them according to the defined parameters, and save the output in the designated directory.

# Socials

Make sure to check out my medium blog! https://medium.com/@lenny.bijan
