# Audio Preprocessing Toolkit for Whisper Fine-Tuning

## Introduction

The Audio Preprocessing Toolkit is designed to streamline the entire process of preparing audio files for fine-tuning OpenAI’s Whisper model. This comprehensive toolkit not only automates the segmentation of audio into high-quality datasets but also facilitates the creation of metadata, uploads the processed datasets to Hugging Face, and sets up custom loading scripts for easy integration and use. It simplifies complex tasks, enhancing efficiency and effectiveness in training machine learning models with precision. This toolkit complements the step-by-step guide provided on my Medium blog and will continue to be updated with additional scripts for fine-tuning, ensuring it remains a valuable resource for developers.

## Features

-**Flexible Configuration Options**: Provides adjustable settings for managing audio segment sizes, frame lengths, and chunk overlaps, tailored to diverse preprocessing needs.

-**Efficient Parallel Processing**: Utilizes Python’s concurrent.futures to handle multiple files concurrently, enhancing the overall processing speed.

-**Versatile Audio Support**: Supports a variety of audio sampling rates, accommodating a range of audio types and sources.

-**Structured Data Management**: Ensures processed audio and metadata are well-organized into dedicated directories for each input file, simplifying dataset handling.

-**Comprehensive Metadata Handling**: Automates the creation and management of metadata files, crucial for effective dataset utilization in machine learning workflows.

-**Seamless Hugging Face Integration**: Facilitates the easy uploading of datasets to the Hugging Face platform, promoting easy access and collaboration.

-**Custom Loading Scripts**: Provides tools for setting up custom data loaders, optimizing the integration of datasets into training and evaluation processes.


## Intended Use
This toolkit is designed for researchers and developers who work with speech recognition and processing. It simplifies the task of breaking down large audio datasets into smaller, contextually intact segments. Additionally, it helps in managing metadata, uploading datasets to Hugging Face, and setting up datasets for training or fine-tuning OpenAI’s Whisper model. This makes it a practical tool for anyone looking to prepare and utilize audio data efficiently.

## Getting Started

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
