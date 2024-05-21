# Audio Preprocessing Toolkit for Whisper Fine-Tuning

## Introduction

The Audio Preprocessing Toolkit is designed to facilitate the preprocessing of audio files for the purpose of fine-tuning OpenAI's Whisper model. It automates the task of audio segmentation to create high-quality datasets necessary for effective machine learning training.

## Features

- **Dynamic Thresholding**: Adaptive calculation of audio energy and ZCR to segment audio based on its content.
- **Adjustable Segment Sizes**: Customizable minimum and maximum chunk lengths and frame length.
- **Concurrency for Efficiency**: Utilizes Python's `concurrent.futures` to process multiple files concurrently.
- **Customizable Sample Rates**: Supports various audio sampling rates to accommodate different types of audio inputs.
- **Organized Output**: Automatically sorts output audio chunks into dedicated directories per input file.

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
