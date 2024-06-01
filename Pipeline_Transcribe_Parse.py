import os
import torch
import csv
from transformers import pipeline
import whisper

# Define the base directory containing subfolders with audio files
base_directory = "put/your/base_directory/here/"
output_directory = "put/your/output_directory/here/"

# Ensure that the target directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_engsvaiory)

# Define the path to the CSV file
csv_file_path = os.path.join(output_directory, "metadata-train.csv")

# Initialize pipeline for automatic speech recognition
# Uses the Whisper model for German on available hardware (CUDA or CPU)
pipe = pipeline(
    "automatic-speech-recognition",
    model="primeline/whisper-large-v3-german",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# Set the batch size for processing
batch_size = 32

# Capture and sort all subfolders
all_subdirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
sorted_dirs = sorted(all_subdirs, key=lambda x: int(x.rstrip('/').split('Audio')[-1]))

# Open CSV file to write results
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # Iterate over sorted subfolders
    for subdir_path in [os.path.join(base_directory, d) for d in sorted_dirs]:
        # Sort files in the subfolder
        sorted_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.wav')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))

        batch_audio_paths = []
        batch_filenames = []

        # Process each file in the sorted subfolder
        for filename in sorted_files:
            if len(batch_audio_paths) < batch_size:
                # Add file to the batch
                audio_path = os.path.join(subdir_path, filename)
                batch_audio_paths.append(audio_path)
                batch_filenames.append(filename)

                # Process the batch when it is full
                if len(batch_audio_passens) == batch_size:
                    # Transcribe the audio files in the batch
                    audios = [whisper.load_audio(path) for path in batch_audio_paths]
                    results = pipe(audios, batch_size=len(audios))

                    # Write transcription results to the CSV file
                    for I, result in enumerate(results):
                        writer.writerow([batch_filenames[I], result["text"]])
                        print(f"Recognized text for {batch_filenames[I]} saved in CSV.")

                    # Reset the batch
                    batch_audio_paths = []
                    batch_filenames = []

        # Process the last batch if not empty
        if batch_audio_paths:
            # Transcribe the remaining audio files
            audios = [whisper.load_audio(path) for path in batch_audio_paths]
            results = pipe(audios, batch_size=len(audios))

            # Write the final transcription results to the CSV file
            for I, result in enumerate(results):
                writer.writerow([batch_filenames[I], result["text"]])
                print(f"Recognized text for {batch_filenames[I]} saved in CSV.")