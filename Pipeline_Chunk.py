import os
import soundfile as sf
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def adjust_to_frame_length(chunk_length_ms, frame_length_ms):
    """
    Adjusts the chunk length to be a multiple of frame length.

    This function ensures that the chunk length is always rounded up to the nearest
    frame boundary to prevent cutting off audio within a frame's duration.
    """
    return ((chunk_length_ms + frame_length_ms - 1) // frame_length_ms) * frame_length_ms

def samples_per_ms(sample_rate, ms):
    """
    Converts milliseconds to the number of samples.

    This conversion is essential for processing the audio at the correct temporal
    resolutions, given a specific sample rate.
    """
    return int(sample_rate * ms / 1000)

def analyze_chunks(audio, sr, min_chunk_length_samples, max_chunk_length_samples, frame_length_samples, overlap_samples):
    """
    Analysiert das Audio, um Chunks basierend auf Energie und Zero-Crossing-Rate zu finden.

    Diese Funktion teilt das Audio in Frames auf und bestimmt, ob jeder Frame still oder aktiv ist.
    Aktive Frames werden zu Chunks aggregiert. Stille Frames signalisieren das Ende eines Chunks,
    wobei die angegebenen minimalen und maximalen Chunk-Längen berücksichtigt werden.
    """
    # Aufteilung des Audios in fünf Segmente zur Berechnung der Schwellenwerte
    segment_length_samples = len(audio) // 5
    # Berechne dynamische Schwellenwerte für Energie und ZCR für jedes Segment
    energy_thresholds, zcr_thresholds = compute_dynamic_thresholds(audio, sr, frame_length_samples, segment_length_samples)

    chunks = []  # Initialisierung der Liste für die gefundenen Chunks
    current_chunk_start = None  # Startpunkt des aktuellen Chunks
    last_valid_end = None  # Endpunkt des letzten gültigen Frames im aktuellen Chunk
    is_previous_frame_silent = False  # Zustand des vorherigen Frames

    # Iteration über das Audio in Frame-Schritten
    for i in range(0, len(audio), frame_length_samples):
        segment_index = i // segment_length_samples
        segment_index = min(segment_index, len(energy_thresholds) - 1)  # Verhindert Index-Überlauf

        # Zuweisung der Schwellenwerte für den aktuellen Frame
        energy_threshold = energy_thresholds[segment_index]
        zcr_threshold = zcr_thresholds[segment_index]

        # Extraktion des aktuellen Frames
        frame = audio[i:min(i + frame_length_samples, len(audio))]
        # Berechnung von Energie und ZCR für den Frame
        frame_energy = np.sum(frame ** 2)
        frame_zcr = np.sum(librosa.zero_crossings(frame, pad=False))

        # Bestimmung, ob der Frame still ist
        is_silent = frame_energy <= energy_threshold and frame_zcr <= zcr_threshold

        # Wenn kein Chunk begonnen hat und der Frame nicht still ist, beginne einen neuen Chunk
            if current_chunk_start is None and not is_silent:
            current_chunk_start = i  # Start eines neuen Chunks
            last_valid_end = i + frame_length_samples

        # Wenn ein Chunk aktiv ist
        if current_chunk_start is not None:
            # Wenn der Frame still ist oder die maximale Chunk-Länge erreicht wurde
            if is_silent and (is_previous_frame_silent or i + frame_length_samples - current_chunk_start >= max_chunk_length_samples):
                # Überprüfe, ob der Chunk lang genug ist, um gespeichert zu werden
                if last_valid_end and last_valid_end - current_chunk_start >= min_chunk_length_samples:
                    chunks.append((current_chunk_start, last_valid_end + overlap_samples))  # Speichern mit Überlappung
                    current_chunk_start = None
            else:
                # Aktualisiere das Ende des aktuellen Chunks
                last_valid_end = i + frame_length_samples
                # Wenn der Frame nicht still ist und die maximale Chunk-Länge erreicht wurde
                if not is_silent and i + frame_length_samples - current_chunk_start >= max_chunk_length_samples:
                    chunks.append((current_chunk_start, min(len(audio), last_valid_end + overlap_samples)))  # Speichern mit Überlappung
                    current_chunk_start = None

        is_previous_frame_silent = is_silent

    # Abspeichern des letzten Chunks, falls notwendig
    if current_chunk_start is not None and last_valid_end - current_chunk_start >= min_chunk_length_samples:
        chunks.append((current_chunk_start, min(len(audio), last_valid_end + overlap_samples)))  # Speichern mit Überlappung

    return chunks

def compute_dynamic_thresholds(audio, sr, frame_length_samples, segment_length_samples):
    """
    Computes thresholds for energy and Zero-Crossing Rate (ZCR).

    The thresholds are determined based on the mean and standard deviation
    of the energy and ZCR within individual segments of the audio signal.
    """

    # Initialization of lists for energy thresholds and ZCR thresholds
    energy_thresholds = []
    zcr_thresholds = []

    # Iterating over the audio signal in segments
    for i in range(0, len(audio), segment_length_samples):
        # Extract the current segment from the audio signal
        segment = audio[i:i + segment_length_samples]
        
        # Calculate the number of frames in the current segment
        n_frames = max(int(len(segment) / frame_length_samples), 1)
        
        # Calculate the energy for each frame in the segment
        energy = np.array([np.sum(segment[j * frame_length_samples:(j + 1) * frame_length_samples] ** 2) for j in range(n_frames)])
        
        # Calculate the Zero-Crossing Rate for each frame in the segment
        zcr = np.array([np.sum(librosa.zero_crossings(segment[j * frame_length_samples:(j + 1) * frame_length_samples], pad=False)) for j in range(n_frames)])
        
        # Add the energy threshold to the array
        # The threshold is the mean plus the standard deviation of the energy in the segment
        energy_thresholds.append(np.mean(energy) + np.std(energy))
        
        # Add the ZCR threshold to the array
        # The threshold is the mean plus the standard deviation of ZCR in the segment
        zcr_thresholds.append(np.mean(zcr) + np.std(zcr))
    
    # Return the lists of thresholds
    return energy_thresholds, zcr_thresholds



def save_chunk(y, sr, start_end_tuple, index, output_dir, file_prefix, output_format):
    """
    Saves an individual chunk of audio to a file.

    Each chunk is saved into its own file within the specified output directory,
    with a filename that reflects its source and index.
    """
    start, end = start_end_tuple
    chunk = y[start:end]
    chunk_audio_path = os.path.join(output_dir, f"{file_prefix}_chunk_{str(index).zfill(4)}.{output_format}")
    sf.write(chunk_audio_path, chunk, sr)

def split_audio(input_file, output_dir, output_format, min_chunk_length_ms, max_chunk_length_ms, frame_length_ms, sr, overlap_samples):
    """
    Splits a single audio file into chunks.

    This function loads an audio file, analyzes it to find coherent chunks, and then
    saves each chunk as a separate file within a dedicated directory for that audio file.
    """
    file_prefix = os.path.splitext(os.path.basename(input_file))[0]
    dedicated_output_dir = os.path.join(output_dir, file_prefix)
    os.makedirs(dedicated_output_dir, exist_ok=True)

    y, _ = librosa.load(input_file, sr=sr)
    chunks = analyze_chunks(y, sr, samples_per_ms(sr, adjust_to_frame_length(min_chunk_length_ms, frame_length_ms)),
                            samples_per_ms(sr, adjust_to_frame_length(max_chunk_length_ms, frame_length_ms)),
                            samples_per_ms(sr, frame_length_ms), overlap_samples)

    with ThreadPoolExecutor() as executor:
        for i, (start, end) in enumerate(chunks):
            executor.submit(save_chunk, y, sr, (start, end), i, dedicated_output_dir, file_prefix, output_format)

    print(f"{len(chunks)} chunks created for {input_file} in {dedicated_output_dir}.")

def process_directory(source_dir, output_dir, output_format='wav', min_chunk_length_ms=5000, max_chunk_length_ms=15000, frame_length_ms=30, sr=16000, overlap_ms=2000):
    """
    Processes all audio files in a directory, splitting them into chunks.

    Each audio file is processed in turn, with its chunks saved to a dedicated subdirectory
    within the output directory.
    """
    overlap_samples = samples_per_ms(sr, overlap_ms)
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.wav'):
            split_audio(os.path.join(source_dir, filename), output_dir, output_format, min_chunk_length_ms, max_chunk_length_ms, frame_length_ms, sr, overlap_samples)

def main():
    source_dir = '/home/lenny/AudioGen/test/wav/'
    output_dir = '/home/lenny/AudioGenFinal/test/wav/'
    process_directory(source_dir, output_dir)

if __name__ == "__main__":
    main()
