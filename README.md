Audio Preprocessing Toolkit for Whisper Fine-Tuning

This toolkit is designed to facilitate the preprocessing of audio files for the purpose of fine-tuning OpenAI's Whisper model. It automates the splitting of long audio recordings into manageable, coherent chunks based on sound energy and Zero-Crossing Rate (ZCR), making it ideal for creating high-quality datasets necessary for machine learning applications.
Overview

The Audio Preprocessing Toolkit simplifies the complex task of audio segmentation, allowing users to customize chunk sizes and overlaps based on their specific needs. It employs robust signal processing techniques to ensure that each audio segment contains meaningful speech content without abrupt beginnings or endings, enhancing the effectiveness of subsequent machine learning models trained with these segments.

    Dynamic Thresholding: Implements dynamic threshold calculation for audio energy and ZCR to adaptively segment audio based on its content.
    
    Adjustable Segment Sizes: Users can specify minimum and maximum chunk lengths, as well as frame length, to tailor the segmentation process to the particularities of their audio data.
    
    Concurrency for Efficiency: Utilizes Python's concurrent.futures module to speed up the processing of audio files by handling multiple files concurrently.
    
    Customizable Sample Rates: Supports various audio sampling rates, making it versatile for different types of audio inputs.

    Output Management: Automatically organizes output audio chunks into dedicated directories for each input file, maintaining a clean and organized file structure.

This toolkit is particularly useful for researchers and developers working on speech recognition and processing tasks who need to preprocess large audio datasets into smaller, more manageable units without losing contextual integrity. It is also an essential step for those preparing datasets for training or fine-tuning the Whisper model.
Future Enhancements

Future updates will include the generation of corresponding metadata files for each processed chunk, and integration with the Whisper fine-tuning process itself, thereby providing an end-to-end solution for custom model training.

Stay tuned for updates and additional features!
