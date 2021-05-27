# Preprocessing

> This code is part of the project “Neural Networks for binary classification on multiple data modalities”.

## Requirements

The code is based on PyTorch and can run both on CPU and GPUs supporting CUDA. In order to run the software, please
install the requirements listed in the `requirements.txt` file.

## Running operations

To run the preprocessing operation, execute `python3 run_pipeline.py` after editing the `operations` list
in `run_pipeline.py`. Operations will be executed sequentially in the order reported in the list. Supported operations
are the following:

| Operation ID                        | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| `preprocess_audio_sequences`        | Preprocesses the audio sequences                             |
| `preprocess_eye_tracking_sequences` | Preprocesses the eye-tracking sequences by fixing invalid eyes and NaN values, applying a cyclic split and possibly collapsing the fixations |
| `preprocess_text`                   | Preprocesses the text                                        |
| `generate_heatmaps`                 | Generates heatmaps from preprocessed eye-tracking sequences  |
| `generate_hsh`                      | Generates Hybrid Scanpath Heatmaps (HSH) from preprocessed eye-tracking sequences |
| `generate_scan_paths`               | Generates scan paths from preprocessed eye-tracking sequences |
| `generate_mfcc_spectrograms`        | Generates MFCC spectrograms as visual representations of audio from raw WAV files |
| `generate_audio_sequences`          | Generates audio sequences from raw audio WAV files           |

Source and destination paths (as well as other operation-specific options) can be specified in the JSON configuration
file of each operation inside the `params/generators` and `params/preprocessors` folders. All paths are relative to the
selected dataset specified as `dataset_name` at `params/dataset.json`.