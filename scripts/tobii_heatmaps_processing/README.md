# Renaming of heatmaps exported using Tobii

## Structure of the directories

The files in the folder containing the file `__main__.py` must be organised according to the following structure of directories: 

```
├── metadata
└── tasks
    ├── cookie_theft
    │   └── raw
    ├── memory
    │   └── raw
    └── reading
        └── raw
```

The `metadata` folder must include a CSV file mapping file names from each task to PIDs.   

## Usage

1. Tune the following parameters at the beginning of the `main` function at `__main__.py`:
   - `path_to_filenames_to_pid_map`: the path to the CSV file mapping file names from each task to PIDs (e.g. `/metadata/heatmaps_name_to_pid.csv`) 
   - `positive_label` and `negative_label`:  the name of the positive and negative labels for the current classification task which will be used to split the data in the proper directories (e.g. `alzheimer` and `healthy`)
   - `tasks`: a list of names of tasks matching the names of the directories in `tasks` (e.g. `["cookie_theft", "memory", "reading"]`)
2. Run with `python3 __main__.py`

## Test environment

Tested on Ubuntu 18.04 and Python 3.6