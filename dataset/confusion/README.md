# Dataset

This dataset is structured according to the requirements of the project “Neural Networks for binary classification on multiple data modalities”.

### Directory structure

```
dataset
|
└── unaugmented
|   └── confused
|	└── not_confused
|	└── fixation_based_confused
|	└── fixation_based_not_confused
|	└── confused_PUMP_OpenEnded
|
└── augmented
|	└── confused
|	└── not_confused
|	└── confused_adjusted_pupils
|	└── not_confused_adjusted_pupils
|
└── full_scanpath_dataset
|
└── short_scanpath_dataset
|
└── emdat_features
|	└── VC_features_openended_vizaoi_5000.csv
|	└── VC_features_vizaoi_5000.csv
|
└── grouped_10_fold_split_list.pickle
```

#### EMDAT features

The output from running each confusion event through the *Eye Movement Data Analysis Toolkit* (EMDAT); this creates engineered features for each item.

| Folder                                  | Description                                                  |
| --------------------------------------- | ------------------------------------------------------------ |
| `VC_features_openended_vizaoi_5000.csv` | EMDAT output for raw eye tracking data corresponding to confusion events obtained during the open ended tasks |
| `VC_features_vizaoi_5000.csv`           | EMDAT output for raw eye tracking data corresponding standard tasks |

### Data item naming key

`userID_valueChartOrientation_taskAbbreviation_taskNumber.pkl`

## Modalities

### Eye-tracking

#### Sequences

###### Features

| Index | Feature                 | Type          |
| ----- | ----------------------- | ------------- |
| 0     | GazePointLeftX..ADCSpx  | Gaze          |
| 1     | GazePointLeftY..ADCSpx  | Gaze          |
| 2     | GazePointRightX..ADCSpx | Gaze          |
| 3     | GazePointRightY..ADCSpx | Gaze          |
| 4     | GazePointX..ADCSpx      | Gaze          |
| 5     | GazePointY..ADCSpx      | Gaze          |
| 6     | GazePointX..MCSpx       | Gaze          |
| 7     | GazePointY..MCSpx       | Gaze          |
| 8     | GazePointLeftX..ADCSmm  | Gaze          |
| 9     | GazePointLeftY..ADCSmm  | Gaze          |
| 10    | GazePointRightX..ADCSmm | Gaze          |
| 11    | GazePointRightY..ADCSmm | Gaze          |
| 12    | DistanceLeft            | Head distance |
| 13    | DistanceRight           | Head distance |
| 14    | PupilLeft               | Pupil         |
| 15    | PupilRight              | Pupil         |
| 16    | ValidityLeft            | Validity \*   |
| 17    | ValidityRight           | Validity \*   |

\* (0 means eye is tracked, 4 means that eye is lost)
