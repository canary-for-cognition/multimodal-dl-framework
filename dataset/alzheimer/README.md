# Dataset

This dataset is structured according to the requirements of the project “Neural Networks for binary classification on multiple data modalities”.

## Modalities

### Eye-tracking

#### Sequences

###### Features

| Index | Feature                  | Type          |
| ----- | ------------------------ | ------------- |
| 0     | GazePointLeftX (ADCSpx)  | Gaze          |
| 1     | GazePointLeftY (ADCSpx)  | Gaze          |
| 2     | GazePointRightX (ADCSpx) | Gaze          |
| 3     | GazePointRightY (ADCSpx) | Gaze          |
| 4     | GazePointX (ADCSpx)      | Gaze          |
| 5     | GazePointY (ADCSpx)      | Gaze          |
| 6     | GazePointX (MCSpx)       | Gaze          |
| 7     | GazePointY (MCSpx)       | Gaze          |
| 8     | GazePointLeftX (ADCSmm)  | Gaze          |
| 9     | GazePointLeftY (ADCSmm)  | Gaze          |
| 10    | GazePointRightX (ADCSmm) | Gaze          |
| 11    | GazePointRightY (ADCSmm) | Gaze          |
| 12    | DistanceLeft             | Head distance |
| 13    | DistanceRight            | Head distance |
| 14    | PupilLeft                | Pupil         |
| 15    | PupilRight               | Pupil         |
| 16    | FixationPointX (MCSpx)   | Fixation      |
| 17    | FixationPointY (MCSpx)   | Fixation      |
| 18    | ValidityLeft             | Validity \*   |
| 19    | ValidityRight            | Validity \*   |

\* *(0 means that the eye was confidently tracked, 4 means that the eye was lost)*
