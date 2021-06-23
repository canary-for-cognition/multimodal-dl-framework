# Dataset

This dataset is based on a study involving a base group of 189 participants, for whom data was recorded in terms of
eye-tracking and audio. All the information related to the experiment is stored at `metadata_participants_log.csv`,
which also provide the ground truth for each participant (i.e., whether it is considered a `Patient` or
a `Healthy Control`).

## Tasks

Data for Alzheimer’s prediction was collected based on the following tasks that patients performed during the
experiments.

| Name              | Description                                                  | ID in dataset       | ID in metadata |
| ----------------- | ------------------------------------------------------------ | ------------------- | -------------- |
| Cookie Theft      | Describing a picture (involving some kids stealing a cookie from a jar) while looking at it | `cookie_theft`      | `CookieTheft`  |
| Reading           | Reading aloud a text                                         | `reading`           | `Reading`      |
| Memory            |                                                              |                     | `Memory`               |
| Pupil calibration | Staring at a cross place at the centre of the screen to calibrate the eye-tracker | `pupil_calibration` | `PupilCalib`   |

## Data

### Eye-tracking

> Resources:
> * https://www.tobiipro.com/learn-and-support/learn/eye-tracking-essentials/how-do-tobii-eye-trackers-work/
> * https://www.tobiipro.com/learn-and-support/learn/eye-tracking-essentials/types-of-eye-movements/

Data recorded using the eye-tracker is provided by participant, not by task. For each participant, data related to a
specific task can be identified using the timestamps provided in `metadata/timestamps.csv`.

#### Sequences

###### Selected features

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

### Images

The following image representations were generated based on eye-tracking data.
