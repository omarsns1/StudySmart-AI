# Dataset documentation

Download the **Student Performance** dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/320/student+performance

Extract `student-por.csv` from the downloaded archive and put it in this folder. Do not rename the columns.

Citation: Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. Proceedings of FUture BUsiness TEChnology (FUBUTEC).

Portuguese-language course: 649 records; mathematics course: 395 records. These may include overlapping students and must not be combined as independent people. This project uses only Portuguese-language course records.

Target: `G3`, final grade from 0 to 20. Features: `studytime` (weekly study category), `absences` (count), `freetime` (ordinal), `goout` (ordinal), `activities`, `schoolsup`, `famsup`, `internet`, `higher`, `health`; history model additionally uses `G1` and `G2` (prior period grades).

Not available: sleep hours, recreational screen time, attendance percentage. Do not describe these as measured features.

No personal identifying data is requested by this repository. Check UCI terms and cite the dataset when redistributing data.
