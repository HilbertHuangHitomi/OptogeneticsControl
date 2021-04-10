# OptogeneticsControl


## started at 2020.09.23
made by Yicong Huang, SCUT.


## update

### 2020.10.29 update
- split several modules for convenience.
- directly running recognition without initialization is possible.
- optimized file structure.
- use 2048 points (about 4.1s) achieve acc of 97%+.

### 2020.10.30 update
- visualization is available.
- saving and loading model is available.
- merge FeatureCalculate and Model.
- simplify Model.
- modify the auto-saving script.
- use 1024 points (about 2.05s) to achieve acc of 98%+.

### 2020.11.03 update
- remove immediate visualization.
- use 2048 points (about 4.1s) to preserve stability with acc of 99%+.
- split hyperparameters to an additional file.
- add maximum value of subbands as new features.
- rename files.
- optimize functions.
- move EMA to Model.
- simplify MLP.
- move DatasetGenerate to Model.
- create floders every 6h and save the overall record.

### 2020.11.04 update
- modify ctime of overall.txt to agree with spike2.
- optimize functions.

### 2020.11.08 update
- use additional seizure data to achieve acc of 95%+.
- optimize functions.

### 2020.11.09 update
- take distribution into consideration.
- add the first subband into features.
- achieve acc of 95%+.

### 2020.11.11 update
- add TestDEMO for data testing.

### 2020.11.12 update
- use a MLP with 2 hidden layer to achieve acc of 99%+.
- optimize save/load function.
- try to read USB without test.

### 2020.11.24 update
- connect inper device successfully.
- try building serial communication but does not work.

### 2020.12.01 update
- serial communication completed.
- close loop system completed.
- add starting delay for observing.
- add events recording.
- add overall report.
- optimize functions.
- optimize file structures.

### 2020.12.03 update
- add solutions for NaN.
- modify delay.
- modify ema coefficient as 0.9.
- modify INPERcontrol to avoid stopping every 284 commands.
- add manually combine overall data to DataIO.

### 2020.12.05 update
- bagging to improve model.
- optimize device control mechanism.
- optimize functions.
- optimize file structures.
- add maximumvalue as a new feature.

### 2020.12.06 update
- reset flag when engage the device to fix bugs.
- move StartFlag to Model.
- optimize TestDEMO to agree with the configuration.
- reorganize dataset.
- adjust combination as liner : reluMLP : tanhMLP = 2:1:1.
- achieve acc of 99%+.

### 2020.12.07 update
- adjust EMA as 0.3 and waiting_times as 2.
- visualize testing prediction.
- fix bugs in StartFlag.
- use PrettyTable to evaluate models.

### 2020.12.08 update
- optimize functiions.
- fix bugs in DataIO.

### 2020.12.21 update
- fix bugs in DataIO.
- use Yield instead of while in Spike2AutomaticallySave.S2S to synchronize.
- optimize SavingTrace in DataIO.

### 2021.01.05 update
- customed model for each subject.
- modify the code structure.

### 2021.01.27 update
- unify and delete customed models.
- fix data saving after activating devices.

### 2021.01.28 update
- fix bugs.

### 2021.01.29 update
- adjust parameters.
- fix bugs.
- optimize feature extractors.

### 2021.02.16 update
- adjust parameters.
- customed model for each subject.
- optimize data structure.
- optimize data reading functions.

### 2021.02.19 update
- adjust parameters.
- adjust models.
- optimize functions.

### 2021.02.20 update
- fix INPER bugs.
- optimize file structures.
- adjust data reading.

### 2021.02.21 update
- optimize DatasetGenerate.
- use ascending threshold.
- combine TestDEMO and Model.

### 2021.03.04 update
- add imblearn for over-sampling.

### 2021.03.08 update
- fix bugs.
- redesign FeatureCalculate with pyentrp.
- redesign models.

### 2021.03.17 update
- fix bugs in FeatureCalculate.

### 2021.03.21 update
- delete shannon entropy as direct features.
- adjust parameters.

### 2021.03.25 ReBurn
- use inception in pytorch instead of traditional ML.
- improve associated interface.
- add use os to specify paths.
- add training support for linux server.
- adjust file structure.
- compute sample_size automatically.
- adjust parameters.

### 2021.03.26 update
- fix issues working on cpu.

### 2021.04.08 update
- adjust parameters.
- fix bugs in the loss function.

### 2021.04.09 update
- adjust parameters.
- delete normalization to prevent 0 input.

### 2021.04.10 update
- delete inputdada after prediction to save memory.

## immediate exsisting issues
- speed of reading and processing large scale data.