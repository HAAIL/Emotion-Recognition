# Emotion-Recognition Using EEG Signals. 

Supervised by **Professor**: [Mohammad Ghassemi](https://ghassemi.xyz/)

## About DEAP Dataset. 

1. **Multimodal Dataset**: The DEAP dataset is a multimodal dataset for analyzing human emotional states. It includes physiological signals and self-reported emotional ratings.

2. **Physiological Signals**: The dataset contains electroencephalogram (EEG) data recorded from 32 channels, along with other physiological signals such as galvanic skin response (GSR), heart rate, and facial electromyography (EMG).

3. **Emotional Stimuli**: Participants in the study were exposed to 40 one-minute-long video clips, which served as emotional stimuli. These videos were selected to elicit varied emotional responses.

4. **Research Applications**: The DEAP dataset is widely used in affective computing and neuroscience research, particularly for studies focusing on emotion recognition, human-computer interaction, and the analysis of physiological correlates of emotions.

### What should be used as Input and what can be expected as an output from the EEG data?

**INPUT-X**

This is for a single participant. For example - 's01.dat' file. we have 32 such files  

1. **Video Trials**: Each subject watches a video, and their brain's response (in the form of EEG signals) is recorded. This is like measuring how the brain 'reacts' to what it sees in the video.

2. **Time Series of Length 8064**: A time series is just a set of data points collected over time. In this case, each data point is an EEG reading. The length of 8064 means there are 8064 data points for each trial. This number comes from recording the brain's activity at a rate of 128 times per second `(128 Hz)` for 63 seconds. So, `( 128 X 63 = 8064 )` data points in total for each video trial.

3. **32 Electrodes**: Electrodes are like tiny sensors placed on the subject's head to pick up the EEG signals. Using 32 electrodes means the researchers are getting a comprehensive view of the brain's activity from 32 different points/channels on the scalp and this data is collected for 40 different subjects.

4. There are 40 channels present in the data. However, we are only choosing 32 channels for training our dataset. the rest of the 8 channels are other physiological signals that are not relevant to EEG data. The goal of the project is to use EEG signals to detect the emotional state. For this reason, we are using 32 channels from the data. 

5. **Total Data Generated**: For each subject, they generate data for every combination of the 40 videos and 32 Channels. So, for each subject, the total amount of data points is `( 40 {videos} X 32 {channels} X 8064 {data points/video})`. This is the total data collected from one subject across all the video trials.

In simple terms, this experiment involves recording how the brain of each subject responds to 40 different videos, using 32 sensors on their head, and collecting a very detailed set of data (8064 points) for each video. This results in a large amount of data for each person, reflecting their brain's activity throughout each video.

**OUTPUT-Y**

* The shape of the labels array `(40, 4)` in the DEAP dataset represents a two-dimensional array with specific dimensions:

1. **First Dimension (40)**: This dimension represents the number of stimuli or trials. where participants are exposed to different video clips, this could mean there are 40 different trials or video clips for which the labels are provided.

2. **Second Dimension (4)**: This dimension indicates that there are 4 labels (or features) associated with each trial.
   - **Arousal**: How activated or stimulated the participant is.
   - **Valence**: The pleasantness or unpleasantness of the emotion.
   - **Dominance**: The degree of control or power the emotion conveys.
   - **Liking**: The level of preference or enjoyment.

Therefore, each row in this array corresponds to a specific trial, and each column provides a specific label or rating for that trial, capturing different aspects of the participant's emotional response to the stimulus. 

#### Potential Challenges While dealing with the data. 

1. **High Dimensionality**: The DEAP dataset includes multiple channels of EEG data plus additional physiological signals. Handling such high-dimensional data requires robust feature selection and dimensionality reduction techniques to avoid model overfitting and to enhance computational efficiency.

2. **Signal Preprocessing**: EEG and other physiological signals often contain noise and artefacts (like muscle movement or eye blinks). Proper preprocessing (filtering, artefact removal) is crucial to ensure the quality of the data before analysis.

3. **Subject Variability**: Different individuals may exhibit varying physiological responses to the same emotional stimuli. This inter-subject variability can complicate the training of generalized models.

4. **Generalizability**: Models trained on a specific dataset may not perform well on data from different sources due to differences in data collection protocols, demographic variations, and equipment used. So the main goal of the project is to recognize the patterns of the EEG signals to detect the emotional states.

5. **Class Imbalance**: The authors in the paper, decided to label the the above 4 classes as between high and low values. However, not all emotions were equally balanced between low and high. This means that more videos were rated as high than low for some emotions. For example, 59% of the videos were rated as high arousal, 57% as high valence, and 67% as high liking. The percentages and standard deviations (which show how much variation there is) indicate that the distribution of high and low ratings wasn't even. Some emotions were more likely to be rated as high than others. This kind of imbalance can make it a bit more challenging to analyze the data and train models, as the models might become biased towards the more common category (like high arousal or high liking).





