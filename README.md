# ECG Arrhythmia classification using CNN:
Each ECG beat is considered separately and classified into one of the arrhythmia types. In this approach 2-D CNN is used and 2-D CNN takes images as input, due to that ECG beats are segmented and plotted using matplotlib, further this image is modified and resized to (128,128) using openCV.  
The dataset used is oldest and highly accurate dataset on ECG signal:https://physionet.org/content/mitdb/1.0.0/. It has a separate annotation file which mentions location and type of each beat.
