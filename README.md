# light_prop_proc
'gen' folder includes 'light_prop_gen' python script for reference. Will not run as part of docker-compose. Typically, generate data independently as this can take hours. 

'light_prop_data' assumes data in 'opt/gen_data/raw/train' and 'opt/gen_data/raw/test' to generate tfrecords. Assumes 'test' and 'train' directories contain sub-directories '0', '1', and '2' to generate data. Subdirectories used for assigning truth labels to data.

'light_prop_data' takes a 'test' and 'train' folder with data files generated from 'light_prop_gen' and converts them to TFRecords for prepcessing. This will allow the 'light_prop_proc' script perform batching and prefetching of the data.

'light_prop_proc' creates ANN model from TFRecords. Can specify 'rnn', 'cnn', or 'crnn' with the 'model_type' variable within the script.

'archive' folder contains images of model architecture, input data images, and save '.h5' files of previously generated models.

'light_prop_heatmap' will create heatmaps based on the first CNN layer for model analysis.