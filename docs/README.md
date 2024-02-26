# Retina optimized for prediction across animal species
This repository contains the code of the spiking retinal model and the code for reproducing all the results reported in our preprint.

## Installing dependencies
Install all required dependencies and activate the retina environment using conda.
```
conda env create -f environment.yml
conda activate retina
```

## Model training
All the models can be trained by running the following script:
```
python scripts/model/fits.py
```
Alternatively, model training can be customized by directly running the training script with custom command-line arguments:
```
python scripts/model/train.py
```
Example command-line arguments modify the model architecture:
- ```--n_in``` the number of spiking units.
- ```--rf_size``` the receptive field spatial size in pixels.
- ```--photo_noise``` the added level of noise to the input clips.

or the training process:
- ```--n_epochs``` the number of training epochs.
- ```--batch_size``` the batch size.
- ```--lr``` the learning rate.

See the training script for a full list of command-line arguments.

### Downloading pre-trained models
Instead of training the models from scratch, pre-trained models can be found under the releases of this repository.

### Natural movie training data
We recorded various natural movies for model training. The dataset can be downloaded from: https://figshare.com/articles/dataset/Natural_movies/24265498. Make sure to set the ```dataset_path``` variable in the ```scripts/model/fits.py``` script to point to the dataset.

## Reproducing paper results
All the figures in the paper can be reproduced using the notebooks found in the ```notebooks``` directory.

### Reconstructing images from spike rates and latencies
To build the noisy datasets run the ```scripts/recon/build_noise_dataset.py``` script. Finally, launch the ```scripts/recon/fit_noise_dataset.py``` script to fit the readouts for reconstructing the input images using the rate and latency spike codes.

### Downloading retina response datasets
All instructions for downloading and processing the retinal recordings can be found in the ```data/neural``` directory.

### Predicting retinal responses from the spiking models
The script to launch each cross-validated retinal dataset fit can be found in the ```scripts/fits/build_and_fit``` directory. Scripts for determining the best performing spatial scale for each dataset are found in the ```scripts/fits/search``` directory.
