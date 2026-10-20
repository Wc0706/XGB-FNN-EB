1. Dataset: Train/Dataset
All datasets related to reactions between linear symmetric ethers and CH3, H, OH, HO2 radicals are stored in the "Dataset" folder.
train-M062-0__1 (1-10).txt: Training set for reaction rate constants of linear symmetric ethers with CH3, H, OH, HO2 radicals.
test-M062-0__1 (1-10).txt: Validation set for reaction rate constants of linear symmetric ethers with CH3, H, OH, HO2 radicals.
H/CH3/OH/HO2__test.txt: Test set for reaction rate constants of linear symmetric ethers with CH3, H,  OH, HO2 radicals.
Data format: similarity, four Mordred topological descriptors, temperature and log k (logarithmic rate constant).

2. Neural network training module: train/
This folder contains Python codes and shell scripts for model training.
XGB-FNN.py: Training code of the XGB-FNN hybrid model.

1.Download Anaconda3-2019.07-Linux-x86_64.sh from the Anaconda official website (https://repo.anaconda.com/archive/). In the Linux terminal, run the following command:
$ sh Anaconda3-2019.07-Linux-x86_64.sh
Follow the prompts to complete the installation.

2.Create and activate the virtual environment. In the Linux terminal, run the following commands:
$ conda create -n ML python=3.7.3
ML is the name of the virtual environment. All packages will be installed within this environment.
Activate environment:
$ conda activate ML
Exit environment:
$ conda deactivate

3.Install required packages by the following commands:
$ conda activate ML
$ conda install pytorch==1.4.0 cpuonly -c pytorch
$ conda install scikit-learn==0.23.2
$ conda install matplotlib==3.3.2
$ conda install numpy==1.15.4
$ conda install pandas==1.1.5
$ conda install dill

After environment configuration, open the Linux terminal under the "train" folder and run the following commands to start model training:
$ conda activate ML
$ sh note.sh
$ sh all.work.sh

3. Prediction module: predict/
The "predict" folder stores trained models, input datasets and prediction codes.
20011.fnn / 20040.fnn /20060.fnn: Trained CH3 model files
20011.fnn / 20050.fnn / 20070.fnn: Trained H  model files
20011.fnn / 20053.fnn / 20060.fnn: Trained OH model files
20022.fnn / 20030.fnn / 20071.fnn: Trained HO2 model files
NN.py: Code for model inference