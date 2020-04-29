# Overview
This project aims to find the most suitable meaning from a pre-defined set of possible meanings to a word given the word in a context

The 2 best performing models of this research are bidirectional LSTM and SVM which are located in the Implementation folder.
# Requirements
* Python <=3.6

# How to run
1. Open terminal and cd into the repository you wish to clone the project
2. Download and extract the WSD dataset in this directory from [here](https://drive.google.com/file/d/1vUqRkMDMreXSclixOYs5MNBB3e5Dqjz-/view?usp=sharing)
3. Clone the git repo: `https://github.com/c1531958/Word-sense-disambiguation.git` or extract the submitted .zip file here
4. cd into the cloned/ extracted repository `cd Word-sense-disambiguation `
5. Install requirements. This also can be done in a virtual environment
```sh
    pip install -r requirements.txt
```

NOTE. Depending on the operating system and Python version, you might need to specify different version of keras, numpy and tensorflow. These requirements were installed on Deepin 20 beta.

6. Run the following command to run the code.
```
    python .\Implementation\lstm.py
    python .\Implementation\svm.py
```

# Background Research/Code Research 
This folder contains other code snippets to test other classifiers such as knn, naive bayes and random forest.

#Misc
This folder contains code snippets to acquire data used in  data analyses and code to plot Pearson Correlation graphs.


