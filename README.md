# DMM: Dynamic Multi-Network Mining of Tensor Time Series


## Requirements
1. Install Python 3.8, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```

(
```
pip install numpy
pip install pandas
pip install matplotlib
pip install sklearn
```
)


## Data Preparation
### Synthetic datasets

```
cd data
python Synthetic.py
```

### Air-quality dataset
Download the Beijing Multi-Site Air-Quality Data Data Set from [UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).
Move them into the data folder.
(/DMM/data/PRSA_Data_20130301-20170228)

### Google dataset
(/DMM/data/google/commerce)


## Usage
### Synthetic experiments
```
python experiment_synthetic.py
```

### Realdata experiments
```
python experiment_realdata.py
```