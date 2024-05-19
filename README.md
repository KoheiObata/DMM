# DMM: Dynamic Multi-Network Mining of Tensor Time Series
Implementation of [DMM](https://dl.acm.org/doi/10.1145/3589334.3645461),
Kohei Obata, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai.
The Web Conference 2024, [WWW'24](https://www2024.thewebconf.org/).



## Requirements
1. Install Python 3.8, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```

```
pip install numpy
pip install pandas
pip install matplotlib
pip install sklearn
```


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


## Citation
If you use this code for your research, please consider citing our WWW paper.
```bibtex
@inproceedings{10.1145/3589334.3645461,
author = {Obata, Kohei and Kawabata, Koki and Matsubara, Yasuko and Sakurai, Yasushi},
title = {Dynamic Multi-Network Mining of Tensor Time Series},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645461},
doi = {10.1145/3589334.3645461},
booktitle = {Proceedings of the ACM on Web Conference 2024},
pages = {4117–4127},
numpages = {11},
keywords = {clustering, graphical lasso, network inference, tensor time series},
location = {, Singapore, Singapore, },
series = {WWW '24}
}
```

<!-- ## More on -->
<!-- * Paper: [[ACM DL]](https://dl.acm.org/doi/10.1145/3543507.3583370) [[arXiv]](https://arxiv.org/abs/2303.03789) -->
<!-- * Short Video: [[YouTube]](https://youtu.be/v-E-QjEBwNk) -->
