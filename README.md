# HubPUG
### (This repository is operational, but the code and documentation is still under development)

This repository accompanies Warfield et al. (2022) ([arXiv:2209.02751](https://arxiv.org/abs/2209.02751))

To recreate the run of Andromeda VII from our paper, make sure that you have a Python environment satisfying the package dependencies in `requirements.txt`, and then copy the shell code below:

```
git clone https://github.com/jackwarfield/and7_tables.git
cd and7_tables
bash fits1.sh
bash fits2.sh
git clone https://github.com/jackwarfield/HubPUG.git
cd HubPUG
python run.py
```
