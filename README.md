

This the official code for "Machine Unlearning on Trajectory Data: An Experimental Analysis"


## prepare

### requirements

```bash
pip install -r requirements.txt
```


### dataset

put data into `../KyData/TrajData/Porto` like path,

or modify the base path at `_tool/mFile.py`,

where the dataset can be download at [here](https://github.com/derekwtian/TRMMA)


## run

pre-process the dataset and pre-train the model

```bash

python mu/traj/main.py --ini_cpu

python mu/traj/main.py --ini_gpu

```

obtain the result tables

```bash

python mu/traj/plt.py

```

