# Monocular-Depth-Estimation

## How to run the experiments (after downloading the files according to `../downloadlink.txt`)

Either pip install the requirements or preferably open the folder as a docker container using the Dockerfile in `.devcontainer/Dockerfile`, the requirements will be installed automatically.

Move into the folder of this readme, if you havent done so already (the folder with Code, data, eval, Images) and run:

```
python Code/main.py
```

Then you are prompted with 

```
(G)LPN or (N)eWCF predictor:
```

and 

```
(K)itti or (N)yu dataset:
```

Please choose the required Model and Dataset (enter either g or n and then either k or n).

Then the experiments should run multithreadedly. If the data is not completely downloaded, please modify the drives array in `kitti()`.

## Run data analysis

To run the data analysis refer to the notebook in `./Code`.