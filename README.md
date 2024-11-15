# Suture Planning Experiment Setup
## Installation
Follow this link to install ZED SDK: https://www.stereolabs.com/docs/installation/linux
```
cd RAFT-Stereo/
conda env create -f environment_cuda11.yaml
conda activate raftstereo
python -m pip install cython numpy opencv-python pyopengl
cd /usr/local/zed
python get_python_api.py
python -m pip install autolab_core
sh download_models.sh
```
## Usage
Will run ZED with RAFT-Stereo depth and save left image, right image, and depth image in output folder
```
cd ~/suture_planning
python zed_stereo.py
```