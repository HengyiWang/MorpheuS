# Installation

Please follow the instructions below to install the repo and dependencies:

```bash
git clone --recursive https://github.com/HengyiWang/MorpheuS.git
cd MorpheuS
```



### Install the environment

```sh
# Create conda environment
conda create -n morpheus python=3.9
conda activate morpheus

# Install the pytorch first (Please check the cuda version)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install NeRFAcc  (Again, please check the cuda version)
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

# Install dependency
pip install -r requirements.txt
```



### Build extension

#### Encoders

```sh
# Build extension (encoders from kiui)
bash scripts/install_ext.sh
```



#### Open3D (headless mode)

Note here we use headless mode of open3d for rendering mesh video & evaluation, if you do not need it, you can download a regular open3d package. See https://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html for details about headless mode rendering.

```sh
# Requirement
sudo apt-get install libosmesa6-dev

# headless mode installation
bash scripts/install_o3d.sh
```



### Download pre-trained weights

```sh
# Download Zero123 weights
bash scripts/download_weights.sh
```

