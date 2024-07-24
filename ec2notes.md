# EC2 Configuration

This file describes additional configuration which must be performed.

## Ensure the NVIDIA GPU is available

Install the NVIDIA drivers as follows:

**Note:** As part of the initial machine configuration, the following
steps have already been run. They're left here for completeness:

```
sudo dnf update -y
sudo dnf install -y dkms kernel-devel kernel-modules-extra
```

(A reboot is required after the instructions have been executed)

### Set up additional details:

```
sudo dnf config-manager -add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf clean expire-cache
sudo dnf module install -y nvidia-driver:latest-dkms
```

When these steps have completed, reboot again.

### Additional requirements

First, make sure that the NVIDIA configration is accurate. Run the following:
`nvidia-smi`

If you see an error, then something "bad" happened, and this should first be fixed.
If all is well, however, proceed to install additional prerequisites:

```
sudo dnf install -y g++ ninja-build pip git python3-devel mesa-libGL
```

Also, edit `.bashrc` and add the following:

```
alias python='python3'
alias pip='pip3'
```

**NOTE:** You may want to open a new shell at this point, so the alias values
are picked up.

## Install Detectron2 prerequisites

Install the following:

```
pip install torch torchvision
pip install opencv-python
pip install wheel
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Optional - Install Jupyter Notebook support

Install the following:

```
pip install notebook
```

The machine should now be properly configured.