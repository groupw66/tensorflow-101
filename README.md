# TensorFlow Tutorial
Learning by playing with http://playground.tensorflow.org/

## Installation Guide
1. Install Anaconda Python 3.x version (https://www.continuum.io/downloads)
2. Install TensorFlow with Anaconda (https://www.tensorflow.org/install)
3. Install matplotlib `pip install --ignore-installed --upgrade matplotlib`

For macOS you can run this script after install Anaconda
```bash
conda create -n tensorflow --yes
source activate tensorflow
yes | pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.2.0-py3-none-any.whl
yes | pip install --ignore-installed --upgrade matplotlib
```
