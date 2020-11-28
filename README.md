# MaJore

Project of Advanced Computer Vision on Multiresolution and Multimodal Speech Recognition Transformers

![alt text](https://github.com/Deathn0t/majore/blob/main/assets/model-graph.png "Model Graph")

## Installation

### On MacOS X

```console
conda create -n vision python=3.6
conda activate vision
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git checkout v0.3.1
git submodule update --init
pip install pyyaml
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

To test the pytorch installation:

```console
cd ..
python -c "import torch; print(torch.__version__)"
```

Then install the "How2" baseline codebase:

```console
git clone https://github.com/srvk/how2-dataset.git
cd how2-dataset/baselines/code/
python setup.py install
```

## References

### Papers

* [Multiresolution and Multimodal Speech Recognition with Transformers](https://arxiv.org/abs/2004.14840)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

### Data

* [How2](https://github.com/srvk/how2-dataset)

### Code

* [ResNeXt](https://github.com/facebookresearch/ResNeXt)