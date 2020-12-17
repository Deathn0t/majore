# MaJore

Project of Advanced Computer Vision on Multiresolution and Multimodal Speech Recognition Transformers

![alt text](https://github.com/Deathn0t/majore/blob/main/assets/model-graph.png "Model Graph")

## Structure of the Repo

The main files are located in the ``notebooks`` folder. Inside this folder you will find 4 notebooks:

1. `Data Inspection`: where we tested how to load the different data we are working with.
2. `Prototyping`: where we developed the different building blocks of the multimodal architecture.
3. `Training of Multimodal Network`: where we assembled the final multimodal network and trained it.
4. `Memory Profiling`: where we experimented the memory usage of our I/O pipeline.

Then, inside the `notebooks/scripts/` you will find the different Python scripts that we developed:

1. Datasets are developed in `load_audio_v1.py`, `load_audio_v2.py`, `load_text.py` and `load_video.py`. In `load_multimodal_data.py` is located the classes responsible of loading audio, video and texts at once.
2. The components of the network architecture are located in `encoders.py`, `decoder.py` and `position_encoder.py`.

## Installation

Assuming you have Anaconda or Miniconda installed.

```console
conda create -n vision python=3.7
conda activate vision
pip install -r requirements.txt
```

The different notebooks can be executed on Colab. To, access the data a form has to be filled on the How2 official repository.

## Results

![alt text](https://github.com/Deathn0t/majore/blob/main/assets/result_training_small_architecture.jpeg "Training of Small Architecture on Colab")
## References

### Papers

* [Multiresolution and Multimodal Speech Recognition with Transformers](https://arxiv.org/abs/2004.14840)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

### Data

* [How2](https://github.com/srvk/how2-dataset)

### Code

* [ResNeXt](https://github.com/facebookresearch/ResNeXt)