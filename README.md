# RP3Net

[![CI](https://github.com/RP3Net/RP3Net/actions/workflows/python-app.yml/badge.svg)](https://github.com/RP3Net/RP3Net/actions/workflows/python-app.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RP3Net/RP3Net/blob/main/rp3_colab.ipynb)
[![DOI:10.1101/2025.05.13.652824](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://doi.org/10.1101/2025.05.13.652824)
[![PyPI - Version](https://img.shields.io/pypi/v/RP3Net)](https://pypi.org/project/RP3Net/)

RP3Net is an AI model for predicting the results of recombinant small-scale protein production in _E. coli_ from the construct sequence. See [the preprint](https://www.biorxiv.org/content/10.1101/2025.05.13.652824v1) and [supplements](https://ftp.ebi.ac.uk/pub/software/RP3Net/) for more details on how it works.

# Try it out
The simplest way to run the model inference is to open the [Colab notebook](https://colab.research.google.com/github/RP3Net/RP3Net/blob/main/rp3_colab.ipynb), paste the sequeces in FASTA format into the first cell and hit `Runtime -> Run All`.

## Docker
Another way to try out the model without disclosing the sequences is via the [Docker image](https://hub.docker.com/r/rp3net/rp3net): `docker pull rp3net/rp3net`. The image contains the binary installation of the package, the checkpoint, a Jupyter server and the notebook. It supports CUDA. To run the docker contaier from the command line, using mounts to send the data in and out:
```
docker run -v /path/to/my/files:/mnt/rp3 rp3net/rp3net rp3 -p rp3net_v0.1_d.ckpt -f /mnt/rp3/sequences.fasta.gz -o /mnt/rp3/scores.csv.gz --log_file /dev/null
```
To interact with the container via the notebook, run `docker run -p 8888:8888 rp3net/rp3net jupyter lab`, open http://localhost:8888 in the browser and open `rp3_colab.ipynb`. There is no need to download the checkpoint and install the dependencies.

# Checkpoints
* https://ftp.ebi.ac.uk/pub/software/RP3Net/v0.1/checkpoints/

# Inference
## Installation
```
pip install RP3Net
```

## Command line
Simple usage:
```
rp3 -p <path_to_checkpoint_file> -f <in_fasta_file> -o <out_csv_file>
```
The `out_csv_file` will contain the dataframe with the ids from the `in_fasta_file` and the predicted probabilities of successfull recombinant small-scale protein production in _E. coli_.
For more information on the command line arguments, type `rp3 -h`.

## Python interface
```python
import RP3Net as rp3
m = rp3.load_model(rp3.RP3_DEFAULT_CONFIG, '/path/to/checkpoint')
scores = m.predict(['PRTEINWQENCE', 'PRTEIN', 'SQWENCE'])
print(scores)
# tensor([0.4223, 0.4134, 0.4165])
score_map = m.predict({'seq1': 'PRTEINWQENCE', 'seq2': 'PRTEIN', 'seq3': 'SQWENCE'})
print(score_map)
# {'seq1': 0.4223055839538574, 'seq2': 0.41336774826049805, 'seq3': 0.4165498912334442}
```

The `load_model` function returns the model object that can be used directly for prediction (`predict`), and is otherwise a fully functional implementation of a [Pytorch module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), so can be used for computing gradients and training as well. The `predict` method accepts either a list of sequences as strings, or a dictionary of sequences keyed by their ids. The return type depends on the input, and is either a one-dimensional tensor or a dictionary of floats. In the former case the order of the scores corresponds to the order of the input sequences, in the latter case the dictionary is keyed by the sequence ids.

## Performance and resource usage
The command line verstion on a modern CPU (base frequency 2.6 GHz) for a batch of 16 constructs with length under 500aa runs in about 3 minutes, using under 5Gb of RAM.

# Training
Note that installation for inference does not bring in the libraries that are used for training.

## Installation
```
pip install 'RP3Net[training]'
```

## Command line
```
rp3_train fit -c <training_config_file>
```
Examples of trainer cofigs can be found under `config` folder. Training is managed by [Pytorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html); more information can be found by typing `rp3_train -h`

## Training data
* https://ftp.ebi.ac.uk/pub/software/RP3Net/v0.1/data/
