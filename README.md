# RP3Net
RP3Net is an AI model for predicting the results of recombinant small-scale protein production in _E. coli_ from the construct sequence. See [the preprint](https://www.biorxiv.org/content/10.1101/2025.05.13.652824v1) and [supplements](https://ftp.ebi.ac.uk/pub/software/RP3Net/) for more details on how it works.

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
