# CANINE for Medical Natural Language Inference on MedNLI data: 

We are interested in Natural Language Inference (NLI) on medical data using CANINE, pre-trained tokenization-free encoder, that operates directly on character sequences—without explicit tokenization and a fixed vocabulary, it is available in this [repo](https://github.com/google-research/language/tree/master/language/canine). We want to predict classify the relation between a hypothesis and a premise as:  Entailement, Contraction or Neutral. We want to perform this task on [MedNLI](https://jgc128.github.io/mednli/), a medical dataset annotated by doctors for NLI. 

## Setup
``` bash
# Clone this repository
git clone https://github.com/loubnabnl/canine-mednli.git
cd canine-mednli/
# Install packages
pip install -r requirements.txt
```

## Data 
Access for the data can be requested [here](https://jgc128.github.io/mednli/). It contains a training, validation and test set with pairs of sentences along with the label of their relation. The data must be placed in the folder `data/` . 

## NLI
Fine-tuned BERT and CANINE models on MedNLI can be downloaded in this [link](), and must be placed in the folder `trained-models/`.

To train a new model on MedNLI you can run the following command
```
python main.py --noisy False --model bert
```

## Noise robustness
Since CANINE doesn't use a fixed vocabulary, it can be intresting to use it on noisy data where there are many out-of-vocabulary words, mispellings and errors. We provide code to generate noisy version of MedNLI for a given noise level. One has to run the following commands

```
cd ./utils
python noisy_data.py --noise_level 0.4
```

To train and evaluate CANINE on noisy data, one can run:

```
python main.py --noisy True
```