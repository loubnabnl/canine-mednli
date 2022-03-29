# CANINE for Medical Natural Language Inference on MedNLI data

We are interested in Natural Language Inference (NLI) on medical data using CANINE, a pre-trained tokenization-free encoder, that operates directly on character sequences without explicit tokenization and a fixed vocabulary, it is available in this [repo](https://github.com/google-research/language/tree/master/language/canine). We want to predict the relation between a hypothesis and a premise as:  Entailement, Contraction or Neutral. We will perform this task on [MedNLI](https://jgc128.github.io/mednli/), a medical dataset annotated by doctors for NLI. 

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
To use our fine-tuned BERT and CANINE models on MedNLI, you can download the weights in this [link](), and you should place them in the folder `trained-models/`.

To train a new model on MedNLI you can run the following command
```
python main.py --model canine --noisy False
```

## Noise robustness
Since CANINE doesn't use a fixed vocabulary, it can be intresting to use it on noisy data where there are many out-of-vocabulary words, mispellings and errors. We provide code to generate noisy versions of MedNLI for a given noise level, by adding, deleting replacing and swapping letters in the words. You can run the following commands:

```
cd ./utils
python noisy_data.py --noise_level 0.4
```

To train and evaluate CANINE on noisy data, you can run:

```
python main.py --model canine --noisy True 
```