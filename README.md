# CANINE for Medical Natural Language Inference on MedNLI data

We are interested in Natural Language Inference (NLI) on medical data using CANINE, a pre-trained tokenization-free encoder, that operates directly on character sequences without explicit tokenization and a fixed vocabulary, it is available in this [repo](https://github.com/google-research/language/tree/master/language/canine). We want to predict the relation between a hypothesis and a premise as:  Entailement, Contraction or Neutral using [MedNLI](https://jgc128.github.io/mednli/), a medical dataset annotated by doctors for NLI. We will also use BERT.

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

## Results
Results on clean data:
<div align="center">
 
|Model | Test accuracy | 
|   -   |   -  | 
| BERT |**77.6<sub>±0.6</sub>** |
| CANINE-C  | 73.07<sub>±0.3</sub> |
 
</div>

Results of noise robustness experiments: the left plot correponds to training on clean data and testing on noisy data and the right plot corresponds to the training on noisy data as well
<div align="center">
 
<img width="710" alt="nli_noise2" src="https://user-images.githubusercontent.com/44069155/164975946-1f5c1ec8-b0d4-4e32-8860-62e31327eaa4.png">
 
</div>

For the NLI task on clean MedNLI we get an accuracy of 77.6% using BERT and an accuracy of 73.07% using CANINE. However when we add a noise with probability 0.4 to the test data, the performance of BERT drops to 59.92% while the accuarcy of CANINE drops only to 65.75%. Training the models on noisy data results in an improvement for both models but CANINE is still preferred to BERT with a 1.4% difference in accuracy. This suggests that CANINE can be more suitable for noisy text than BERT, but for clean data we didn't see and advantadge for CANINE in this task.
