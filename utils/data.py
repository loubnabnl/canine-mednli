import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import CanineTokenizer, AutoTokenizer

def read_nli_data(filename, noisy=False):
    """
    Read MedNLI data and return a DataFrame

    Arguments:
    filename: path to data
    noisy: if True load noisy data (preformatted csv file)

    Returns:
    pandas dataframe with columns: sentence1, sentence2, label
    """
    if noisy:
      nli_data = pd.read_csv(filename)
    else:
      all_rows = []
      with open(str(filename)) as f:
          for i, line in enumerate(f):
              row = json.loads(line)
              all_rows.append(row)
      nli_data = pd.DataFrame(all_rows)
      #keep only raw sentences and label
      nli_data = nli_data[['sentence1', 'sentence2', 'gold_label']]
      nli_data.rename(columns={'gold_label': 'label'}, inplace=True)
      nli_data['label'].replace(['neutral', 'entailment', 'contradiction'], [0, 1, 2], inplace=True)
        
    return nli_data

class NLI_dataset(Dataset):
    
    def __init__(self, data, maxlen=700, name = "canine"):

        self.data = data  
        self.name = name
        
        if self.name == "canine":
            self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c") 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  

        label = self.data.loc[index, 'label']
        return token_ids, attn_masks, token_type_ids, label  