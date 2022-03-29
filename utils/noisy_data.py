"""script to generate a noisy version of MedNLI"""
import random
import argparse
from data import read_nli_data

def edits(word):
    """Create a set of edits that are one edit away from `word`
    source code: https://norvig.com/spell-correct.html
    """

    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def noisy_sentence(sentence, level = 0.2):
    """replace each word in sentence with a random edit
     with probability=level"""
    L = sentence.strip().split(' ')
    for i, word in enumerate(L):
        if random.random() < level:
        #choose a mispelling for the word randomly
            noise = edits(word)
            L[i] = random.choice(list(noise))
    sentence = ' '.join(L)
    return sentence

def mispell(row, level = 0.2):
  row['sentence1'] = noisy_sentence(row['sentence1'], level)
  row['sentence2'] = noisy_sentence(row['sentence2'], level)
  return row

def generate_noisy_data(args):
    df_train = read_nli_data(args.train_path)
    df_val = read_nli_data(args.val_path)
    df_test = read_nli_data(args.test_path)

    print(f'Generating noisy data with level {args.noise_level} ...')
    df_train_noisy = df_train.apply(lambda row: mispell(row, args.noise_level), axis=1)
    df_val_noisy = df_val.apply(lambda row: mispell(row, args.noise_level), axis=1)
    df_test_noisy = df_test.apply(lambda row: mispell(row, args.noise_level), axis=1)

    print('Saving noisy data as csv...')
    df_train_noisy.to_csv(f'{args.save_path}/df_train_noisy.csv', index=False)
    df_val_noisy.to_csv(f'{args.save_path}/df_val_noisy.csv', index=False)
    df_test_noisy.to_csv(f'{args.save_path}/df_test_noisy.csv', index=False)

if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/mli_train_v1.jsonl", 
        help="Path to the train data")
    parser.add_argument("--val_path", type=str, default="../data/mli_dev_v1.jsonl", 
        help="Path to the validation data")
    parser.add_argument("--test_path", type=str, default="../data/mli_test_v1.jsonl", 
        help="Path to the test data")
    parser.add_argument("--save_path", type=str, default="../data", 
        help="Path to save the generated noisy data")
    parser.add_argument("--noise_level", type=float, default=0.4,
        help="Noise level of the new data")
    
    generate_noisy_data(parser.parse_args())