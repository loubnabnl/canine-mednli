import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
import os

from utils.data import read_nli_data, NLI_dataset
from utils.utils import set_seed
from utils.trainer import train_model, evaluate_loss
from models import CANINE_model, BERT_model

BATCH_SIZE = 16  
SEED = 1


def train(args):
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    if args.model.lower() == 'canine':
        model = CANINE_model()
        maxlen = args.canine_maxlen
    else:
        model = BERT_model() 
        maxlen = args.bert_maxlen

    if args.load_model:
        #load a fine-tuned model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Fine-tuned model successfully loaded.")
    else: 
        #train a new model
        model.to(device)

        df_train = read_nli_data(args.train_path, args.noise)
        df_val = read_nli_data(args.val_path, args.noise)

        train_set = NLI_dataset(df_train, maxlen)
        val_set = NLI_dataset(df_val, maxlen)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        num_training_steps = args.epochs * len(train_loader)  
        t_total = len(train_loader) * args.epochs  
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup, num_training_steps=t_total)
                
        print("Training begins...")
        train_model(model, device, criterion, optimizer, args.lr, lr_scheduler, train_loader, val_loader, args.epochs, args.save_path)
        
    print('Evaluation on the test set...')
    df_test = read_nli_data(args.train_path, args.noise)
    test_set = NLI_dataset(df_test, maxlen)
    test_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)
    loss, f1, acc = evaluate_loss(model, device, criterion, test_loader)
    print(f"Test Loss: {loss} F1 score: {f1} Accuracy: {acc}")

if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="canine", 
        help="Choose BERT or CANINE")
    parser.add_argument("--canine_maxlen", type=int, default=700, 
        help="Maximum length of inputs to CANINE")
    parser.add_argument("--bert_maxlen", type=int, default=256, 
        help="Maximum length of inputs to BERT")
    parser.add_argument("--train_path", type=str, default="./data/mli_train_v1.jsonl", 
        help="Path to the train data")
    parser.add_argument("--val_path", type=str, default="./data/mli_dev_v1.jsonl", 
        help="Path to the validation data")
    parser.add_argument("--test_path", type=str, default="./data/mli_test_v1.jsonl", 
        help="Path to the test data")
    parser.add_argument("--load_model", type=bool, default=False, 
        help="If True load already fine-tuned models")
    parser.add_argument("--model_path", type=str, default="./trained-models/canine_weights.pth", 
        help="Path to save the fine-tuned model to load")
    parser.add_argument("--save_path", type=str, default="./trained-models", 
        help="Path to save the new fine-tuned model")
    parser.add_argument("--noise", type=bool, default=False,
        help="If True use noisy MedNLI data")
    parser.add_argument("--noise_path", type=bool, default="./data",
        help="Path to noisy data if generated")
    parser.add_argument("--epochs", type=int, default=10,
        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5,
        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.1,
        help="warmup steps")
    train(parser.parse_args())