import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

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

    if args.model == 'canine':
        model = CANINE_model()
        maxlen = args.canine_maxlen
    else:
        model = BERT_model() 
        maxlen = args.bert_maxlen

    if args.load_model:
        #load a fine-tuned model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Trained model successfully loaded.")
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