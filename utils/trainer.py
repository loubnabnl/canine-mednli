import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()
    sm = nn.Softmax(dim=1)
    mean_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            # loss
            mean_loss += criterion(logits, labels).item()
            # f1 score
            probs = sm(logits)
            probs = probs.detach().cpu().numpy()
            predicted = np.argmax(probs, axis=1)
            predictions.append(predicted)
            targets.append(labels.cpu())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    f1score = f1_score(targets, predictions, average='macro')
    acc = accuracy_score(targets, predictions)
    return mean_loss/len(dataloader), f1score, acc


def train_model(net, device, criterion, optimizer, lr, lr_scheduler, train_loader, val_loader, epochs, save_path):
    
    best_loss, best_epoch = np.Inf, 1
    nb_iterations = len(train_loader)
    log_interval = nb_iterations // 5 
    sm = nn.Softmax(dim=1)
    
    for ep in range(epochs):

        net.train()
        train_loss = 0.0 # loss accumulation over train_loader
        running_loss = 0.0 # loss accumulation every log_interval
        correct = 0.0

        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            labels = labels.to(torch.int64)
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            logits = net(seq, attn_masks, token_type_ids)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()

            probs = sm(logits)
            probs = probs.detach().cpu().numpy()
            pred = np.argmax(probs, axis=1)
            correct += (pred==labels.data.cpu().tolist()).sum()

            if (it + 1) % log_interval == 0:  
                acc = 100. * correct/(log_interval * len(labels))
                print(f"Iteration {it+1}/{nb_iterations} of epoch {ep+1} Training Loss : {running_loss/log_interval} Acc: {acc}")
                running_loss = 0.0
                correct = 0.0

        val_loss, f1, val_acc = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        lr_scheduler.step(val_loss)
        print(f"Epoch {ep+1}, Validation Loss : {val_loss} Acc {val_acc} F1 score {f1}")

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_epoch = ep + 1

    # Saving the best model 
    path_to_model=f'{save_path}/model_epoch_{best_epoch}.pt'
    torch.save(net_copy.state_dict(), path_to_model)
    print("Best model saved.")
    del loss
    torch.cuda.empty_cache()