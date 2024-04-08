import torch
import numpy as np
import time, os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@torch.no_grad() 
def evaluate(model, criterion, data_loader, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    for i, (batch, labels) in enumerate(tqdm(data_loader)):
        logits = model(batch)
        loss = criterion(logits.squeeze(0), labels.to(device)) 
        losses.append(loss.item())
        _, preds = torch.max(logits, dim=2)
        preds = preds.squeeze(0).cpu().numpy()
        labels = labels.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())


    accu = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') 
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    model.train()
    return np.mean(losses), accu, f1, precision, recall


def train(epochs, device, batch_size, model, criterion, optimizer, val_dataloader, train_dataloader, use_data, save_dir, best_dir, writer):
    print("Started training")
    print(sum(p.numel() for p in model.parameters())/1e6, "million parameters")
    global_step = 0
    start_time = time.time()
    best_accuracy = float('-inf')
    plot_data = []
    for epoch in range(1, epochs + 1):
        for step, (batch, labels) in enumerate(tqdm(train_dataloader)):
            model.train()
            global_step += 1
            
            optimizer.zero_grad()
            logits = model(batch)  
            loss = criterion(logits.squeeze(0), labels.to(device)) 
            writer.add_scalar("Loss/train", loss, global_step)
            loss.backward() 
            optimizer.step() 
            
            # Log training metrics
            if global_step % 50 == 0:
                eval_loss, eval_accu, eval_f1, eval_p, eval_r = evaluate(model, criterion, val_dataloader, device) 
                loss = criterion(logits.squeeze(0), labels.to(device)) 
                print("**"*20)
                print(f"Epoch {epoch} Global step {global_step}: Train loss {loss.item():.4f}, Val loss {eval_loss:.4f}, ")
                print(f"Validation set Accuracy: {eval_accu:.4f}, F1: {eval_f1:.4f}, Precision: {eval_p:.4f}, Recall: {eval_r:.4f}")
                print(f"Time since beginning: {(time.time() - start_time):.2f} s. Samples Gone Through: {step*batch_size}, ")
                print("**"*20)
                writer.add_scalar("Loss/val", eval_loss, global_step)
                writer.add_scalar("Val_Scores/accuracy", eval_accu, global_step)
                writer.add_scalar("Val_Scores/F1", eval_f1, global_step)
                writer.add_scalar("Val_Scores/Precision", eval_p, global_step)
                writer.add_scalar("Val_Scores/Recall", eval_r, global_step)
                save_param_path = os.path.join(save_dir, f"last_checkpoint.pth")
                torch.save(model.state_dict(), save_param_path)

                # Save best model
                if eval_accu > best_accuracy:
                    best_accuracy = eval_accu
                    save_param_path = os.path.join(best_dir, f'model_best_{best_accuracy:.4f}acc_{use_data}_epoch{epoch}_{model.num_heads}heads_{model.num_blocks}blocks_{batch_size}batch.pth')
                    torch.save(model.state_dict(), save_param_path)
                    print(f"New best model saved with accuracy: {best_accuracy}")
            
def full_train(epochs, device, batch_size, model, criterion, optimizer, val_dataloader, train_dataloader, use_data, save_dir, best_dir, writer):
    print("Started full training")
    print(sum(p.numel() for p in model.parameters())/1e6, "million parameters")
    global_step = 0
    start_time = time.time()
    best_accuracy = float('-inf')
    plot_data = []
    for epoch in range(1, epochs + 1):
        for step, (batch, labels) in enumerate(tqdm(train_dataloader)):
            model.train()
            global_step += 1
            
            optimizer.zero_grad()
            logits = model(batch)  
            loss = criterion(logits.squeeze(0), labels.to(device)) 
            writer.add_scalar("Loss/train", loss, global_step)
            loss.backward() 
            optimizer.step() 
            
            # Log training metrics
            if global_step % 50 == 0:
                test_loss, test_accu, test_f1, test_p, test_r = evaluate(model, criterion, val_dataloader, device) 
                loss = criterion(logits.squeeze(0), labels.to(device)) 
                print("**"*20)
                print(f"Epoch {epoch} Global step {global_step}: Train loss {loss.item():.4f}, Val loss {eval_loss:.4f}, ")
                print(f"Validation set Accuracy: {eval_accu:.4f}, F1: {eval_f1:.4f}, Precision: {eval_p:.4f}, Recall: {eval_r:.4f}")
                print(f"Time since beginning: {(time.time() - start_time):.2f} s. Samples Gone Through: {step*batch_size}, ")
                print("**"*20)
                writer.add_scalar("Loss/test", test_loss, global_step)
                writer.add_scalar("Test_Scores/accuracy", test_accu, global_step)
                writer.add_scalar("Test_Scores/F1", test_f1, global_step)
                writer.add_scalar("Test_Scores/Precision", test_p, global_step)
                writer.add_scalar("Test_Scores/Recall", test_r, global_step)
                save_param_path = os.path.join(save_dir, f"full_train_last_checkpoint.pth")
                torch.save(model.state_dict(), save_param_path)

                # Save best model
                if test_accu > best_accuracy:
                    best_accuracy = test_accu
                    save_param_path = os.path.join(best_dir, f'full_model_best_{best_accuracy:.4f}acc_{use_data}_epoch{epoch}_{model.num_heads}heads_{model.num_blocks}blocks_{batch_size}batch.pth')
                    torch.save(model.state_dict(), save_param_path)
                    print(f"New best model saved with accuracy: {best_accuracy}")
            