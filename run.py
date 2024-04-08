import os
from datetime import datetime
import torch 
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from dataset.prepare_dataset import prepare_data_for_clip
from model.clip_model import RumorDetectClip, RumorDetectClipTwoTransformerBlock, RumorDetectClipTransformer
from train.train import train, evaluate
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'exp/exp_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')
torch.manual_seed(1339) 



# Settings #
use_data = 'zh'
save_dir = "checkpoint_two_transformer/"
best_dir = "best_two_transformer_model"
CLIP_NAME = "ViT-L/14"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)
############
# Hyper params #
num_attn_heads = 12
num_blocks=3
batch_size = 32
epochs = 30
learning_rate = 3e-5
dropout = 0.2
################

# transform all text and image into CLIP embeddings, put into batches
train_dataloader, val_dataloader, test_dataloader = prepare_data_for_clip(clip_name=CLIP_NAME, 
                                                                          device=device, 
                                                                          data_root_path='./data/', 
                                                                          use_data=use_data, 
                                                                          batch_size=batch_size)

model = RumorDetectClipTwoTransformerBlock(num_heads=num_attn_heads, clip_out=768, num_classes=3, dropout=dropout)
model.to(device)

# optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

criterion = CrossEntropyLoss()


if __name__ == '__main__':
    '''training'''
    # with torch.autocast(device_type="cuda"):
    #     train(epochs, device, batch_size, model, criterion, optimizer, val_dataloader, train_dataloader, use_data, save_dir, best_dir, writer) 
    
    
    '''inference'''
    with torch.autocast(device_type="cuda"):
        print("Evaluating on test set...")
        best_model_path = "best_two_transformer_model/model_best_0.955182acc_zh_epoch17_12heads_32batch.pth"
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        print(f"{best_model_path} LOADED Succesfully")
        loss, accu, f1, p, r = evaluate(model, criterion, train_dataloader, device)
        print(f"test loss: {loss}, test accuracy: {accu}, test f1: {(2*p*r)/(p+r)}, test precision: {p}, test recall: {r}")
        