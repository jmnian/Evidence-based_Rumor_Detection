from model.clip_model import RumorDetectClipTwoTransformerBlock
from dataset.prepare_dataset import prepare_data_for_clip
from train.train import evaluate
import torch 
from torch.nn import CrossEntropyLoss


USE_DATA = 'zh'
use_shuffled_data = True
data_root_path='/Users/jimmynian/code/CSEN342/MR2/data/'
CLIP_NAME = "ViT-L/14"
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)

train_dataloader, val_dataloader, test_dataloader = prepare_data_for_clip(clip_name=CLIP_NAME, 
                                                                          device=device, 
                                                                          data_root_path=data_root_path, 
                                                                          use_data=USE_DATA, 
                                                                          batch_size=batch_size,
                                                                          use_shuffled_data=use_shuffled_data)
criterion = CrossEntropyLoss()


model_weights_path = "/Users/jimmynian/code/CSEN342/MR2/best_model/model_best_0.9440acc_zh_epoch7_12heads_32batch.pth"
model = RumorDetectClipTwoTransformerBlock(num_heads=12, clip_out=768, num_classes=3)
if device == 'cpu':
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
elif device == 'cuda':
    model.load_state_dict(torch.load(model_weights_path))
model.to(device)
model.eval()

print(model)
print(sum(p.numel() for p in model.parameters())/1e6, "million parameters")

# with torch.autocast(device_type="cuda"): 
#     print(f"{model_weights_path} LOADED Succesfully")
#     loss, accu, f1, p, r = evaluate(model, criterion, test_dataloader, device)
#     print(f"test loss: {loss}, test accuracy: {accu}, test f1: {(2*p*r)/(p+r)}, test precision: {p}, test recall: {r}")