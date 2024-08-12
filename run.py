import json
import os
from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
# from model.clip_model import RumorDetectClip
from train import train, evaluate
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
from model.clipTwoBlocks import ClipTwoTransformerBlock
from model.clipFourBlocks import RumorDetectClipFourTransformerBlock
from util.load_data import prepare_data_for_clip, collate_fn
from train import evaluate, evaluate_single_sample
import torch
from torch.nn import CrossEntropyLoss
from util.evidence_evaluation import get_single_sample, ablated_samples
from PIL import Image
import clip

# Settings #
use_data = 'zh'
save_dir = "checkpoint"
best_dir = "best_two_transformer_model"
data_root_path = "data/"
CLIP_NAME = "ViT-L/14"
use_shuffled_data = True
os.makedirs(save_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)
device = "cuda"
print("using", device)
############
# Hyper params #
num_attn_heads = 12
num_blocks = 3
# for Nvidia geforce rtx 2080 super
batch_size = 32
epochs = 15
learning_rate = 3e-5
dropout = 0.2
clip_out = 768
weight_decay = 5e-4


################

def train_model(model_name):
    # transform all text and image into CLIP embeddings, put into batches
    train_dataloader, val_dataloader, _ = prepare_data_for_clip(clip_name=CLIP_NAME,
                                                                device=device,
                                                                data_root_path=data_root_path,
                                                                use_data=use_data,
                                                                batch_size=batch_size,
                                                                use_shuffled_data=True)

    writer = SummaryWriter(f'exp\exp_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}')
    if model_name == "fourblocks":
        model = RumorDetectClipFourTransformerBlock(num_heads=num_attn_heads, clip_out=clip_out, num_classes=num_blocks,
                                                    dropout=dropout)
    elif model_name == "twoblocks":
        model = ClipTwoTransformerBlock(num_heads=num_attn_heads, clip_out=clip_out, num_classes=num_blocks,
                                        dropout=dropout)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = CrossEntropyLoss()
    with torch.autocast(device_type="cuda"):
        train(model_name, epochs, device, batch_size, model, criterion, optimizer,
              val_dataloader, train_dataloader, use_data,
              save_dir, best_dir, writer)


def test_model(model_name):
    _, _, test_dataloader = prepare_data_for_clip(clip_name=CLIP_NAME,
                                                  device=device,
                                                  data_root_path=data_root_path,
                                                  use_data=use_data,
                                                  batch_size=batch_size,
                                                  use_shuffled_data=True)

    if model_name == "fourblocks":
        model = RumorDetectClipFourTransformerBlock(num_heads=num_attn_heads, clip_out=clip_out, num_classes=num_blocks,
                                                    dropout=dropout)
        # best_model_path = "RumorDetectClipFourTransformerBlock_best_0.9566acc_zh_epoch8_12heads_32batch.pth"
        best_model_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main" \
                          "\\RumorDetectClipFourTransformerBlock_best_0.9566acc_zh_epoch8_12heads_32batch.pth"
    elif model_name == "twoblocks":
        model = ClipTwoTransformerBlock(num_heads=num_attn_heads, clip_out=clip_out, num_classes=num_blocks,
                                        dropout=dropout)
        best_model_path = "model_best_0.9454acc_zh_epoch7_12heads_32batch.pth"

    criterion = CrossEntropyLoss()
    with torch.autocast(device_type="cuda"):
        print("Evaluating on test set...", model_name)
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        print(f"{best_model_path} LOADED Succesfully")
        loss, accu, f1, p, r = evaluate(model, criterion, test_dataloader, device)
        print(
            f"test loss: {loss}, test accuracy: {accu}, test f1: {(2 * p * r) / (p + r)}, test precision: {p}, test recall: {r}")


def ablation_test(CLIP_NAME, device, key):
    samples = ablated_samples(CLIP_NAME, device, key)
    print(len(samples))
    for sample in samples:
        print(sample['ablated_feature'])

    features, labels = collate_fn(samples)

    model = RumorDetectClipFourTransformerBlock(num_heads=num_attn_heads, clip_out=clip_out, num_classes=num_blocks,
                                                dropout=dropout)
    best_model_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main" \
                      "\\best_two_transformer_model\\fourblocks_best_0.9603acc_zh_epoch10_12heads_32batch.pth"
    criterion = CrossEntropyLoss()

    for sample in samples:
        model.eval()
        with torch.autocast(device_type="cuda"):
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
            model.eval()
            _, importance_scores = evaluate_single_sample(model, criterion, features, labels, device)


if __name__ == '__main__':
    torch.manual_seed(1339)
    # test_model("fourblocks")
    train_model("twoblocks")
    # get_single_sample(CLIP_NAME, device, 6755)
    # claim_key = 3972
    # print("for claim key: ", claim_key)
    # ablation_test(CLIP_NAME, device, claim_key)
