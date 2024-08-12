import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import BertModel
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),  # going back to the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RumorDetectClipFourTransformerBlock(nn.Module):
    def __init__(self, num_heads, clip_out, num_classes, dropout):
        super(RumorDetectClipFourTransformerBlock, self).__init__()
        self.num_heads = num_heads
        # First transformer block
        self.text_attn1 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.img_attn1 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.proj11 = nn.Linear(clip_out, clip_out)
        self.proj12 = nn.Linear(clip_out, clip_out)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.ffwd1_text = FeedForward(clip_out, dropout)
        self.ffwd1_img = FeedForward(clip_out, dropout)
        self.ln1_text = nn.LayerNorm(clip_out)
        self.ln1_img = nn.LayerNorm(clip_out)

        # Second transformer block (added)
        self.text_attn2 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.img_attn2 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.proj21 = nn.Linear(clip_out, clip_out)
        self.proj22 = nn.Linear(clip_out, clip_out)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.ffwd2_text = FeedForward(clip_out, dropout)
        self.ffwd2_img = FeedForward(clip_out, dropout)
        self.ln2_text = nn.LayerNorm(clip_out)
        self.ln2_img = nn.LayerNorm(clip_out)

        # Additional transformer blocks
        self.text_attn3 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.img_attn3 = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.proj31 = nn.Linear(clip_out, clip_out)
        self.proj32 = nn.Linear(clip_out, clip_out)
        self.dropout31 = nn.Dropout(dropout)
        self.dropout32 = nn.Dropout(dropout)
        self.ffwd3_text = FeedForward(clip_out, dropout)
        self.ffwd3_img = FeedForward(clip_out, dropout)
        self.ln3_text = nn.LayerNorm(clip_out)
        self.ln3_img = nn.LayerNorm(clip_out)

        self.ln_final = nn.LayerNorm(clip_out * 6)
        self.fc = nn.Linear(in_features=clip_out * 6, out_features=num_classes)

    def forward(self, inputs):
        post_text_features = inputs['post_text_features'].to(device)
        evidence_text_features = inputs['evidence_text_features'].to(device)
        post_img_features = inputs['post_img_features'].to(device)
        evidence_img_features = inputs['evidence_img_features'].to(device)

        # First transformer block for text and image
        text_attn_output1, _ = self.text_attn1(post_text_features, evidence_img_features, evidence_img_features)
        img_attn_output1, _ = self.img_attn1(post_img_features, evidence_text_features, evidence_text_features)
        text_attn = text_attn_output1 + self.dropout11(self.proj11(text_attn_output1))
        img_attn = img_attn_output1 + self.dropout12(self.proj12(img_attn_output1))
        text_attn1 = text_attn + self.ffwd1_text(self.ln1_text(text_attn))
        img_attn1 = img_attn + self.ffwd1_img(self.ln1_img(img_attn))

        # Second transformer block for text and image
        text_attn_output2, _ = self.text_attn2(post_text_features, evidence_img_features, evidence_img_features)
        img_attn_output2, _ = self.img_attn2(post_img_features, evidence_text_features, evidence_text_features)
        text_attn2 = text_attn_output2 + self.dropout21(self.proj21(text_attn_output2))
        img_attn2 = img_attn_output2 + self.dropout22(self.proj22(img_attn_output2))
        text_attn2 = text_attn1 + self.ffwd2_text(self.ln2_text(text_attn2))
        img_attn2 = img_attn1 + self.ffwd2_img(self.ln2_img(img_attn2))

        # post text and evidence text
        text_attn_output3, _ = self.text_attn3(post_text_features, evidence_text_features, evidence_text_features)
        text_attn3 = text_attn_output3 + self.dropout31(self.proj31(text_attn_output3))
        text_attn3 = text_attn3 + self.ffwd3_text(self.ln3_text(text_attn3))

        # post image and evidence image
        img_attn_output3, _ = self.img_attn3(post_img_features, evidence_img_features, evidence_img_features)
        img_attn3 = img_attn_output3 + self.dropout32(self.proj32(img_attn_output3))
        img_attn3 = img_attn3 + self.ffwd3_img(self.ln3_img(img_attn3))

        # Final processing and classification
        combined_features = torch.cat([
            text_attn2,
            img_attn2,  # Outputs from the original second transformer blocks
            text_attn3,
            img_attn3,  # Outputs from the additional transformer blocks
            post_text_features, post_img_features  # Original post features
        ], dim=-1)  # Concatenating along the feature dimension

        combined_features = self.ln_final(combined_features)
        logits = self.fc(combined_features)
        return logits