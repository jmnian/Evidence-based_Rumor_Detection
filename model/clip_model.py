import clip
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image 
from transformers import BertModel
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

'''how to use CLIP to get embeddings'''
# model, preprocess = clip.load("ViT-L/14", device=device)
# image = preprocess(Image.open("/home/jnian/Documents/CSEN342/MR2/data/test/img/0.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["全国各地医疗队万人援沪 其中部分医疗队已入驻方舱医院 - 实时热点 - 人"], truncate=True)
# with torch.no_grad():
#     text_features = model.encode_text(text)
#     image_features = model.encode_image(image)
#     print(text_features.shape)  # (1, 768)
#     print(image_features.shape) # (1, 768)

class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''

    def __init__(self, n_embd, dropout): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), # going back to the residual path way
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class RumorDetectClip(nn.Module):
    ''' mimicking transformer block, 1 block for text, 1 block for img'''
    
    def __init__(self, num_heads, clip_out, num_classes, dropout):
        super(RumorDetectClip, self).__init__()
        self.num_heads = num_heads
        self.text_attn = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.img_attn = nn.MultiheadAttention(embed_dim=clip_out, num_heads=num_heads)
        self.proj1 = nn.Linear(clip_out, clip_out)
        self.proj2 = nn.Linear(clip_out, clip_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(clip_out)
        self.ln2 = nn.LayerNorm(clip_out)
        self.ln3 = nn.LayerNorm(int(clip_out*4))
        self.ffwd1 = FeedForward(clip_out, dropout)
        self.ffwd2 = FeedForward(clip_out, dropout)
        
        self.fc = nn.Linear(in_features=int(clip_out*4), out_features=num_classes)
        
    def forward(self, inputs):
        # (1, batch, 768) (5, batch, 768), (1, batch, 768), (num_img_evidence, batch, 768)
        post_text_features = inputs['post_text_features']
        evidence_text_features = inputs['evidence_text_features']
        post_img_features = inputs['post_img_features']
        evidence_img_features = inputs['evidence_img_features']
        
        # attention input is query, key, value
        # query should have dimensions (1, batch_size, embedding_size)
        # key and value should be the same, with dimensions (num_evidence, batch_size, embedding_size)
        # text attend image, image attend text. This makes sense for CLIP embeddings
        # text_attn_output, _ = self.text_attn(post_text_features.to(device), evidence_img_features.to(device), evidence_img_features.to(device))
        # img_attn_output, _ = self.img_attn(post_img_features.to(device), evidence_text_features.to(device), evidence_text_features.to(device))
        # text attend text , img attend img. initial attempt
        text_attn_output, _ = self.text_attn(post_text_features.to(device), evidence_text_features.to(device), evidence_text_features.to(device))
        img_attn_output, _ = self.img_attn(post_img_features.to(device), evidence_img_features.to(device), evidence_img_features.to(device))
        
        # text attn output -> add & norm -> FFWD -> add & norm
        # img attn output -> add & norm -> FFWD -> add & norm
        text_attn = text_attn_output + self.dropout1(self.proj1(text_attn_output)) 
        img_attn = img_attn_output + self.dropout2(self.proj2(img_attn_output))
        text_attn = text_attn + self.ffwd1(self.ln1(text_attn)) 
        img_attn = img_attn + self.ffwd2(self.ln2(img_attn)) 
        x = torch.cat([text_attn, post_text_features, img_attn, post_img_features], dim=-1) # torch.Size([1, 8, 768*4])
        x = self.ln3(x) # 4 vectors -> norm -> Linear layer for classification
        logits = self.fc(x) 
        return logits 
    
class RumorDetectClipTwoTransformerBlock(nn.Module):
    ''' mimicking transformer block, 1 block for text, 1 block for img
        then add another set of transformers after them '''
    
    def __init__(self, num_heads, clip_out, num_classes, dropout=0.2):
        super(RumorDetectClipTwoTransformerBlock, self).__init__()
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
        
        self.ln_final = nn.LayerNorm(clip_out*4)
        self.fc = nn.Linear(in_features=clip_out*4, out_features=num_classes)
        
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
        # text_attn1 = self.ln1_text(post_text_features + self.ffwd1_text(text_attn_output1))
        # img_attn1 = self.ln1_img(post_img_features + self.ffwd1_img(img_attn_output1))
        
        # Second transformer block for text and image (added)
        text_attn_output2, _ = self.text_attn2(post_text_features, evidence_img_features, evidence_img_features)
        img_attn_output2, _ = self.img_attn2(post_img_features, evidence_text_features, evidence_text_features)
        text_attn2 = text_attn_output2 + self.dropout21(self.proj21(text_attn_output2)) 
        img_attn2 = img_attn_output2 + self.dropout22(self.proj22(img_attn_output2))
        text_attn2 = text_attn1 + self.ffwd2_text(self.ln2_text(text_attn2)) 
        img_attn2 = img_attn1 + self.ffwd2_img(self.ln2_img(img_attn2)) 
        # text_attn2 = self.ln2_text(text_attn1 + self.ffwd2_text(text_attn_output2))
        # img_attn2 = self.ln2_img(img_attn1 + self.ffwd2_img(img_attn_output2))
        
        # Final processing and classification
        combined_features = torch.cat([text_attn2, img_attn2, post_text_features, post_img_features], dim=-1)  
        # combined_features = torch.cat([text_attn1, img_attn1, text_attn2, img_attn2, post_text_features, post_img_features], dim=-1)  
        combined_features = self.ln_final(combined_features)
        logits = self.fc(combined_features)
        
        return logits
    
class MyTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MyTransformerBlock, self).__init__()
        self.text_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.img_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.proj_text = nn.Linear(embed_dim, embed_dim)
        self.proj_img = nn.Linear(embed_dim, embed_dim)
        self.dropout_text = nn.Dropout(dropout)
        self.dropout_img = nn.Dropout(dropout)
        self.ffwd_text = FeedForward(embed_dim, dropout)
        self.ffwd_img = FeedForward(embed_dim, dropout)
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_img = nn.LayerNorm(embed_dim)

    def forward(self, post_text_features, evidence_text_features, post_img_features, evidence_img_features):
        # Transformer block for text
        text_attn_output, _ = self.text_attn(post_text_features, evidence_img_features, evidence_img_features)
        text_attn = text_attn_output + self.dropout_text(self.proj_text(text_attn_output))
        text_attn = text_attn + self.ffwd_text(self.ln_text(text_attn))

        # Transformer block for image
        img_attn_output, _ = self.img_attn(post_img_features, evidence_text_features, evidence_text_features)
        img_attn = img_attn_output + self.dropout_img(self.proj_img(img_attn_output))
        img_attn = img_attn + self.ffwd_img(self.ln_img(img_attn))

        return text_attn, img_attn
    
class RumorDetectClipTransformer(nn.Module):
    def __init__(self, num_heads, clip_out, num_classes, dropout, num_blocks):
        super(RumorDetectClipTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.transformer_blocks = nn.ModuleList([
            MyTransformerBlock(clip_out, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(clip_out * 4)
        self.fc = nn.Linear(in_features=clip_out * 4, out_features=num_classes)

    def forward(self, inputs):
        post_text = inputs['post_text_features'].to(device)
        post_img = inputs['post_img_features'].to(device)
        post_text_features = inputs['post_text_features'].to(device)
        evidence_text_features = inputs['evidence_text_features'].to(device)
        post_img_features = inputs['post_img_features'].to(device)
        evidence_img_features = inputs['evidence_img_features'].to(device)

        for block in self.transformer_blocks:
            text_attn, img_attn = block(post_text_features, evidence_text_features, post_img_features, evidence_img_features)
            post_text_features, post_img_features = text_attn, img_attn

        combined_features = torch.cat([post_text, post_text_features, post_img, post_img_features], dim=-1)
        combined_features = self.ln_final(combined_features)
        logits = self.fc(combined_features)
        return logits
    
class RumorDetectClipGRU(nn.Module):
    def __init__(self, clip_out, num_classes, dropout):
        super(RumorDetectClipGRU, self).__init__()
        self.text_gru = nn.GRU(input_size=clip_out, hidden_size=clip_out, batch_first=False)
        self.img_gru = nn.GRU(input_size=clip_out, hidden_size=clip_out, batch_first=False)
        self.proj1 = nn.Linear(clip_out, clip_out)
        self.proj2 = nn.Linear(clip_out, clip_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(clip_out)
        self.ln2 = nn.LayerNorm(clip_out)
        self.ln3 = nn.LayerNorm(int(clip_out*4)) # Adjusted to clip_out * 4 for the concatenated vector
        self.ffwd1 = FeedForward(clip_out, dropout)
        self.ffwd2 = FeedForward(clip_out, dropout)

        self.fc = nn.Linear(in_features=int(clip_out*4), out_features=num_classes)

    
    def forward(self, inputs):
        evidence_text_features = inputs['evidence_text_features']  # [seq_len, batch, feature]
        # print(f"evidence_text_features shape: {evidence_text_features.shape}")
        
        post_text_features = inputs['post_text_features'].squeeze(0)  # Adjust to [batch, feature]
        evidence_img_features = inputs['evidence_img_features']  # [seq_len, batch, feature]
        post_img_features = inputs['post_img_features'].squeeze(0)  # Adjust to [batch, feature]

        # Process evidence text and image features with GRUs
        # We use only the output, ignoring the hidden state
        text_gru_output, _ = self.text_gru(evidence_text_features)  # [seq_len, batch, feature]
        # print(f"text_gru_output shape: {text_gru_output.shape}")
        
        img_gru_output, _ = self.img_gru(evidence_img_features)  # [seq_len, batch, feature]
        
        # Use the last output of the GRUs for each sequence as the representation
        text_repr = text_gru_output[-1]  # Take last timestep [batch, feature]
        # print(f"text_repr shape: {text_repr.shape}")
       
        img_repr = img_gru_output[-1]  # Take last timestep [batch, feature]
        
        # text rnn output -> add & norm -> FFWD -> add & norm
        # img rnn output -> add & norm -> FFWD -> add & norm
        text_repr = text_repr + self.dropout1(self.proj1(text_repr)) 
        img_repr = img_repr + self.dropout2(self.proj2(img_repr))
        text_repr = text_repr + self.ffwd1(self.ln1(text_repr)) 
        img_repr = img_repr + self.ffwd2(self.ln2(img_repr)) 
        
        # Concatenate post features with the last GRU outputs
        x = torch.cat([text_repr, post_text_features, img_repr, post_img_features], dim=-1)  # [batch, feature * 4]

        x = self.ln3(x) # 4 vectors -> norm -> Linear layer for classification
        logits = self.fc(x) 
        
        return logits




class BertResClip(nn.Module):
    def __init__(self, num_classes, dropout, device):
        super(BertResClip, self).__init__()
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=self.bert.config.hidden_size * 4, out_features=self.bert.config.hidden_size)
        self.fc2 = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_classes)

    def forward(self, inputs):
        post_text = inputs['post_text']
        evidence_text = inputs['evidence_text']
        post_img = inputs['post_img']
        evidence_img = inputs['evidence_img']

        # Process post text and evidence text with CLIP
        with torch.no_grad():
            post_text_features = self.clip_model.encode_text(post_text)
            evidence_text_features = self.clip_model.encode_text(evidence_text.view(-1, evidence_text.size(-1)))
            evidence_text_features = evidence_text_features.view(evidence_text.size(0), evidence_text.size(1), -1)

        # Process post text features with BERT
        post_text_outputs = self.bert(post_text_features)
        post_text_pooled = post_text_outputs.pooler_output

        # Process evidence text features with BERT
        evidence_text_outputs = self.bert(evidence_text_features.view(-1, evidence_text_features.size(-1)))
        evidence_text_pooled = evidence_text_outputs.pooler_output
        evidence_text_pooled = evidence_text_pooled.view(evidence_text.size(0), evidence_text.size(1), -1)
        evidence_text_pooled = torch.mean(evidence_text_pooled, dim=1)

        # Process post image with CLIP
        with torch.no_grad():
            post_img_features = self.clip_model.encode_image(post_img)

        # Process evidence image with ResNet
        evidence_img = evidence_img.view(-1, evidence_img.size(-3), evidence_img.size(-2), evidence_img.size(-1))
        evidence_img_pooled = self.resnet(evidence_img)
        evidence_img_pooled = evidence_img_pooled.view(evidence_img.size(0), -1, self.bert.config.hidden_size)
        evidence_img_pooled = torch.mean(evidence_img_pooled, dim=1)

        # Concatenate post and evidence representations
        x = torch.cat([post_text_pooled, evidence_text_pooled, post_img_features, evidence_img_pooled], dim=-1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class RumorDetectClipCNN(nn.Module):
    def __init__(self, clip_out, num_classes, dropout):
        super(RumorDetectClipCNN, self).__init__()
        self.text_cnn = nn.Sequential(
            CNNBlock(clip_out, 256, 3, dropout),
            CNNBlock(256, 128, 3, dropout),
            nn.AdaptiveMaxPool1d(1)
        )
        self.img_cnn = nn.Sequential(
            CNNBlock(clip_out, 256, 3, dropout),
            CNNBlock(256, 128, 3, dropout),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(128 * 2 + clip_out * 2, num_classes)
        
    def forward(self, inputs):
        evidence_text_features = inputs['evidence_text_features']  # [seq_len, batch, feature]
        post_text_features = inputs['post_text_features'].squeeze(0)  # Adjust to [batch, feature]
        evidence_img_features = inputs['evidence_img_features']  # [seq_len, batch, feature]
        post_img_features = inputs['post_img_features'].squeeze(0)  # Adjust to [batch, feature]
        
        # Transpose evidence text and image features to [batch, feature, seq_len]
        evidence_text_features = evidence_text_features.permute(1, 2, 0)
        evidence_img_features = evidence_img_features.permute(1, 2, 0)
        
        # Process evidence text and image features with CNNs
        text_repr = self.text_cnn(evidence_text_features).squeeze(-1)  # [batch, 128]
        img_repr = self.img_cnn(evidence_img_features).squeeze(-1)  # [batch, 128]
        
        # Concatenate post features with the CNN outputs
        x = torch.cat([text_repr, post_text_features, img_repr, post_img_features], dim=-1)  # [batch, 128 * 2 + clip_out * 2]
        logits = self.fc(x)
        
        return logits