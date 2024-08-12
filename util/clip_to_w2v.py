import clip
import torch
import numpy as np

# # Define the text file path
# text_file = r'G:\PycharmProjects\Evidence-based_Rumor_Detection-main\Evidence-based_Rumor_Detection-main\WeiboDataset' \
#             r'\weibo\tencent-ailab-embedding-zh-d100-v0.2.0-s\tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
#
# # Read the text file and extract the first token of each line
# with open(text_file, 'r', encoding='utf-8') as file:
#     next(file)  # Skip the first line
#     texts = [line.split()[0] for line in file if line.strip()]  # Extract the first word from each line
#
# # 加载CLIP模型
# device = "cuda"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
#
# # 词汇表大小
# voc_size = len(texts)
# batch_size = 32  # 根据可用内存选择合适的批次大小
#
# text_embeddings = []
# for i in range(0, len(texts), batch_size):
#     batch_texts = texts[i:i + batch_size]
#     text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
#
#     with torch.no_grad():
#         batch_embeddings = model.encode_text(text_tokens)
#         text_embeddings.append(batch_embeddings.cpu().numpy())
#
# text_embeddings_np = np.concatenate(text_embeddings, axis=0)
#
# # embedding的维度（假设所有embedding的维度一致）
# embedding_dim = text_embeddings_np.shape[1]
#
# # 将embedding保存到txt文件中
# with open('text_embeddings.txt', 'w', encoding='utf-8') as f:
#     # 写入第一行：词汇表大小和embedding维度
#     f.write(f"{voc_size} {embedding_dim}\n")
#
#     # 写入每个单词及其对应的embedding向量
#     for i, text in enumerate(texts):
#         # 写入文本
#         f.write(f"{text} ")
#         # 写入对应的embedding向量，每个值以空格分隔
#         f.write(' '.join(map(str, text_embeddings_np[i])) + '\n')
#

import pickle
import numpy as np

# 读取txt文件并解析内容
txt_file = r'G:\PycharmProjects\Evidence-based_Rumor_Detection-main\Evidence-based_Rumor_Detection-main\util\text_embeddings.txt'


def load_embeddings_from_txt(txt_file):
    embeddings_dict = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        # 跳过第一行（包含词汇表大小和embedding维度）
        next(f)
        for line in f:
            parts = line.strip().split()
            # 第一部分是词汇
            word = parts[0]
            # 其余部分是对应的embedding向量
            embedding = np.array(parts[1:], dtype=float)
            embeddings_dict[word] = embedding
    return embeddings_dict


# 保存字典为pickle文件
def save_to_pickle(data, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


pickle_file = 'w2v.pickle'  # 输出的pickle文件

# 加载embedding并保存为pickle文件
embeddings_dict = load_embeddings_from_txt(txt_file)
save_to_pickle(embeddings_dict, pickle_file)

print(f"Embeddings successfully saved to {pickle_file}")
