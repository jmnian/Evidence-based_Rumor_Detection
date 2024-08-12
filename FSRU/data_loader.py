"""
An Lao
"""
import os
import copy
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from collections import defaultdict
from torchvision import transforms
import numpy as np


def get_image(img_data_path):
    image_dict = {}
    path_list = [img_data_path + 'val/img', img_data_path + 'train/img', img_data_path + 'test/img']
    for path in path_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for i, filename in enumerate(os.listdir(path)):
            try:
                img_path = path + '/' + filename
                im = Image.open(img_path).convert('RGB')
                im = data_transforms(im)
                image_dict[img_path] = im  # remove '.jpg'
            except:
                print(filename)
    print("image length " + str(len(image_dict)))
    return image_dict


def count(labels):
    r, nr = 0, 0
    for label in labels:
        if label == 0:
            nr += 1
        elif label == 1:
            r += 1
    return r, nr


def get_data(data_path, mode, image_dict):
    file = data_path + mode + '_data.csv'

    data = pd.read_csv(file, sep=',')
    tweet = data['caption'].tolist()
    image_url = data['image_path'].tolist()
    label = data['label'].tolist()

    texts, images, labels = [], [], []
    ct = 0
    for url in image_url:
        texts.append(str(tweet[ct]).split())
        images.append(image_dict['../MR2/' + url])
        labels.append(label[ct])
        ct += 1
    print('weibo:', len(texts), 'samples...')

    r, nr = count(labels)
    print(mode, 'contains:', r, 'rumor tweets,', nr, 'real tweets.')

    # rt_data = {'text': np.array(texts), 'image': np.array(images), 'label': np.array(labels)}
    rt_data = {'text': texts, 'image': images, 'label': labels}
    return rt_data


def get_vocab(train_data, test_data):
    vocab = defaultdict(float)
    all_text = train_data['text'] + test_data['text']
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


"""refer to EANN"""


def add_unknown_words(w2v, vocab, min_df=1, k=512):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in w2v and vocab[word] >= min_df:
            w2v[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(w2v, k=512):
    word_idx_map = dict()
    W = np.zeros(shape=(len(w2v) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in w2v:
        # for word in w2v.key_to_index.keys():
        W[i] = w2v[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def word2vec(text, word_idx_map, seq_len):
    word_embedding = []
    for sentence in text:
        sentence_embed = []
        for i, word in enumerate(sentence):
            sentence_embed.append(word_idx_map[word])
        # padding
        while len(sentence_embed) <= seq_len:
            sentence_embed.append(0)
        # cutting
        sentence_embed = sentence_embed[:seq_len]
        word_embedding.append(copy.deepcopy(sentence_embed))
    return word_embedding


def load_data(args):

    print('Loading image...')
    image_dict = get_image(args.img_data_path)

    print('Loading data...')
    train_data = get_data(args.data_path, 'train', image_dict)
    test_data = get_data(args.data_path, 'test', image_dict)

    vocab, all_text = get_vocab(train_data, test_data)
    print("vocab size: " + str(len(vocab)))
    max_len = len(max(all_text, key=len))
    print("max sentence length: " + str(max_len))

    print("Loading word2vec...")
    word_embedding_path = 'w2v.pickle'
    w2v = pickle.load(open(word_embedding_path, 'rb'))
    # w2v = w2v.wv
    print("Number of words already in word2vec:" + str(len(w2v)))
    add_unknown_words(w2v, vocab)

    # reduced_w2v = reduce_dimensions(w2v, n_components=32)
    W, word_idx_map = get_W(w2v)
    # w_file = open(args.data_path+'word_embedding.pickle', 'wb')
    # pickle.dump([W, word_idx_map, vocab, max_len], w_file)
    # w_file.close()
    args.vocab_size = len(vocab)
    print('Translate text to embedding...')
    train_data['text_embed'] = word2vec(train_data['text'], word_idx_map, args.seq_len)
    test_data['text_embed'] = word2vec(test_data['text'], word_idx_map, args.seq_len)

    text = all_text
    text_embed = train_data['text_embed'] + test_data['text_embed']

    image = train_data['image'] + test_data['image']
    label = train_data['label'] + test_data['label']
    # data = {'text': np.array(text), 'text_embed':np.array(text_embed),
    #         'image': np.array(image), 'label': np.array(label)}
    text_embed_np = np.array(text_embed)
    print("Converted text_embed to np.array successfully.")
    label_np = np.array(label)
    print("Converted label to np.array successfully.")
    image_np = np.array(image)
    print("Converted image to np.array successfully.")
    return text_embed_np, image_np, label_np, W
