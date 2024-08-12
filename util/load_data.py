import clip
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, os, pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>', '')
    input_str = input_str.replace('</b>', '')
    # input_str = unidecode(input_str)
    return input_str


class DatasetClipEmbs(Dataset):
    def __init__(self, clip_name, data_dict, query_root_dir, device, split):
        self.device = device
        self.data_dict = data_dict
        self.query_root_dir = query_root_dir
        self.idx_to_keys = list(data_dict.keys())
        self.clip, self.preprocess = clip.load(clip_name, device=device)
        print(type(clip))
        self.split = split

    def __len__(self):
        return len(self.data_dict)

    def load_captions(self, inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions', 'all_partially_matched_captions']
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)

                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue
                            sub_captions_list.append(sub_caption_filter)
                            unfiltered_captions.append(sub_caption)
                        captions = captions + sub_captions_list

        pages_with_title_only_keys = ['partially_matched_no_text', 'fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self, direct_dict):
        captions = ['']
        keys = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
        for key1 in keys:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    if 'page_title' in page.keys():
                        item = page['page_title']
                        item = process_string(item)
                        captions.append(item)
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue
                            sub_captions_list.append(sub_caption_filter)
                            unfiltered_captions.append(sub_caption)
                        captions = captions + sub_captions_list
        return captions

    def load_imgs_direct_search(self, item_folder_path, direct_dict):
        image_paths = []
        keys_to_check = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path, page['image_path'].split('/')[-1])
                    image_paths.append(image_path)
        return image_paths

    def load_queries(self, key):
        caption = self.data_dict[key]['caption']
        image_path = os.path.join(self.query_root_dir, self.data_dict[key]['image_path'])
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(caption, truncate=True)
        with torch.no_grad():
            post_text_features = self.clip.encode_text(text_tokens.to(device))
            post_image_features = self.clip.encode_image(image.to(device))
        return post_image_features, post_text_features

    def normalize_tensor_rowwise(self, tensor):
        mean_val = tensor.mean(dim=1, keepdim=True)
        std_val = tensor.std(dim=1, keepdim=True) + 1e-9  # Adding a small value to avoid division by zero
        normalized_tensor = (tensor - mean_val) / std_val
        return normalized_tensor

    def normalize_tensor(self, tensor):
        mean_val = tensor.mean()
        std_val = tensor.std() + 1e-9  # Adding a small value to avoid division by zero
        normalized_tensor = (tensor - mean_val) / std_val
        return normalized_tensor

    def __getitem__(self, index):
        key = self.idx_to_keys[index]
        item = self.data_dict.get(str(key))
        direct_path_item = os.path.join(self.query_root_dir, item['direct_path'])
        inverse_path_item = os.path.join(self.query_root_dir, item['inv_path'])
        inv_anno_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))


        '''Evidence Text'''
        captions = self.load_captions(inv_anno_dict)  # did not look closely into this function
        captions += self.load_captions_weibo(direct_dict)  # did not look closely into this function
        captions = [s for s in captions if s]  # remove empty strings
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
        with torch.no_grad():
            evidence_text_features = self.clip.encode_text(text_tokens)  # (n, 768)
        evidence_text_features = self.normalize_tensor_rowwise(evidence_text_features)

        '''Evidence Img'''
        image_paths = self.load_imgs_direct_search(direct_path_item, direct_dict)
        images = []
        for path in image_paths:
            try:
                image = Image.open(path)
                processed_image = self.preprocess(image).unsqueeze(0).to(device)
                images.append(processed_image)
            except Exception as e:
                print(f"Error processing image: {path}, Error: {e}")
                continue
        evidence_img_features = []
        with torch.no_grad():
            for img in images:
                evi_img_emb = self.clip.encode_image(img)
                evi_img_emb = self.normalize_tensor(evi_img_emb)
                evidence_img_features.append(evi_img_emb)
        try:
            evidence_img_features = torch.stack(evidence_img_features)
        except Exception as e:
            print("Error in evidence_img_features:", e)
            evidence_img_features = torch.zeros(5, 1, 768)

        '''Post Img, Post Text'''
        post_img_features, post_text_features = self.load_queries(key)
        post_text_features = self.normalize_tensor(post_text_features)
        post_img_features = self.normalize_tensor(post_img_features)
        label = torch.tensor(int(item['label']))

        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            sample = {'label': label,  # torch.Size(1)
                      'post_text_features': post_text_features,  # torch.Size([1, 768])
                      'evidence_text_features': evidence_text_features,  # torch.Size([num_text_evidence, 768])
                      'post_img_features': post_img_features,  # torch.Size([1, 768])
                      'evidence_img_features': evidence_img_features}  # torch.Size([num_image_evidence, 768])

        return sample


def collate_fn(batch):
    # (1, batch, 768) (num_text_evidence, batch, 768), (1, batch, 768), (num_img_evidence, batch, 768)
    # post_text_features, evidence_text_features, post_img_features, evidence_img_features = batch

    post_text_features = torch.stack([item['post_text_features'] for item in batch], dim=1)
    post_img_features = torch.stack([item['post_img_features'] for item in batch], dim=1)

    # Pad and stack evidence_text_features
    max_text_length = max([item['evidence_text_features'].shape[0] for item in batch])
    padded_evidence_text_features = []
    for item in batch:
        padded_text_feat = F.pad(item['evidence_text_features'],
                                 pad=(0, 0, max_text_length - item['evidence_text_features'].shape[0], 0))
        padded_evidence_text_features.append(padded_text_feat)
    evidence_text_features = torch.stack(padded_evidence_text_features, dim=1)

    # Pad and stack evidence_img_features (same logic as above)
    max_length = max([item['evidence_img_features'].shape[0] for item in batch])
    padded_evidence_img_features = []
    for item in batch:
        tensor = item['evidence_img_features']
        pad_amount = max_length - tensor.shape[0]
        padded_tensor = F.pad(tensor, pad=(0, 0, 0, 0, 0, pad_amount)).to(device)  # Pad on the first dimension
        padded_evidence_img_features.append(padded_tensor)

    evidence_img_features = torch.stack(padded_evidence_img_features, dim=1).squeeze(2).to(device)

    features = {
        'post_text_features': post_text_features,
        'evidence_text_features': evidence_text_features,
        'post_img_features': post_img_features,
        'evidence_img_features': evidence_img_features,
    }

    if 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])
        return features, labels
    else:
        return features


def prepare_data_for_clip(clip_name, device, data_root_path, use_data, batch_size, use_shuffled_data):
    if use_data == "zh" and use_shuffled_data:
        data_items_train = json.load(open(f"{data_root_path}c_train_shuffled.json", encoding='utf-8'))
        data_items_val = json.load(open(f"{data_root_path}c_val_shuffled.json", encoding='utf-8'))
        data_items_test = json.load(open(f"{data_root_path}c_test_shuffled.json", encoding='utf-8'))
    elif use_data == "zh" and not use_shuffled_data:
        data_items_train = json.load(open(f"{data_root_path}new_train.json", encoding='utf-8'))
        data_items_val = json.load(open(f"{data_root_path}new_val.json", encoding='utf-8'))
        data_items_test = json.load(open(f"{data_root_path}new_test.json", encoding='utf-8'))

    train_dataset = DatasetClipEmbs(clip_name, data_items_train, data_root_path, device, 'train')
    val_dataset = DatasetClipEmbs(clip_name, data_items_val, data_root_path, device, 'val')
    test_dataset = DatasetClipEmbs(clip_name, data_items_test, data_root_path, device, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for sample in test_dataloader:
        pass

    return train_dataloader, val_dataloader, test_dataloader
