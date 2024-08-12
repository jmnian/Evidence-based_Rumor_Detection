import clip
import json
import os
import torch
from PIL import Image


def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>', '')
    input_str = input_str.replace('</b>', '')
    return input_str


def load_captions(inv_dict):
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


def load_captions_weibo(direct_dict):
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


def load_imgs_direct_search(item_folder_path, direct_dict):
    image_paths = []
    keys_to_check = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
    for key1 in keys_to_check:
        if key1 in direct_dict.keys():
            for page in direct_dict[key1]:
                image_path = os.path.join(item_folder_path, page['image_path'].split('/')[-1])
                image_paths.append(image_path)
    return image_paths


def normalize_tensor_rowwise(tensor):
    mean_val = tensor.mean(dim=1, keepdim=True)
    std_val = tensor.std(dim=1, keepdim=True) + 1e-9  # Adding a small value to avoid division by zero
    normalized_tensor = (tensor - mean_val) / std_val
    return normalized_tensor


def normalize_tensor(tensor):
    mean_val = tensor.mean()
    std_val = tensor.std() + 1e-9  # Adding a small value to avoid division by zero
    normalized_tensor = (tensor - mean_val) / std_val
    return normalized_tensor


def get_single_sample(clip_name, device, key):
    clip_model, preprocess = clip.load(clip_name, device=device)
    json_file_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main\\data" \
                     "\\c_test_shuffled.json"
    data_root_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main\\data"

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
        cur_sample_dict = data_dict.get(str(key))
        direct_path = os.path.join(data_root_path, cur_sample_dict['direct_path'])
        inverse_path = os.path.join(data_root_path, cur_sample_dict['inv_path'])
        direct_path = direct_path.replace("/", "\\")  # 将正斜杠替换为反斜杠
        inverse_path = inverse_path.replace("/", "\\")  # 将正斜杠替换为反斜杠
        inv_anno_dict = json.load(open(os.path.join(inverse_path, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path, 'direct_annotation.json'), encoding='utf-8'))

        '''Evidence Text'''
        captions = load_captions(inv_anno_dict)
        captions_weibo = load_captions_weibo(direct_dict)

        captions += captions_weibo
        captions = [s for s in captions if s]  # remove empty strings

        text_tokens = clip.tokenize(captions, truncate=True).to(device)

        with torch.no_grad():
            evidence_text_features = clip_model.encode_text(text_tokens)  # (n, 768)
        evidence_text_features = normalize_tensor_rowwise(evidence_text_features)

        '''Evidence Img'''
        image_paths = load_imgs_direct_search(direct_path, direct_dict)
        images = []
        for path in image_paths:
            try:
                image = Image.open(path)
                processed_image = preprocess(image).unsqueeze(0).to(device)
                images.append(processed_image)
            except Exception as e:
                print(f"Error processing image: {path}, Error: {e}")
                continue
        evidence_img_features = []
        with torch.no_grad():
            for img in images:
                evi_img_emb = clip_model.encode_image(img)
                evi_img_emb = normalize_tensor(evi_img_emb)
                evidence_img_features.append(evi_img_emb)
        try:
            evidence_img_features = torch.stack(evidence_img_features)
        except Exception as e:
            print("Error in evidence_img_features:", e)
            evidence_img_features = torch.zeros(5, 1, 768)

        '''Post Img, Post Text'''
        caption = cur_sample_dict['caption']
        image_path = os.path.join(data_root_path, cur_sample_dict['image_path'])
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(caption, truncate=True)
        with torch.no_grad():
            post_text_features = clip_model.encode_text(text_tokens.to(device))
            post_img_features = clip_model.encode_image(image.to(device))

        post_text_features = normalize_tensor(post_text_features)
        post_img_features = normalize_tensor(post_img_features)
        label = torch.tensor(int(cur_sample_dict['label']))

        sample = {'label': label,  # torch.Size(1)
                  'post_text_features': post_text_features,  # torch.Size([1, 768])
                  'evidence_text_features': evidence_text_features,  # torch.Size([num_text_evidence, 768])
                  'post_img_features': post_img_features,  # torch.Size([1, 768])
                  'evidence_img_features': evidence_img_features}  # torch.Size([num_image_evidence, 768])

        return sample


def ablated_samples(clip_name, device, key):
    samples = []
    clip_model, preprocess = clip.load(clip_name, device=device)
    json_file_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main\\data" \
                     "\\c_test_shuffled.json"
    data_root_path = "C:\\pyProject\\Evidence-based_Rumor_Detection-main\\Evidence-based_Rumor_Detection-main\\data"

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
        cur_sample_dict = data_dict.get(str(key))
        direct_path = os.path.join(data_root_path, cur_sample_dict['direct_path'])
        inverse_path = os.path.join(data_root_path, cur_sample_dict['inv_path'])
        direct_path = direct_path.replace("/", "\\")  # 将正斜杠替换为反斜杠
        inverse_path = inverse_path.replace("/", "\\")  # 将正斜杠替换为反斜杠

        inv_anno_dict = json.load(open(os.path.join(inverse_path, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path, 'direct_annotation.json'), encoding='utf-8'))

        '''Evidence Text'''
        captions = load_captions(inv_anno_dict)
        captions_weibo = load_captions_weibo(direct_dict)

        captions += captions_weibo
        captions = [s for s in captions if s]  # remove empty strings

        text_tokens = clip.tokenize(captions, truncate=True).to(device)

        with torch.no_grad():
            evidence_text_features = clip_model.encode_text(text_tokens)  # (n, 768)
        evidence_text_features = normalize_tensor_rowwise(evidence_text_features)

        '''Evidence Img'''
        image_paths = load_imgs_direct_search(direct_path, direct_dict)
        images = []
        for path in image_paths:
            try:
                image = Image.open(path)
                processed_image = preprocess(image).unsqueeze(0).to(device)
                images.append(processed_image)
            except Exception as e:
                print(f"Error processing image: {path}, Error: {e}")
                continue
        evidence_img_features = []
        with torch.no_grad():
            for img in images:
                evi_img_emb = clip_model.encode_image(img)
                evi_img_emb = normalize_tensor(evi_img_emb)
                evidence_img_features.append(evi_img_emb)
        try:
            evidence_img_features = torch.stack(evidence_img_features)
        except Exception as e:
            print("Error in evidence_img_features:", e)
            evidence_img_features = torch.zeros(5, 1, 768)

    '''Post Img, Post Text'''
    caption = cur_sample_dict['caption']
    image_path = os.path.join(data_root_path, cur_sample_dict['image_path'])
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(caption, truncate=True)
    with torch.no_grad():
        post_text_features = clip_model.encode_text(text_tokens.to(device))
        post_img_features = clip_model.encode_image(image.to(device))

    post_text_features = normalize_tensor(post_text_features)
    post_img_features = normalize_tensor(post_img_features)
    label = torch.tensor(int(cur_sample_dict['label']))

    sample = {'label': label,  # torch.Size(1)
              'ablated_feature': 'all_include',
              'ablated_type': 'all_include',
              'post_text_features': post_text_features,  # torch.Size([1, 768])
              'evidence_text_features': evidence_text_features,  # torch.Size([num_text_evidence, 768])
              'post_img_features': post_img_features,  # torch.Size([1, 768])
              'evidence_img_features': evidence_img_features}  # torch.Size([num_image_evidence, 768])
    samples.append(sample)


    for text_evidence in captions:
        ablated_type = "text"
        ablated_feature = text_evidence

        remain_captions = []
        found_text_evidence = False

        for caption in captions:
            if caption == text_evidence and not found_text_evidence:
                found_text_evidence = True
            else:
                remain_captions.append(caption)

        remain_captions = [s for s in remain_captions if s]  # remove empty strings

        # Ensure at least one remaining caption exists
        if remain_captions:
            text_tokens = clip.tokenize(remain_captions, truncate=True).to(device)
            with torch.no_grad():
                remain_evidence_text_features = clip_model.encode_text(text_tokens)  # (n, 768)
            remain_evidence_text_features = normalize_tensor_rowwise(remain_evidence_text_features)
        else:
            # If no remaining captions, create a placeholder feature
            remain_evidence_text_features = torch.zeros(1, 768).to(device)  # Placeholder feature

        sample = {'label': label,  # torch.Size(1)
                  'ablated_feature': ablated_feature,
                  'ablated_type': ablated_type,
                  'post_text_features': post_text_features,  # torch.Size([1, 768])
                  'evidence_text_features': remain_evidence_text_features,  # torch.Size([num_text_evidence, 768])
                  'post_img_features': post_img_features,  # torch.Size([1, 768])
                  'evidence_img_features': evidence_img_features}  # torch.Size([num_image_evidence, 768])
        samples.append(sample)

    for excluded_image_path in image_paths:
        images = []
        for path in image_paths:
            if path != excluded_image_path:  # Exclude the current image path
                try:
                    image = Image.open(path)
                    processed_image = preprocess(image).unsqueeze(0).to(device)
                    images.append(processed_image)
                except Exception as e:
                    print(f"Error processing image: {path}, Error: {e}")
                    continue

        remain_evidence_img_features = []
        with torch.no_grad():
            for img in images:
                evi_img_emb = clip_model.encode_image(img)
                evi_img_emb = normalize_tensor(evi_img_emb)
                remain_evidence_img_features.append(evi_img_emb)
        try:
            remain_evidence_img_features = torch.stack(remain_evidence_img_features)
        except Exception as e:
            print("Error in evidence_img_features:", e)
            remain_evidence_img_features = torch.zeros(5, 1, 768)

        sample = {'label': label,  # torch.Size(1)
                  'ablated_feature': excluded_image_path,
                  'ablated_type': "image",
                  'post_text_features': post_text_features,  # torch.Size([1, 768])
                  'evidence_text_features': evidence_text_features,  # torch.Size([num_text_evidence, 768])
                  'post_img_features': post_img_features,  # torch.Size([1, 768])
                  'evidence_img_features': remain_evidence_img_features}  # torch.Size([num_image_evidence, 768])
        samples.append(sample)

    return samples
