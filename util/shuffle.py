import json
import random
import pandas as pd


def shuffle_MR2_clip_fusion():
    # Load the JSON files into dictionaries
    with open("c_test.json", encoding='utf-8') as f:
        test_data = json.load(f)

    with open("c_train.json", encoding='utf-8') as f:
        train_data = json.load(f)

    with open("c_test.json", encoding='utf-8') as f:
        val_data = json.load(f)

    # Create a new dictionary to store the combined data with unique keys
    combined_data = {}
    unique_key = 0

    # Iterate over each dictionary and add its items to the combined dictionary
    for data in [test_data, train_data, val_data]:
        for value in data.values():
            combined_data[str(unique_key)] = value
            unique_key += 1

    # Convert the combined dictionary into a list of key-value pairs
    combined_list = list(combined_data.items())

    # Shuffle the list of key-value pairs
    random.shuffle(combined_list)

    # Convert the shuffled list back into a dictionary
    shuffled_data = dict(combined_list)

    # Get the sizes of the original files
    test_size = len(test_data)
    train_size = len(train_data)
    val_size = len(val_data)

    # Split the shuffled dictionary into new dictionaries with the same sizes as the original files
    new_test_data = dict(list(shuffled_data.items())[:test_size])
    new_train_data = dict(list(shuffled_data.items())[test_size:test_size + train_size])
    new_val_data = dict(list(shuffled_data.items())[test_size + train_size:])

    # Save the new dictionaries as JSON files with UTF-8 encoding and ensure_ascii=False
    with open("new_test.json", "w", encoding='utf-8') as f:
        json.dump(new_test_data, f, indent=4, ensure_ascii=False)

    with open("new_train.json", "w", encoding='utf-8') as f:
        json.dump(new_train_data, f, indent=4, ensure_ascii=False)

    with open("new_val.json", "w", encoding='utf-8') as f:
        json.dump(new_val_data, f, indent=4, ensure_ascii=False)


def mr2_to_FSRU_csv(mode):
    file_path = '../MR2/c_' + mode + '_shuffled.json'
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    data_list = []
    for cur_key in data:
        cur = data[cur_key]
        if cur['label'] != 2:
            data_list.append({'key': cur_key,
                              'caption': cur['caption'],
                              'image_path': cur['image_path'],
                              'label': cur['label']})

    df = pd.DataFrame(data_list)

    # Save DataFrame to CSV
    output_file_path = mode + '_data.csv'
    df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    mr2_to_FSRU_csv('train')
    mr2_to_FSRU_csv('test')
