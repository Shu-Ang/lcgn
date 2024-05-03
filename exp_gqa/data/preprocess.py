import json
import numpy as np
from collections import Counter

data_path = '../gqa_dataset/sceneGraphs/train_sceneGraphs.json'


def filter_objects():
    objects_list = []
    with open(data_path, 'r') as f:
        data = json.load(f)
        values = data.values()
        for value in values:
            object_name_dicts = value['objects'].values()
            for object_name_dict in object_name_dicts:
                name = object_name_dict['name']
                objects_list.append(name)

    count = Counter(objects_list)
    filtered_count = {string: cnt for string, cnt in count.items() if cnt >= 2000}
    return filtered_count


def build_co_occur_matrix(filtered_count):
    images = []
    with open(data_path, 'r') as f:
        data = json.load(f)
        values = data.values()
        for value in values:
            object_name_dicts = value['objects'].values()
            obj_list = []
            for object_name_dict in object_name_dicts:
                name = object_name_dict['name']
                if name in filtered_count:
                    obj_list.append(name)
            images.append(obj_list)

    objects = list(filtered_count.keys())
    num_objects = len(objects)
    co_occurrence_matrix = [[0 for _ in range(num_objects)] for _ in range(num_objects)]
    object_to_index = {obj: i for i, obj in enumerate(objects)}
    for image in images:
        # 将图像中的每个对象转换为索引
        image_indices = [object_to_index[obj] for obj in image]
        # 对于图像中的每对对象，增加共现矩阵中的计数
        for i in range(len(image_indices)):
            for j in range(i + 1, len(image_indices)):
                co_occurrence_matrix[image_indices[i]][image_indices[j]] += 1

    co_occurrence_matrix = np.array(co_occurrence_matrix)
    print(co_occurrence_matrix.shape)
    np.save('./co_occur_matrix.npy', co_occurrence_matrix)


def build_word2idx():
    word2idx = {}
    with open(name_path, 'r', encoding='utf-8') as file:
        objects = [line.strip() for line in file.readlines()]
    for i in range(len(objects)):
        word2idx[objects[i]] = i
    with open('./word2idx.json', 'w', encoding='utf-8') as f:
        # 使用json.dump()写入字典，设置indent参数可以格式化输出，使JSON文件具有可读性
        json.dump(word2idx, f, ensure_ascii=False, indent=4)


def build_idx2word():
    word2idx = {}
    with open(name_path, 'r', encoding='utf-8') as file:
        objects = [line.strip() for line in file.readlines()]
    for i in range(len(objects)):
        word2idx[i] = objects[i]
    with open('./idx2word.json', 'w', encoding='utf-8') as f:
        # 使用json.dump()写入字典，设置indent参数可以格式化输出，使JSON文件具有可读性
        json.dump(word2idx, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    filter_count = filter_objects()
    build_co_occur_matrix(filter_count)
