"""
Add to all ret_meta.json small values in bboxes/
"""

import os
import json
import random

def distort_ret_meta(data: dict, max_deviation=0.005):
    for i in range(len(data['items'])):
        item = data['items'][i]

        for trained_pic in item['trained']:

            for coord in range(4):
                deviation = random.uniform(-max_deviation, max_deviation)
                trained_pic['bbox'][coord] = max(0., min(trained_pic['bbox'][coord] + deviation, 1.))

    return data



meta_path = '/home/achernikov/CLS/data_2301/meta'

for group in os.listdir(meta_path):
    picsets = os.listdir(os.path.join(meta_path, group))
    
    for picset in picsets:
        print(group, picset)

        ret_meta_path = os.path.join(meta_path, group, picset, 'ret_meta.json')
        with open(ret_meta_path) as f:
            data = json.load(f)

        data = distort_ret_meta(data)
        distorted_ret_meta_path = os.path.join(meta_path, group, picset, 'distorted_ret_meta.json')
        with open(distorted_ret_meta_path, 'w') as f:
            json.dump(data, f, indent=4)

