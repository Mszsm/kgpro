import json
import numpy as np
import os
import re


re_pattern_triple = r".+\(.+,.+,.+\)"

# load multiple json files into one list
def load_json_files(file_list):
    data = []
    for file in file_list:
        with open(file, 'r') as f:
            data.append(json.load(f))
    return data

# convert the label in json files to machine readable format
def convert_label_gen(data, data_save_path=None):
    first_label_set = set()
    second_label_set = set()
    additional_label_set = set()
    not_rel_label_set = set()
    only_one_label_set = set()
    max_label_num = 0
    for item in data:
        item["relation_list"] = []
        for ans in item["answer"]:
            ans_types = ans['type']
            ans_type_list = ans_types.replace("（", "(").replace("）", ")").replace("，",",").replace(' ', '_').replace("__", "_").split('_')
            label_num = len(ans_type_list)
            if label_num > max_label_num:
                max_label_num = label_num
            if len(ans_type_list) == 1:
                only_one_label_set.add(ans_type_list[0])
            elif len(ans_type_list) >= 2:
                ans["first_label"] = ans_type_list[0]
                first_label_set.add(ans_type_list[0])
                if re.findall(re_pattern_triple, ans_type_list[1]) != []:
                    split_pos = ans_type_list[1].index("(")
                    ans["second_label"] = ans_type_list[1][:split_pos]
                    second_label_set.add(ans_type_list[1][:split_pos])
                    item["relation_list"].append(ans_type_list[1][split_pos+1:-1])
                    print()
                else:
                    ans["second_label"] = ans_type_list[1]
                    second_label_set.add(ans_type_list[1])
                if len(ans_type_list) == 3:
                    additional_label_set.add(ans_type_list[2])
                    if '(' not in ans_type_list[2]:
                        not_rel_label_set.add(ans_type_list[2])
                        ans["third_label"] = ans_type_list[2]
                    else:
                        item["relation_list"].append(ans_type_list[2])

            pass
    print()
    # with open(data_save_path, 'w') as f:
    #     for item in data:
    #         json.dump(item, f, ensure_ascii=False)
    #         f.write('\n')


# main function
if __name__ == '__main__':
    dir = '/home/wtang/Data/kg_project/banking_finance_disputes/plain_data'
    # read all files in the directory into list
    file_list = [os.path.join(dir, file) for file in os.listdir(dir)]
    data = load_json_files(file_list)

    # we should split data first for some context is too long that need to be cut intwo parts
    data_len = len(data)
    split_rate = 0.8
    train_data = data[:int(data_len * split_rate)]
    test_data = data[int(data_len * split_rate):]

    # convert label
    train_data_save_path = '/home/wtang/Data/kg_project/banking_finance_disputes/train_data.json'
    test_data_save_path = '/home/wtang/Data/kg_project/banking_finance_disputes/test_data.json'
    convert_label_gen(train_data, train_data_save_path)
    convert_label_gen(test_data, test_data_save_path)


