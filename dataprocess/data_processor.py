import os
import re
import unicodedata
import math
import json
import random
import copy
import numpy as np
from tqdm import tqdm
from utils import load_json, load_dict, write_dict, str_q2b
import dataprocess.rel2text
import random

from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def save_dict(dict, name):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(f'{name}.txt', 'w', encoding='utf-8') as f:
        f.write(str(dict))  # dict to str

def remove_stress_mark(text):
    text = "".join([c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"])
    return text
 
def change_case(str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return re.sub(r'[^\w\s]','',s2)

class TypeClsProcessor(object):
    def __init__(self,
                root,
                tokenizer,
                is_lower=False,
                dataset_name='test_data'):
        # self.task_data_dir = os.path.join(root, dataset_name)
        # self.train_path = os.path.join(self.task_data_dir, 'train_data.json')
        # self.dev_path = os.path.join(self.task_data_dir, 'test_data.json')
        # self.test_path = os.path.join(self.task_data_dir, 'test_data.json')
        self.train_path = '/home/jli/UniRel-main/dataset/train'
        self.dev_path = '/home/jli/UniRel-main/dataset/dev'
        self.test_path = '/home/jli/UniRel-main/dataset/test'

        self.label_list = list(set(['邻里纠纷','物业纠纷','交通事故损害赔偿纠纷','银行业纠纷（金融纠纷）','民间借贷纠纷','山林土地纠纷','征地拆迁纠纷','房屋类纠纷','婚姻家庭纠纷','劳资劳务纠纷','医患纠纷','消费纠纷','其他纠纷']))
        self.lable2id =  {key: value for value,key in enumerate(self.label_list)}
        self.num_rels = 13

    def get_train_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.train_path, token_len, data_nums)
    
    def get_dev_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.dev_path, token_len, data_nums)
    
    def get_test_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.test_path, token_len, data_nums)
    def get_pridict_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.test_path, token_len, data_nums)
    
    def _data_process(self, path, token_len=512, data_nums=-1, ignore_cache=False):
        output = {
            "context":[],
            "type":[],
            "position":[],
        }
        # use for debug
        # ignore_cache = True
        cache_path = path + f"_token_len_{token_len}.json"
        if False:
        
        # if os.path.exists(cache_path) and not ignore_cache:
            # cache_fp = open(cache_path, "r", encoding="utf-8")
            # while line := cache_fp.readline():
            #     data = json.loads(line)
            #     output["context"].append(data["context"])
            #     output["type"].append(data["type"])
            output = load_dict(cache_path)
        else:
            mask_point = ["。", ".", "！", "？", "?", "，", ",", "\n"]
            dir_lists = os.listdir(path)
            for dir_path in tqdm(dir_lists):
                file_lsits = os.listdir(path+"/"+dir_path)
                for file_path in file_lsits:
                    # print(path+"/"+dir_path+"/"+file_path)
                    with open(path+"/"+dir_path+"/"+file_path) as fp:
                        start_flag = True
                        position_id = 0

                        sample = json.load(fp)
                        text = sample['qwContent_pain']
                        type = sample['type']
                        type_id = self.lable2id[random.choice(type)]

                        #text_part 开始位置
                        start = 0
                        # 上一句end_pos
                        last_end = 0
                        #截断时前后两个text——part会有重合部分
                        back_end = 0
                        max_len = 512
                        # 滑动窗口，以为截断点
                        text_len = len(text)
                        for end in range(text_len):
                            if text[end] in mask_point:
                                if end - start < max_len:
                                    back_end = last_end
                                    last_end = end
                                else:
                                    # print('qia',end,start,last_end,back_end)
                                    if last_end - start < max_len-1:
                                        text_part = text[start:last_end]
                                        # TODO: Cache code will write here
                                        if start_flag:
                                            start_flag = False
                                            position_id = 0
                                        else:
                                            position_id = 1
                                        output["context"].append(text_part)
                                        output["type"].append(type_id)
                                        output["position"].append(position_id)
                                        
                                    start = back_end + 1
                                    back_end = last_end
                                    last_end = end
                    output["position"][-1] = 2
            
            write_dict(cache_path, output)
        if data_nums != -1:
            output["context"] = output["context"][:data_nums]
            output["type"] = output["type"][:data_nums]
            output["position"] = output["position"][:data_nums]

        return output

class KgProjectGenProcessor(object):
    def __init__(self,
                root,
                tokenizer,
                is_lower=False,
                dataset_name='banking_finance_disputes'):
        self.task_data_dir = os.path.join(root, dataset_name)
        self.train_path = os.path.join(self.task_data_dir, 'train_data.json')
        self.dev_path = os.path.join(self.task_data_dir, 'test_data.json')
        self.test_path = os.path.join(self.task_data_dir, 'test_data.json')

        # first label set for banking disputes  {'委托代理人', '冲突烈度', '涉及金额', '纠纷当事人', '冲突时间', '原因', '判决结果', '所涉金融产品'}
        self.needed_label_set = set(['委托代理人', '涉及金额', '纠纷当事人', '冲突时间', '所涉金融产品'])

    def get_train_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.train_path, token_len, data_nums)
    
    def get_dev_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.dev_path, token_len, data_nums)
    
    def get_test_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.test_path, token_len, data_nums)
    def get_pridict_sample(self, token_len=512, data_nums=-1):
        return self._data_process(self.test_path, token_len, data_nums)
    def _data_process(self, path, token_len=512, data_nums=-1, ignore_cache=False):
        output = {
            "context":[],
            "answer":[],
            "answer_context":[],
        }
        # use for debug
        # ignore_cache = True

        cache_path = path + f"_token_len_{token_len}.json"
        if os.path.exists(cache_path) and not ignore_cache:
            cache_fp = open(cache_path, "r", encoding="utf-8")
            while line := cache_fp.readline():
                data = json.loads(line)
                output["context"].append(data["context"])
                output["answer"].append(data["answer"])
                output["answer_context"].append(data["answer_context"])
        else:
            cache_fp = open(cache_path, "w", encoding="utf-8")
            fp = open(path, "r", encoding="utf-8")
            break_set = set(["。", ".", "！", "？", "?", "，", ",", "\n"])
            cnt = 0
            while line := fp.readline():
                cnt += 1
                if cnt % 50 == 0:
                    print(cnt)
                data = json.loads(line)
                context_all = data["qwContent_pain"]
                # split context into parts with nearly 512 tokens. Make sure period is the end of each part.
                # This code is slow but have to leave it here. For token_len is changeable that would be inconvenient if at preprocess.
                # But we can cache the results once we done for a token_len
                context_parts = []
                context_part = ""
                idx = 0

                # The answer span should also be considered.
                # Sentence cannot be splited inside the answer span
                unsafe_idx_checkbox = np.zeros(len(context_all), dtype=np.int)
                for ans in data["answer"]:
                    if "first_label" not in ans:
                        continue
                    if ans["first_label"] in self.needed_label_set:
                        unsafe_idx_checkbox[ans["start_pos"]:ans["end_pos"]] = 1
                while idx < len(context_all):
                    sent = context_all[idx]
                    if len(context_part) + len(sent) < token_len:
                        context_part += sent
                    else:
                        context_part += sent
                        max_back_step = 128
                        while max_back_step > 0 and idx > 0: 
                            max_back_step -= 1
                            if context_all[idx] in break_set and unsafe_idx_checkbox[idx] == 0:
                                break
                            idx -= 1
                            context_part = context_part[:-1]
                        context_parts.append(context_part)
                        context_part = ""
                    idx += 1

                for idx, context in enumerate(context_parts):
                    output["context"].append(context)
                    answer_parts = []
                    answer_context = ""
                    for ans in data["answer"]:
                        if "first_label" not in ans:
                            continue
                        if ans["first_label"] in self.needed_label_set:
                            if ans["start_pos"] >= idx * token_len and ans["end_pos"] <= (idx + 1) * token_len:
                            # if ans["end_pos"] <= (idx + 1) * token_len:
                                answer_parts.append(ans)
                                answer_context += ans["first_label"] + "——" + ans["second_label"] + "：" + ans["context"]  + "。"
                    output["answer"].append(answer_parts)
                    if answer_context == "":
                        # Consider randomly discard some samples without answer
                        answer_context = "无"
                    output["answer_context"].append(answer_context)

                    # TODO: Cache code will write here
                    cache_fp.write(json.dumps({
                        "context":context,
                        "answer":answer_parts,
                        "answer_context":answer_context,
                    }, ensure_ascii=False) + "\n")
                
            fp.close()
        if data_nums != -1:
            output["context"] = output["context"][:data_nums]
            output["answer"] = output["answer"][:data_nums]
            output["answer_context"] = output["answer_context"][:data_nums]
        return output


# Driver code
class UniRelDataProcessor(object):
    def __init__(self,
                 root,
                 tokenizer,
                 is_lower=False,
                 dataset_name='nyt',
                 ):
        self.task_data_dir = os.path.join(root, dataset_name)
        self.train_path = os.path.join(self.task_data_dir, 'train_split.json')
        self.dev_path = os.path.join(self.task_data_dir, 'valid_data.json')
        self.test_path = os.path.join(self.task_data_dir, 'test_data.json')

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        self.label_map_cache_path = os.path.join(self.task_data_dir,
                                                 dataset_name + '.dict')

        self.label2id = None
        self.id2label = None
        self.max_label_len = 0

        self._get_labels()
        if dataset_name == "nyt":
            self.pred2text=dataprocess.rel2text.nyt_rel2text
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        elif dataset_name == "nyt_star":
            self.pred2text=dataprocess.rel2text.nyt_rel2text
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        elif dataset_name == "webnlg":
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
            self.pred2text=dataprocess.rel2text.webnlg_rel2text
            cnt = 1
            exist_value=[]
            # Some hard to convert relation directly use [unused]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
        elif dataset_name == "webnlg_star":
            self.pred2text={}
            for pred in self.label2id.keys():
                try:
                    self.pred2text[pred] = dataprocess.rel2text.webnlg_rel2text[pred]
                except KeyError:
                    print(pred)
            cnt = 1
            exist_value=[]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        self.num_rels = len(self.pred2text.keys())
        self.max_label_len = 1
        self.pred2idx = {}
        idx = 0
        self.pred_str = ""
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1
        self.pred_str = self.pred_str[:-1]
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.num_labels = self.num_rels

    def get_train_sample(self, token_len=100, data_nums=-1):
        return self._pre_process(self.train_path,
                                 token_len=token_len,
                                 is_predict=False,
                                 data_nums=data_nums)

    def get_dev_sample(self, token_len=150, data_nums=-1):
        return self._pre_process(self.dev_path,
                                 token_len=token_len,
                                 is_predict=True,
                                 data_nums=data_nums)

    def get_test_sample(self, token_len=150, data_nums=-1):
        samples = self._pre_process(self.test_path,
                                    token_len=token_len,
                                    is_predict=True,
                                    data_nums=data_nums)
        # json.dump(self.complex_data, self.wp, ensure_ascii=False)
        return samples

    def get_specific_test_sample(self, data_path, token_len=150, data_nums=-1):
        return self._pre_process(data_path,
                                 token_len=token_len,
                                 is_predict=True,
                                 data_nums=data_nums)

    def _get_labels(self):
        label_num_dict = {}
        # if os.path.exists(self.label_map_cache_path):
        #     label_map = load_dict(self.label_map_cache_path)
        # else:
        label_set = set()
        for path in [self.train_path, self.dev_path, self.test_path]:
            fp = open(path)
            samples = json.load(fp)
            for data in samples:
                sample = data
                for spo in sample["relation_list"]:
                    label_set.add(spo["predicate"])
                    if spo["predicate"] not in label_num_dict:
                        label_num_dict[spo["predicate"]] = 0
                    label_num_dict[spo["predicate"]] += 1
        label_set = sorted(label_set)
        labels = list(label_set)
        label_map = {idx: label for idx, label in enumerate(labels)}
        # write_dict(self.label_map_cache_path, label_map)
        # fp.close()
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, token_len, is_predict, data_nums):
        outputs = {
            'text': [],
            "spo_list": [],
            "spo_span_list": [],
            "tail_label": [],
        }
        token_len_big_than_100 = 0
        token_len_big_than_150 = 0
        max_token_len = 0
        max_data_nums = math.inf if data_nums == -1 else data_nums
        data_count = 0
        data = json.load(open(path))
        label_dict = {}
        for line in tqdm(data):
            if len(line["relation_list"]) == 0:
                continue
            text = line["text"]
            input_ids = self.tokenizer.encode(text)
            token_encode_len = len(input_ids)
            if token_encode_len > 100+2:
                token_len_big_than_100 += 1
            if token_encode_len > 150+2:
                token_len_big_than_150 += 1
            max_token_len = max(max_token_len, token_encode_len)
            if token_encode_len > token_len + 2:
                continue
            spo_list = set()
            spo_span_list = set()
            # [CLS] texts [SEP] rels
            tail_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels])

            e2e_set = set()
            h2r_dict = dict()
            t2r_dict = dict()
            spo_tail_set = set()
            spo_tail_text_set = set()
            spo_text_set = set()
            for spo in line["relation_list"]:
                pred = spo["predicate"]
                if pred not in label_dict:
                    label_dict[pred] = 0
                label_dict[pred] += 1
                sub = spo["subject"]
                obj = spo["object"]
                spo_list.add((sub, pred, obj))
                sub_span = spo["subj_tok_span"]
                obj_span = spo["obj_tok_span"]
                pred_idx = self.pred2idx[pred]
                plus_token_pred_idx = pred_idx + token_len + 2
                spo_span_list.add((tuple(sub_span), pred_idx, tuple(obj_span)))

                h_s, h_e = sub_span
                t_s, t_e = obj_span
                # Entity-Entity Interaction
                tail_matrix[h_e][t_e] = 1
                tail_matrix[t_e][h_e] = 1
                # Subject-Relation Interaction
                tail_matrix[h_e][plus_token_pred_idx] = 1
                # Relation-Object Interaction
                tail_matrix[plus_token_pred_idx][t_e] = 1
                
                spo_tail_set.add((h_e, plus_token_pred_idx, t_e))
                spo_tail_text_set.add((
                    self.tokenizer.decode(input_ids[h_e]),
                    pred,
                    self.tokenizer.decode(input_ids[t_e])
                ))
                spo_text_set.add((
                    self.tokenizer.decode(input_ids[h_s+1:h_e+1]),
                    pred,
                    self.tokenizer.decode(input_ids[t_s+1:t_e+1])
                ))
                e2e_set.add((h_e, t_e))
                e2e_set.add((t_e, h_e))

            outputs["text"].append(text)
            outputs["spo_list"].append(list(spo_list))
            outputs["spo_span_list"].append(list(spo_span_list))
            outputs["tail_label"].append(tail_matrix)

            data_count += 1
            if data_count >= max_data_nums:
                break

        print(max_token_len)
        print(f"more than 100: {token_len_big_than_100}")
        print(f"more than 150: {token_len_big_than_150}")
        return outputs

