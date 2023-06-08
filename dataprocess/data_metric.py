from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, f1_score
from transformers import EvalPrediction, set_seed, AutoTokenizer
import numpy as np

# tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
tokenizer = AutoTokenizer.from_pretrained("lemon234071/t5-base-Chinese")

def calclulate_f1(statics_dict):
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {"prec": prec, "recall": recall, "f1": f1}

def kg_gent_metric(p:EvalPrediction):
    label_ids = p.label_ids
    preds = p.predictions
    preds_ids = np.argmax(preds, axis=-1)
    correct, gold, pred = 0, 0, 0
    first_level_correct, first_level_gold, first_level_pred = 0, 0, 0
    for l_ids, p_ids in zip(label_ids, preds_ids):
        # TODO: The decode procedure should be more robust
        l_text = tokenizer.decode(l_ids)
        p_text = tokenizer.decode(p_ids)
        correct += len(set(l_text.split("。")[:-1]).intersection(set(p_text.split("。")[:-1])))
        gold += len(l_text.split("。"))
        pred += len(p_text.split("。"))
        first_level_l_set = set()
        for item in l_text.split("。"):
            if "——" not in item:
                continue
            first_level_l_set.add((item.split("——")[0], item.split("——")[1].split(":")[-1]))
        first_level_p_set = set()
        for item in p_text.split("。"):
            if "——" not in item:
                continue
            first_level_p_set.add((item.split("——")[0], item.split("——")[1].split(":")[-1]))
        first_level_correct += len(first_level_l_set.intersection(first_level_p_set))
        first_level_gold += len(first_level_l_set)
        first_level_pred += len(first_level_p_set)
    # return both results
    results = calclulate_f1({"c": correct, "p": pred, "g": gold})
    results["first_level"] = calclulate_f1({"c": first_level_correct, "p": first_level_pred, "g": first_level_gold})
    return results


def unirel_metric(p: EvalPrediction):
    token_len = 102
    tail_labels = p.label_ids
    tail_preds = p.predictions
    tail_acc, tail_recall, tail_f1, _ = precision_recall_fscore_support(
        y_pred=tail_preds.reshape(-1),
        y_true=tail_labels.reshape(-1),
        labels=[1],
        average='micro')
 
    return {
        "acc": tail_acc,
        "recall": tail_recall,
        "f1": tail_f1,
    }

def type_cls_metric(p: EvalPrediction):
    token_len = 512
    type_labels, postions = p.label_ids
    type_preds,_ = p.predictions
    start_mask = postions == 0
    middle_mask = postions == 1
    end_mask = postions == 2
    start_pred = type_preds[start_mask]
    start_label = type_labels[start_mask]
    middle_pred = type_preds[middle_mask]
    middle_label = type_preds[middle_mask]
    end_pred = type_preds[end_mask]
    end_label = type_labels[end_mask]
    results = {}
    acc, recall, f1, _ = precision_recall_fscore_support(
        y_pred=start_pred.reshape(-1),
        y_true=start_label.reshape(-1),
        labels=[1],
        average='micro')
    results["start_acc"]= acc
    results["start_rec"] = recall
    results["start_f1"]= f1
    
    acc, recall, f1, _ = precision_recall_fscore_support(
        y_pred=middle_pred.reshape(-1),
        y_true=middle_label.reshape(-1),
        labels=[1],
        average='micro')
    results["middle_acc"]= acc
    results["middle_rec"] = recall
    results["middle_f1"]= f1

    acc, recall, f1, _ = precision_recall_fscore_support(
        y_pred=end_pred.reshape(-1),
        y_true=end_label.reshape(-1),
        labels=[1],
        average='micro')
    results["end_acc"]= acc
    results["end_rec"] = recall
    results["end_f1"]= f1
    
 
    return results
