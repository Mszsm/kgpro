import torch
import dataclasses
from typing import Optional
from torch import nn

from transformers import (PreTrainedModel, BertPreTrainedModel, BertConfig, AutoModel, T5ForConditionalGeneration,
                          BertTokenizerFast)
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead, BertForMaskedLM, BertLMHeadModel
from .modify_bert import BertModel

@dataclasses.dataclass
class UniRelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    tail_preds: Optional[torch.FloatTensor] = None

@dataclasses.dataclass
class TypeClsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    preds: Optional[torch.FloatTensor] = None
    position: Optional[torch.FloatTensor] = None


class KGProjectModelGen(PreTrainedModel):
    def __init__(self, config, model_dir=None):
        super(KGProjectModelGen, self).__init__(config)
        self.config = config
        if model_dir is None:
            self.m = T5ForConditionalGeneration(config)
        else:
            self.m = T5ForConditionalGeneration.from_pretrained(model_dir, config=config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None
    ):
        outputs = self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=False,
        )
        return {
            "loss": outputs["loss"],
            "logits": outputs["logits"],
        }


class UniRelModel(BertPreTrainedModel):
    """
    Model for learning Interaction Map
    """
    def __init__(self, config, model_dir=None):
        super(UniRelModel, self).__init__(config=config)
        self.config = config
        if model_dir is not None:
            self.bert = BertModel.from_pretrained(model_dir, config=config)
        else:
            self.bert = BertModel(config)
        
        # Easy debug
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-cased", do_basic_tokenize=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Abaltion experiment
        if config.is_additional_att or config.is_separate_ablation:
            self.key_linear = nn.Linear(768, 64)
            self.value_linear = nn.Linear(768, 64)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        token_len_batch=None,
        labels=None,
        tail_label=None,
        head_label=None,
        span_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        TOKEN_LEN = token_len_batch[0]
        tail_logits = None
        # For span extraction
        head_logits= None
        span_logits = None
        # 
        if not self.config.is_separate_ablation:
            # Encoding the sentence and relations simultaneously, and using the inside Attention score
            outputs = self.bert(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=False,
                            output_attentions_scores=True,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
            attentions_scores = outputs.attentions_scores[-1]
            BATCH_SIZE, ATT_HEADS, ATT_LEN, _ = attentions_scores.size()
            ATT_LAYERS = len(attentions_scores)
            tail_logits = nn.Sigmoid()(
                    attentions_scores[:, :, :, :].mean(1)
                )
        else:
            # Encoding the sentence and relations in a separate manner, and add another attention layer
            text_outputs = self.bert(
                            input_ids=input_ids[:, :TOKEN_LEN],
                            attention_mask=attention_mask[:, :TOKEN_LEN],
                            token_type_ids=token_type_ids[:, :TOKEN_LEN],
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=None,
                            output_attentions=False,
                            output_attentions_scores=False,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
            pred_outputs = self.bert(
                            input_ids=input_ids[:, TOKEN_LEN:],
                            attention_mask=attention_mask[:, TOKEN_LEN:],
                            token_type_ids=token_type_ids[:, TOKEN_LEN:],
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=None,
                            output_attentions=False,
                            output_attentions_scores=False,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

            last_hidden_state = torch.cat((text_outputs.last_hidden_state, pred_outputs.last_hidden_state), -2)
            key_layer = self.key_linear(last_hidden_state)
            value_layer = self.value_linear(last_hidden_state)
            tail_logits = nn.Sigmoid()(torch.matmul(key_layer, value_layer.permute(0, 2,1)))

        loss = None

        if tail_label is not None:
            tail_loss = nn.BCELoss()(tail_logits.float().reshape(-1),
                                    tail_label.reshape(-1).float())
            if loss is None:
                loss = tail_loss
            else:
                loss += tail_loss
        if tail_logits is not None:
            tail_predictions = tail_logits > self.config.threshold
        else:
            tail_predictions = None
        return UniRelOutput(
            loss=loss,
            tail_preds=tail_predictions,
        )

class TypeClsModel(BertPreTrainedModel):
    
    def __init__(self, config, model_dir=None):
        super(TypeClsModel, self).__init__(config=config)
        self.config = config
        if model_dir is not None:
            self.bert = AutoModel.from_pretrained(model_dir, config=config)
        else:
            self.bert = AutoModel(config)
        
        # Easy debug
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-chinese", do_basic_tokenize=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(768,13)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        token_len_batch=None,
        labels=None,
        type_label=None,
        head_label=None,
        span_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
        position = None,
    ):
        
        tail_logits = None
        # For span extraction
        head_logits= None
        span_logits = None
        
            # Encoding the sentence and relations simultaneously, and using the inside Attention score
        outputs = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=False,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict)
        # pooler_output = outputs.
        avg_states = outputs.last_hidden_state.mean(1)
        # print(last_hidden_state.size())
        output = self.cls(avg_states)

        # print(output.size())
        tail_logits = nn.Softmax(dim=1)(output)
        # print(tail_logits)
            # print(output.size())
            

        
            
            # key_layer = self.key_linear(last_hidden_state)
            # value_layer = self.value_linear(last_hidden_state)
            # tail_logits = nn.Sigmoid()(torch.matmul(key_layer, value_layer.permute(0, 2,1)))
        # print(len(tail_logits[0][0]))
        loss = None
        if type_label is not None:
            type_loss = nn.CrossEntropyLoss()(output.reshape(-1,13),
                                    type_label.reshape(-1))
            if loss is None:
                loss = type_loss
            else:
                loss += type_loss
        if  tail_logits is not None:
            _,predictions = torch.max(tail_logits, dim=1)
            # print(predictions)
            # print(predictions.size())
        else:
            predictions = None
        # print(predictions.size())
        return TypeClsOutput(
            loss=loss,
            preds=predictions,
            position=position
        )