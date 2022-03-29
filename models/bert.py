import torch.nn as nn
from transformers import AutoModel

class BERT_model(nn.Module):
    
    def __init__(self, freeze_bert=False, nclasses=3, name="bert-base-uncased", dropout=0.1):
        super(BERT_model, self).__init__()

        self.bert_layer = AutoModel.from_pretrained(name)

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, nclasses)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor of token ids
            -attn_masks : Tensor of attention masks (for non-padded values)
            -token_type_ids : Tensor of token type ids to identify sentence1 and sentence2
        '''
        out = self.bert_layer(input_ids, attn_masks, token_type_ids)
        pooler_output = out.pooler_output
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits
