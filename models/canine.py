import torch.nn as nn
from transformers import CanineModel

class CANINE_model(nn.Module):
    
    def __init__(self, freeze_canine=False, nclasses=3, name = "google/canine-c", dropout=0.3):
        super(CANINE_model, self).__init__()
        self.canine_layer = CanineModel.from_pretrained(name)

        if freeze_canine:
            for p in self.canine_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, nclasses)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor of token ids
            -attn_masks : Tensor of attention masks (for non-padded values)
            -token_type_ids : Tensor of token type ids to identify sentence1 and sentence2
        '''

        out = self.canine_layer(input_ids, attn_masks, token_type_ids)
        pooler_output = out.pooler_output
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits