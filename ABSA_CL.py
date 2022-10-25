import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import EntropyLoss
import math
from pytorch_transformers import BertPreTrainedModel, BertModel, BertConfig

class BiLSTM(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=1, batch_first = True):
        super(BiLSTM, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_feature,
            hidden_size=out_feature,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=0.5
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device))

    def forward(self, input):
        batch_size, seq_len, hidden_size = input.shape
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(batch_size, seq_len, self.out_feature * 2)

class BiLSTMEncoder(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=1, batch_first = True):
        super(BiLSTMEncoder, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_feature,
            hidden_size=out_feature,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=0.5
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device))

    def forward(self, input):
        batch_size, seq_len, hidden_size = input.shape
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(batch_size, seq_len, self.out_feature * 2), hidden

class Transformer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Transformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, input):
        output = self.transformer(input.transpose(0, 1))
        return output.transpose(0, 1)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(self, sequence, input):
        output = self.transformer(sequence.transpose(0, 1), input.transpose(0, 1))
        return output.transpose(0, 1)

class ABSA_CL(nn.Module):
    def __init__(self, tag_size=100, aspect_tag_size=2, bert_path=None, embedding='embedding', encoder="transformer",
                 num_layers=2, criteria_size=2, multi_criteria=True, adversary=True, adv_coefficient=1, pooling="max_pooling"):
        super(ABSA_CL, self).__init__()
        self.tag_size = tag_size
        self.criteria_size = criteria_size
        self.num_layers = num_layers
        self.embedding_type = embedding
        self.encoder_type = encoder
        self.multi_criteria = multi_criteria
        self.adversary = adversary
        self.adv_coefficient = adv_coefficient
        bert_config = BertConfig.from_pretrained(bert_path)
        hidden_size = bert_config.hidden_size
        self.vocab_size = bert_config.vocab_size
        self.config = bert_config
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.pooling = pooling
        self.aspect_tag_size = aspect_tag_size
        if multi_criteria is False:
            self.criteria_size = 1
        if encoder == "transformer":
            self.bert = BertModel.from_pretrained(bert_path)
            self.private_encoder = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=self.num_layers)
            self.private_encoder_cl = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=self.num_layers)
            
            self.classifier_sa = torch.nn.Linear(hidden_size*2, tag_size)
            self.CL_decoder = TransformerDecoder(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=self.num_layers)
            self.classifier_cl = torch.nn.Linear(hidden_size, self.vocab_size)
            if self.adversary:
                self.classifier_at = torch.nn.Linear(hidden_size, 2)
        elif encoder == "BiLSTM":
            self.bert = BertModel.from_pretrained(bert_path)
            self.private_encoder = BiLSTM(hidden_size, hidden_size, num_layers=self.num_layers)
            bilstm_hidden_size = int(hidden_size/2)
            self.private_encoder_cl = BiLSTMEncoder(hidden_size, bilstm_hidden_size, num_layers=self.num_layers)
            
            self.classifier_sa = torch.nn.Linear(hidden_size*3, tag_size)
            self.CL_decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=0.5)
            self.classifier_cl = torch.nn.Linear(hidden_size, self.vocab_size)
            if self.adversary:
                self.classifier_at = torch.nn.Linear(hidden_size, 2)
        else:
            raise Exception("Invalid encoder")
            
    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output
    
    def cons_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len = sequence_output.shape
        for i in range(batch_size):
            for j in range(max_len):
                if valid_ids[i][j] == 1:
                    sequence_output[i][j] = 1
        return sequence_output
        
    def extract_entity(self, sequence, e_mask):
        if self.pooling == "max_pooling":
            return self.max_pooling(sequence, e_mask)
        elif self.pooling == "avg_pooling":
            return self.avg_pooling(sequence, e_mask)
            
    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def avg_pooling(self, sequence, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(
            extended_e_mask.float(), sequence.float()).squeeze(1)
        entity_output = extended_e_mask.float() / (e_mask != 0).sum(dim=1).unsqueeze(1)
        return entity_output.type_as(sequence)

    def forward(self, input_ids, criteria_index, token_type_ids=None, attention_mask=None, labels=None, labels_AR=None, aspect_mask=None,
                valid_ids=None, b_use_valid_filter=False):
        if criteria_index not in [0, 1]:
            raise Exception("criteria_index Invalid")

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids)
        shared_output = sequence_output

        if valid_ids is not None:
            shared_output = self.get_valid_seq_output(shared_output, valid_ids)
        
        tmp_shared_output = shared_output
        if self.encoder_type == "BiLSTM":
            private_output_cl, hidden = self.private_encoder_cl(tmp_shared_output)
        elif self.encoder_type == "transformer":
            private_output_cl = self.private_encoder_cl(tmp_shared_output)
       
        aspect_h_p = self.extract_entity(private_output_cl, aspect_mask)
        aspect_h_p = aspect_h_p.unsqueeze(1)
        
        private_output = torch.cat([shared_output, aspect_h_p], dim=1)
        private_output = self.private_encoder(private_output)

        if self.pooling == "max_pooling":
            shared_output = torch.max(shared_output, 1)[0]
            private_output = torch.max(private_output, 1)[0]
        elif self.pooling == "avg_pooling":
            shared_output = torch.mean(shared_output, 1)
            private_output = torch.mean(private_output, 1)

        sequence_output = torch.cat([shared_output, private_output], dim=-1)
        sequence_output = self.dropout(sequence_output)

        logits_sa = self.classifier_sa(sequence_output)

        if criteria_index:
            if self.encoder_type == "BiLSTM":
                num_layers_double, batch_size, out_feature = hidden[0].shape
                hidden_0 = hidden[0].contiguous().view(self.num_layers, batch_size, out_feature * 2)
                hidden_1 = hidden[1].contiguous().view(self.num_layers, batch_size, out_feature * 2)
                logits_cl, hidden_ = self.CL_decoder(tmp_shared_output, (hidden_0, hidden_1))
            elif self.encoder_type == "transformer":
                logits_cl = self.CL_decoder(tmp_shared_output, private_output_cl)
            logits_cl = self.classifier_cl(logits_cl)
            if self.adversary:
                logits_at = self.classifier_at(tmp_shared_output)
            

        tag_seq = torch.argmax(logits_sa, -1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_sa = loss_fct(logits_sa.view(-1, self.tag_size), labels.view(-1))
            if criteria_index:
                loss_cl = loss_fct(logits_cl.view(-1, self.vocab_size), input_ids.view(-1))
                ar_results = torch.argmax(F.log_softmax(logits_cl, dim=2), dim=2)
                if self.adversary:
                    comparison_result = (input_ids == ar_results)
                    comparison_result = comparison_result.long()
                    loss_at = loss_fct(logits_at.view(-1, 2), comparison_result.view(-1))
                    loss_at = torch.sigmoid(loss_at)
                    return logits_sa, loss_sa, loss_cl, loss_at
                else:
                    return logits_sa, loss_sa, loss_cl, 1
            else:
                return logits_sa, loss_sa, 0, 0
        else:
            if self.encoder_type == "BiLSTM":
                num_layers_double, batch_size, out_feature = hidden[0].shape
                hidden_0 = hidden[0].contiguous().view(self.num_layers, batch_size, out_feature * 2)
                hidden_1 = hidden[1].contiguous().view(self.num_layers, batch_size, out_feature * 2)
                logits_cl, hidden_ = self.CL_decoder(tmp_shared_output, (hidden_0, hidden_1))
            elif self.encoder_type == "transformer":
                logits_cl = self.CL_decoder(tmp_shared_output, private_output_cl)
            logits_cl = self.classifier_cl(logits_cl)
            generated_seq = torch.argmax(logits_cl, -1)
            return tag_seq, generated_seq
