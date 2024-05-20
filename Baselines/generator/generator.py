from torch import nn
from transformers import BartForConditionalGeneration, BartConfig
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel

from io import open
import unicodedata
import string
import re
import random

import torch
from torch import optim
import torch.nn.functional as F

import os


class Generator(nn.Module):
    def __init__(self, model_name, tokenizer, max_decode_len, dropout):
        super().__init__()
        self.tokenizer = tokenizer  # tokenizer with extended vocabulary
        self.max_decode_len = max_decode_len
        self.model_name = model_name

        print("Initializing Huggingface "+ model_name+ ' model...')
        if model_name == 'facebook/bart-base':
            bart_config = BartConfig.from_pretrained(model_name)
            bart_config.__dict__["dropout"] = dropout
            bart_config.output_attentions = True
            bart_config.return_dict = True
            bart_config.output_hidden_states = True

            self.model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
            self.model.config.max_position_embeddings

        if model_name == "gpt2":
            gpt_config = GPT2Config.from_pretrained(model_name)
            gpt_config.__dict__["dropout"] = dropout
            gpt_config.output_attentions = True
            gpt_config.return_dict = True
            gpt_config.max_length = 1024
            self.model = GPT2LMHeadModel.from_pretrained(model_name, config=gpt_config)

        if model_name =='t5-small':
            t5_config = T5Config.from_pretrained(model_name)
            t5_config.__dict__["dropout"] = dropout
            t5_config.output_attentions = True
            t5_config.return_dict = True
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=t5_config)
            # self.model.config.max_position_embeddings

        print('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.vocab_size = len(self.tokenizer)
        self.logsftmax = nn.LogSoftmax(dim=-1)
        self.padding_idx = self.tokenizer.pad_token_id



    def forward(self, src_input, src_mask, tgt_output):
        src_mask = src_mask.type(src_input.type())
        if self.model_name == 'gpt2':
            outputs = self.model(input_ids=src_input, labels=src_mask)
        else:
            outputs = self.model(input_ids=src_input, attention_mask=src_mask, labels=tgt_output,
                                 output_attentions=True)
        # loss = outputs[0]
        return outputs

    def generate(self, src_input, src_mask):
        result_list = []
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, num_beams=4,
                                      max_length=self.max_decode_len)
        for predicted_ids in outputs:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
        return result_list

    def save_model(self, model_path):
        path = os.path.join(model_path, "model_files")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the  tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs