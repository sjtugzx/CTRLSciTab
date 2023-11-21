import random
from torch.nn.utils import rnn
import progressbar
from Modules.utlis import *
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, T5TokenizerFast
from transformers import GPT2Tokenizer, GPT2Config, GPT2TokenizerFast
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


UNSEEN_SLOT_KEY, EOS, SEP, PAD, BOS = '__None__', '__EOS__', '__SEP__', '__PAD__', \
                                           '__BOS__'


class Data:
    def __init__(self, train_data_dict, dev_data_dict, max_table_len, max_content_plan_len,
                 max_tgt_len, model_name, special_token_path, min_slot_key_cnt):


        self.max_table_len, self.max_content_plan_len, self.max_tgt_len = \
            max_table_len, max_content_plan_len, max_tgt_len
        self.special_token_list, self.special_token_dict = [], {}
        with open(special_token_path, 'r', encoding='utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_special_token = l.strip('\n').split()[0]
                cnt = int(l.strip('\n').split()[1])
                if cnt >= min_slot_key_cnt:
                    self.special_token_list.append(one_special_token)
                    self.special_token_dict[one_special_token] = 1
                else:
                    pass
        print('Number of Special Token is %d' % len(self.special_token_list))

        self.model_name = model_name
        print("model_name: ", model_name)
        if model_name == "facebook/bart-base":
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
            self.decode_tokenizer = BartTokenizer.from_pretrained(model_name)
            print('original vocabulary Size %d' % len(self.tokenizer))
            self.tokenizer.add_tokens(self.special_token_list)
            self.decode_tokenizer.add_tokens(self.special_token_list)
            print('vocabulary size after extension is %d' % len(self.tokenizer))
            self.sep_idx = self.tokenizer.convert_tokens_to_ids([SEP])[0]
            self.eos_idx = self.tokenizer.convert_tokens_to_ids([EOS])[0]

        if model_name == "t5-small":
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            config = T5Config.from_pretrained(model_name)
            self.bos_token_id = config.decoder_start_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
            # self.tokenizer.add_tokens(self.special_token_list + [PAD])
            self.decode_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.decode_tokenizer.add_tokens(self.special_token_list + [PAD])
            print('original vocabulary Size %d' % len(self.tokenizer))
            self.tokenizer.add_tokens(self.special_token_list)
            self.decode_tokenizer.add_tokens(self.special_token_list)
            print('vocabulary size after extension is %d' % len(self.tokenizer))
            self.sep_idx = self.tokenizer.convert_tokens_to_ids([SEP])[0]
            self.eos_idx = self.tokenizer.convert_tokens_to_ids([EOS])[0]
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids([PAD])[0]

        print('Start loading training data...')
        # self.train_path_text_list, self.train_path_id_list, self.train_src_text_list
        self.train_table_id_list, self.train_tgt_id_list, self.train_reference_text_list, self.train_table_text_list \
            = self.load_data(train_data_dict)

        print('Training data loaded.')

        print('Start loading validation data...')
        self.dev_table_id_list, self.dev_tgt_id_list, self.dev_reference_text_list, \
        self.dev_table_text_list = self.load_data(dev_data_dict)

        print('Validation data loaded.')

        self.train_num, self.dev_num = len(self.train_table_id_list), len(self.dev_table_id_list)
        print('train number is %d, dev number is %d' % (self.train_num, self.dev_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]

        self.dev_current_idx = 0

    def load_one_text_id(self, text, max_len):
        """
        convert text to id
        :param text: token text
        :param max_len: max length of text list
        :return: token id list
        """

        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True,
                                             add_special_tokens=False)[:max_len]

        return text_id_list

    def load_text_id_list(self, path, max_len):
        """
        load text from file and convert text to id list
        :param path: file path
        :param max_len: max length of text
        :return: id list of all file
        """
        text_list = []
        with open(path, 'r', encoding='utf8') as i:
            lines = i.readlines()
            if self.model_name == 't5-small':
                prefix = 'Table to text: '
                for l in lines:
                    text_list.append(prefix + l.strip('\n'))
            else:
                for l in lines:
                    text_list.append(l.strip('\n'))

        res_id_list = []
        idx = 0
        for text in text_list:
            # p.update(idx + 1)
            one_id_list = self.load_one_text_id(text, max_len)
            res_id_list.append(one_id_list)
            idx += 1
        # p.finish()
        return res_id_list


    def load_text_id_list_(self, text_list, max_len):
        """
        load text id list from a text_list
        :param text_list: text list
        :param max_len: max length of text
        :return: id list
        """
        # print("text_id_list")
        text_list = text_list

        res_id_list = []
        idx = 0
        for text in text_list:

            one_id_list = self.load_one_text_id(text, max_len)

            # print("len(one_text_list)", len(text))
            # print("len(one_id_list)", len(one_id_list))
            res_id_list.append(one_id_list)
            idx += 1

        # print("len(text_list): ", len(text_list))
        # print("len(id_list): ", len(res_id_list))

        return res_id_list

    def load_data(self, data_dict):

        table_text_path = data_dict['table_text_path']


        print('Loading Table Data...')
        table_id_list = self.load_text_id_list(table_text_path, self.max_table_len)
        table_id_list = [[self.sep_idx] + one_id_list for one_id_list in table_id_list]


        print('Loading Reference Data...')
        reference_sentence_path = data_dict['reference_sentence_path']
        tgt_id_list = self.load_text_id_list(reference_sentence_path, self.max_tgt_len)
        assert len(table_id_list) == len(tgt_id_list)

        tgt_id_list = [[self.bos_token_id] + item + [self.eos_token_id] for item in tgt_id_list]

        # get reference text list
        reference_text_list = self.get_text_list(reference_sentence_path)

        # source text
        table_text_list = self.get_text_list(table_text_path)

        return table_id_list,tgt_id_list, reference_text_list, table_text_list

    @staticmethod
    def get_text_list(reference_sentence_path):
        reference_text_list = []
        with open(reference_sentence_path, 'r', encoding='utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_text = l.strip('\n')
                reference_text_list.append(one_text)
        return reference_text_list


    def process_source_tensor(self, batch_src_id_list):
        """
        get tensor via source id list as training input
        :param batch_src_id_list: batch source id list
        :return: batch source tensor and batch mask tensor
        """
        batch_src_tensor_list = [torch.LongTensor(item) for item in batch_src_id_list]
        batch_src_tensor = rnn.pad_sequence(batch_src_tensor_list, batch_first=True,
                                            padding_value=self.pad_token_id)
        # ---- compute src mask ---- #
        batch_src_mask = torch.ones_like(batch_src_tensor)
        batch_src_mask = batch_src_mask.masked_fill(batch_src_tensor.eq(self.pad_token_id),
                                                    0.0).type(torch.FloatTensor)
        return batch_src_tensor, batch_src_mask

    def process_one_source_tensor(self, one_src_id_list):
        one_src_tensor_list = [torch.LongTensor(one_src_id_list)]
        one_src_tensor = rnn.pad_sequence(one_src_tensor_list, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
        # ---- compute src mask ---- #
        one_src_mask = torch.ones_like(one_src_tensor)
        one_src_mask = one_src_mask.masked_fill(one_src_tensor.eq(self.pad_token_id),
                                                0.0).type(torch.FloatTensor)
        return (one_src_tensor, one_src_mask)

    def process_decoder_tensor(self, batch_tgt_id_list):
        """
        get decoder tensor from target id list as training input
        :param batch_tgt_id_list: batch target id list
        :return: batch labels
        """
        batch_tgt_tensor = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor = rnn.pad_sequence(batch_tgt_tensor, batch_first=True,
                                            padding_value=self.pad_token_id)
        batch_labels = batch_tgt_tensor
        batch_labels[batch_labels[:, :] == self.tokenizer.pad_token_id] = -100
        return batch_labels

    def get_next_train_batch(self, batch_size):

        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_table_id_list, batch_tgt_id_list, batch_reference_text_list, \
        batch_src_id_list, batch_table_text_list, batch_src_text_list = [], [], [], [], [], []

        for idx in batch_idx_list:
            one_table_id_list = self.train_table_id_list[idx]
            one_tgt_id_list = self.train_tgt_id_list[idx]

            batch_table_id_list.append(one_table_id_list)
            batch_tgt_id_list.append(one_tgt_id_list)
            batch_src_id_list.append(one_table_id_list)


            one_reference_text = self.train_reference_text_list[idx]
            batch_reference_text_list.append(one_reference_text)

            one_table_text = self.train_table_text_list[idx]
            batch_table_text_list.append(one_table_text)
            one_src_text = one_table_text
            batch_src_text_list.append(one_src_text)


        batch_table_tensor, batch_table_mask = self.process_source_tensor(
            batch_table_id_list)


        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_tgt_tensor = self.process_decoder_tensor(batch_tgt_id_list)




        return (batch_table_tensor, batch_table_mask), \
               (batch_src_tensor, batch_src_mask), \
               batch_tgt_tensor, \
               (batch_reference_text_list, batch_table_text_list, batch_src_text_list), \
               (batch_table_id_list, batch_src_id_list)

    def get_next_dev_batch(self, batch_size):

        batch_table_id_list, batch_tgt_id_list, batch_reference_text_list, \
        batch_src_id_list, batch_table_text_list, batch_src_text_list = [], [], [], [], [], []


        if self.dev_current_idx + batch_size < self.dev_num - 1:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i

                one_table_id_list = self.dev_table_id_list[curr_idx]
                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]

                batch_table_id_list.append(one_table_id_list)
                batch_src_id_list.append(one_table_id_list)
                batch_tgt_id_list.append(one_tgt_id_list)

                one_reference_text = self.dev_reference_text_list[curr_idx]
                one_table_text = self.dev_table_text_list[curr_idx]

                one_src_text = one_table_text
                batch_src_text_list.append(one_src_text)

                batch_reference_text_list.append(one_reference_text)
                batch_table_text_list.append(one_table_text)

            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                if curr_idx > self.dev_num - 1:
                    curr_idx = 0
                    self.dev_current_idx = 0
                else:
                    pass
                one_table_id_list = self.dev_table_id_list[curr_idx]
                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]

                batch_table_id_list.append(one_table_id_list)

                batch_src_id_list.append(one_table_id_list)
                batch_tgt_id_list.append(one_tgt_id_list)

                one_reference_text = self.dev_reference_text_list[curr_idx]
                one_table_text = self.dev_table_text_list[curr_idx]
                one_src_text = one_table_text
                batch_src_text_list.append(one_src_text)

                batch_reference_text_list.append(one_reference_text)
                batch_table_text_list.append(one_table_text)
            self.dev_current_idx = 0

        batch_table_tensor, batch_table_mask = self.process_source_tensor(
            batch_table_id_list)
        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_tgt_tensor = self.process_decoder_tensor(batch_tgt_id_list)


        return (batch_table_tensor, batch_table_mask), \
               (batch_src_tensor, batch_src_mask), \
               batch_tgt_tensor, \
               (batch_reference_text_list, batch_table_text_list, batch_src_text_list), \
               (batch_table_id_list, batch_src_id_list)
