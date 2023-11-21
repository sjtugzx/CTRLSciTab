import subprocess
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm
import pickle
from transformers import BartTokenizer, BartTokenizerFast
import torch
from transformers import AutoFeatureExtractor
from torchvision import transforms
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn as nn
import sacrebleu as scb
# from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict


def eval_totto(prediction_path, target_path):
    command = 'bash ../language/totto/totto_eval.sh --prediction_path ' + prediction_path \
              + ' --target_path ' + target_path
    # command = 'bash ' + eval_script_path+' --prediction_path ' + prediction_path \
    #           + ' --target_path ' + target_path
    try:
        result = subprocess.run(command,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    res = result.stdout.decode("utf-8")
    content_list = res.split(r'BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.10 = ')
    overall_bleu = float(content_list[1].split()[0])
    overlap_bleu = float(content_list[2].split()[0])
    nonoverlap_bleu = float(content_list[3].split()[0])
    return overall_bleu, overlap_bleu, nonoverlap_bleu


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--save_model_name', type=str)
    parser.add_argument('--save_output', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--process_mode', type=str, default="AVG_POOLING_LAYER")
    parser.add_argument('--train_table_text_path', type=str, default="../data/train/totto_reasoning_table.txt")
    parser.add_argument('--train_table_header_path', type=str, default='')
    parser.add_argument('--train_content_text_path', type=str, default='../data/train/totto_reasoning_used_headers.txt')
    parser.add_argument('--train_table_highlight_path', type=str, default='../data/train/totto_reasoning_highlight_headers.txt')
    parser.add_argument('--train_reference_sentence_path', type=str, default='../data/train/totto_reasoning_reference.txt')
    parser.add_argument('--train_path_text_path', type=str)
    parser.add_argument('--test_table_text_path', type=str)
    parser.add_argument('--test_table_header_path', type=str)
    parser.add_argument('--test_content_text_path', type=str)
    parser.add_argument('--test_table_highlight_path', type=str)
    parser.add_argument('--test_reference_sentence_path', type=str)
    parser.add_argument('--test_path_text_path', type=str)
    parser.add_argument('--dev_table_text_path', type=str, default='../data/dev/totto_reasoning_table.txt')
    parser.add_argument('--dev_table_header_path', type=str)
    parser.add_argument('--dev_content_text_path', type=str, default='../data/dev/totto_reasoning_used_headers.txt')
    parser.add_argument('--dev_reference_sentence_path', type=str, default='../data/dev/totto_reasoning_reference.txt')
    parser.add_argument('--dev_path_text_path', type=str)
    parser.add_argument('--dev_table_highlight_path', type=str, default='../data/dev/totto_reasoning_highlight_headers.txt')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--pretrained_ckpt_path', type=str)
    parser.add_argument('--planner_pretrained_ckpt_path', type=str)
    parser.add_argument('--finetune_generator_ckpt_path', type=str)
    parser.add_argument('--backbone_generator_ckpt_path', type=str)
    parser.add_argument('--special_token_path', type=str, default='../DataAnalysis/totto_col_header_vocab.txt')
    parser.add_argument('--dev_reference_path', type=str, default='../data/raw_data/totto_dev_data.jsonl')
    parser.add_argument('--max_table_len', type=int, default=256)
    parser.add_argument('--max_content_plan_len', type=int, default=128)
    parser.add_argument('--max_tgt_len', type=int, default=256)
    parser.add_argument('--min_slot_key_cnt', type=int, default=10)
    parser.add_argument('--generator_model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--planner_model_name', type=str, default='microsoft/resnet-50')
    parser.add_argument('--max_decode_len', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ckpt_path', type=str, default=r'./ckpt/bart_pretrain/')
    parser.add_argument('--inference_output_file', type=str)


    return parser.parse_args()


def map_cuda(tensor_item, device, is_cuda):
    res_list = []
    if is_cuda:
        res_list.append(tensor_item[0].cuda(device))
        res_list.append(tensor_item[1].cuda(device))
    else:
        res_list = tensor_item
    return res_list


def map_cuda_(tensor_list, device, is_cuda):
    if is_cuda:
        res_list = tensor_list.cuda(device)
    else:
        res_list = tensor_list
    return res_list


def pickle_write(file_path, data):
    f = open(file_path, 'wb')
    pickle.dump(data, f)
    f.close()


def pickle_read(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    return data


def process_attention_data_extractor(one_cross_attention):
    # print("process_attention_data")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    one_input = extractor(one_cross_attention, return_tensors='pt')[
        'pixel_values']
    # print(one_input)
    return one_input


def process_attention_data(one_cross_attention):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # print("process")
    one_input_tensor = preprocess(one_cross_attention).unsqueeze(0)
    # print(one_input_tensor)
    return one_input_tensor


def img_evaluate(predicted_labels, labels):
    # print(len(predicted_labels), len(labels))
    assert (len(predicted_labels) == len(labels))
    print(type(predicted_labels))
    print(type(labels))
    tp = (labels * predicted_labels).sum().to(torch.float32)
    tn = ((1 - labels) * (1 - predicted_labels)).sum().to(torch.float32)
    fp = ((1 - labels) * predicted_labels).sum().to(torch.float32)
    fn = (labels * (1 - predicted_labels)).sum().to(torch.float32)

    # print(tp, tn, fp, fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall_score = tp / (tp + fn)
    f1_score = 2 * (precision * recall_score) / (precision + recall_score)

    return precision, accuracy, recall_score, f1_score


def cvtOneTensorToPIL(inputTensor):
    # print(inputTensor.device)
    input_tensor = inputTensor
    im = Image.fromarray(np.uint8(cm.gist_earth(input_tensor) * 255), mode="RGB")
    return im


def initial_tokenizer(special_token_path="../../data/totto_col_header_vocab.txt",
                      min_slot_key_cnt=10):
    model_name = "facebook/bart-base"
    special_token_list, special_token_dict = [], {}
    with open(special_token_path, 'r', encoding='utf8') as i:
        lines = i.readlines()
        for l in lines:
            one_special_token = l.strip('\n').split()[0]
            cnt = int(l.strip('\n').split()[1])
            if cnt >= min_slot_key_cnt:
                special_token_list.append(one_special_token)
                special_token_dict[one_special_token] = 1
            else:
                pass
    print('Number of Special Token is %d' % len(special_token_list))
    tokenizer = BartTokenizerFast.from_pretrained(model_name)
    decode_tokenizer = BartTokenizer.from_pretrained(model_name)
    print('original vocabulary Size %d' % len(tokenizer))
    tokenizer.add_tokens(special_token_list)
    decode_tokenizer.add_tokens(special_token_list)
    print('vocabulary size after extension is %d' % len(tokenizer))
    return tokenizer, decode_tokenizer


def process_header_data(tokenizer, batch_size, batch_table_id_list, batch_header_id_list,
                        batch_highlight_id_list, batch_encoder_attentions, batch_cross_attentions):
    batch_header_cross_attention_list = []
    batch_header_cat_attention_list = []
    batch_labels = []
    batch_header_dict = {}
    batch_encoder_attentions = batch_encoder_attentions[-1]
    batch_cross_attentions = batch_cross_attentions[-1]

    for bitem in range(batch_size):

        src_text_id_list = batch_table_id_list[bitem]
        header_text_id_list = batch_header_id_list[bitem]
        highlight_text_id_list = batch_highlight_id_list[bitem]

        rm_special_token_ids = [tokenizer.convert_tokens_to_ids('<pad>'),
                                tokenizer.convert_tokens_to_ids('__SEP__'),
                                tokenizer.convert_tokens_to_ids('<pad>'),
                                tokenizer.convert_tokens_to_ids('__None__'),
                                tokenizer.convert_tokens_to_ids('__None__'),
                                tokenizer.convert_tokens_to_ids('__EOS__'),
                                tokenizer.convert_tokens_to_ids('__PAD__'),
                                1437]

        # all headers
        header_text_id_set = set(header_text_id_list)

        for special_token in rm_special_token_ids:
            if special_token in header_text_id_set:
                header_text_id_set.remove(special_token)

        # get header position in src text
        position = []
        for i, src_token_id in enumerate(src_text_id_list):
            if src_token_id in header_text_id_set:
                position.append(i)



        cross_attention = batch_cross_attentions[bitem]
        encoder_attention = batch_encoder_attentions[bitem]

        header_cross_attention_list = []
        header_cat_attention_list = []
        labels = []

        # extract all headers' attentions & labels
        for i in position:
            if (src_text_id_list[i]) in highlight_text_id_list:
                labels.append(1)
            else:
                labels.append(0)

            header_cross_attention_i = cross_attention[:, :, i]
            header_encoder_attention_i = encoder_attention[:, :, i]

            header_cross_attention_PIL = cvtOneTensorToPIL(header_cross_attention_i.detach())
            header_cross_attention_list.append(header_cross_attention_PIL)

            cat_attention = torch.cat((header_encoder_attention_i, header_cross_attention_i),
                                      dim=1)
            cat_attention_PIL = cvtOneTensorToPIL(cat_attention.detach())
            header_cat_attention_list.append(cat_attention_PIL)

        batch_header_cross_attention_list.append(header_cross_attention_list)
        batch_header_cat_attention_list.append(header_cat_attention_list)
        batch_labels.append(labels)

    batch_header_dict['crossAttentions'] = batch_header_cross_attention_list
    batch_header_dict['catAttentions'] = batch_header_cat_attention_list
    batch_header_dict['labels'] = batch_labels
    return batch_header_dict


def process_sentence_header_data(tokenizer, data_table_id_list, data_header_id_list,
                                 data_highlight_id_list, data_encoder_attentions,
                                 data_cross_attentions):

    data_labels = []
    data_header_dict = {}


    ln_encoder_attentions = []
    for i in data_encoder_attentions:
        i = i.cpu()
        layer_norm_i = nn.LayerNorm([i.shape[1], i.shape[2], i.shape[3]])
        data_encoder_attention = layer_norm_i(i)
        ln_encoder_attentions.append(data_encoder_attention)

    ln_cross_attentions = []
    for i in data_cross_attentions:
        i = i.cpu()
        layer_norm_i = nn.LayerNorm([i.shape[1], i.shape[2], i.shape[3]])
        data_cross_attention = layer_norm_i(i)
        ln_cross_attentions.append(data_cross_attention)

    src_text_id_list = data_table_id_list
    header_text_id_list = data_header_id_list
    highlight_text_id_list = data_highlight_id_list

    print("len highlight id list: ", len(highlight_text_id_list))
    # print("header id list: ", len(header_text_id_list))

    rm_special_token_ids = [tokenizer.convert_tokens_to_ids('<pad>'),
                            tokenizer.convert_tokens_to_ids('__SEP__'),
                            tokenizer.convert_tokens_to_ids('<pad>'),
                            tokenizer.convert_tokens_to_ids('__None__'),
                            tokenizer.convert_tokens_to_ids('__None__'),
                            tokenizer.convert_tokens_to_ids('__EOS__'),
                            tokenizer.convert_tokens_to_ids('__PAD__'),
                            1437]

    # all headers
    header_text_id_set = set(header_text_id_list)



    # print(rm_special_token_ids)
    for special_token in rm_special_token_ids:
        if special_token in header_text_id_set:
            header_text_id_set.remove(special_token)

    # print("len set highlight id list: ", len(header_text_id_set))
    # get header position in src text
    position = []
    for i, src_token_id in enumerate(src_text_id_list):
        if src_token_id in header_text_id_set:
            position.append(i)

    header_cross_attention_list = []
    header_cat_attention_list = []
    header_tokens = []

    # extract all headers' attentions & labels
    for i in position:
        if (src_text_id_list[i]) in highlight_text_id_list:
            data_labels.append(1)
        else:
            data_labels.append(0)

        header_token = tokenizer.convert_ids_to_tokens(src_text_id_list[i])
        header_tokens.append(header_token)

        encoder_attentions = []
        for encoder_attention in ln_encoder_attentions:
            encoder_attention = encoder_attention.squeeze(0)
            header_encoder_attention_i = encoder_attention[:, :, i]
            encoder_attentions.append(header_encoder_attention_i)

        cross_attentions = []
        for cross_attention in ln_cross_attentions:
            cross_attention = cross_attention.squeeze(0)
            header_cross_attention_i = cross_attention[:, :, i]
            cross_attentions.append(header_cross_attention_i)

        cat_encoder_attentions = encoder_attentions[0]
        for encoder_a in encoder_attentions[1:]:
            cat_encoder_attentions = torch.cat([cat_encoder_attentions, encoder_a],
                                               dim=0)

        cat_cross_attentions = cross_attentions[0]
        for cross_a in cross_attentions[1:]:
            cat_cross_attentions = torch.cat([cat_cross_attentions, cross_a], dim=0)

        header_cross_attention_PIL = cvtOneTensorToPIL(cat_cross_attentions)
        header_cross_attention_list.append(header_cross_attention_PIL)



        cat_attention = torch.cat((cat_encoder_attentions, cat_cross_attentions),
                                                            dim=1)

        cat_attention_PIL = cvtOneTensorToPIL(cat_attention)
        header_cat_attention_list.append(cat_attention_PIL)

        # print("cat_encoder_attentions shape ", cat_encoder_attentions.shape)
        # print("cat_cross_attentions shape ", cat_cross_attentions.shape)
        # print("cat_attention shape ", cat_attention.shape)

    data_header_dict['crossAttentions'] = header_cross_attention_list
    data_header_dict['catAttentions'] = header_cat_attention_list
    data_header_dict['labels'] = data_labels
    data_header_dict['tokens'] = header_tokens
    return data_header_dict


def process_sentence_header_data_cellInfo(tokenizer, data_table_id_list, data_header_id_list,
                                 data_highlight_id_list, data_encoder_attentions,
                                 data_cross_attentions):


    print("Process Cell Info!!!!!!")

    data_labels = []
    data_header_dict = {}


    ln_encoder_attentions = []
    for i in data_encoder_attentions:
        i = i.cpu()
        layer_norm_i = nn.LayerNorm([i.shape[1], i.shape[2], i.shape[3]])
        data_encoder_attention = layer_norm_i(i)
        ln_encoder_attentions.append(data_encoder_attention)

    ln_cross_attentions = []
    for i in data_cross_attentions:
        i = i.cpu()
        layer_norm_i = nn.LayerNorm([i.shape[1], i.shape[2], i.shape[3]])
        data_cross_attention = layer_norm_i(i)
        ln_cross_attentions.append(data_cross_attention)

    src_text_id_list = data_table_id_list
    header_text_id_list = data_header_id_list
    highlight_text_id_list = data_highlight_id_list

    # print("src_text_list: ", tokenizer.decode(src_text_id_list))
    # print("len(src_text_list): ", len(tokenizer.decode(src_text_id_list)))
    # print("len(src_id_list): ", len(src_text_id_list))
    # print("src_text_id_list: ", src_text_id_list)
    # print("header_text_list: ", tokenizer.decode(header_text_id_list))
    # print("highlight_text_list: ", tokenizer.decode(highlight_text_id_list,
    #                                                 ))
    # print("highlight_text_id_list", highlight_text_id_list)

    print("check special tokens: ", tokenizer.convert_ids_to_tokens(highlight_text_id_list))

    print("__SEP__: ", tokenizer.convert_tokens_to_ids("__SEP__"))

    # print("len highlight id list: ", len(highlight_text_id_list))
    # print("header id list: ", len(header_text_id_list))

    # rm_special_token_ids = [tokenizer.convert_tokens_to_ids('<pad>'),
    #                         tokenizer.convert_tokens_to_ids('__SEP__'),
    #                         tokenizer.convert_tokens_to_ids('<pad>'),
    #                         tokenizer.convert_tokens_to_ids('__None__'),
    #                         tokenizer.convert_tokens_to_ids('__None__'),
    #                         tokenizer.convert_tokens_to_ids('__EOS__'),
    #                         tokenizer.convert_tokens_to_ids('__PAD__'),
    #                         1437]

    # all headers
    # header_text_id_set = set(header_text_id_list)



    # print(rm_special_token_ids)
    # for special_token in rm_special_token_ids:
    #     if special_token in header_text_id_set:
    #         header_text_id_set.remove(special_token)


    # print("len set highlight id list: ", len(header_text_id_set))
    # get header position in src text
    # position = []
    # for i, src_token_id in enumerate(src_text_id_list):
    #     if src_token_id in header_text_id_set:
    #         position.append(i)
    #
    # print("positions: ", position)
    #
    # eos_position = []
    # eos_id = tokenizer.convert_tokens_to_ids('__EOS__')
    # print(eos_id)
    # for i in range(len(src_text_id_list)):
    #     if src_text_id_list[i] == eos_id:
    #         eos_position.append(i)
    # for i, eos_id in enumerate(src_text_id_list):
    #     if eos_id == src_text_id_list[i]:
    #         eos_position.append(i)

    # print(eos_position)
    #
    # header_cross_attention_list = []
    # header_cat_attention_list = []
    # header_tokens = []
    #
    # start_end_positions = sorted(position+eos_position)
    # print(start_end_positions)
    #
    # print("split header cells")
    # cells_list = []
    # for i in range(len(start_end_positions)):
    #     if i != len(start_end_positions)-1:
    #         if src_text_id_list[start_end_positions[i]] != eos_id:
    #             cell = src_text_id_list[start_end_positions[i]:start_end_positions[i+1]]
    #             cells_list.append(cell)
    #
    #
    #
    #
    # for cell in cells_list:
    #     print(tokenizer.convert_ids_to_tokens(cell))
    #
    #
    # # extract all headers' attentions & labels
    # for i in position:
    #     if (src_text_id_list[i]) in highlight_text_id_list:
    #         data_labels.append(1)
    #     else:
    #         data_labels.append(0)
    #
    #     header_token = tokenizer.convert_ids_to_tokens(src_text_id_list[i])
    #     print(header_token)
    #     header_tokens.append(header_token)

        # encoder_attentions = []
        # for encoder_attention in ln_encoder_attentions:
        #     encoder_attention = encoder_attention.squeeze(0)
        #     header_encoder_attention_i = encoder_attention[:, :, i]
        #     encoder_attentions.append(header_encoder_attention_i)
        #
        # cross_attentions = []
        # for cross_attention in ln_cross_attentions:
        #     cross_attention = cross_attention.squeeze(0)
        #     header_cross_attention_i = cross_attention[:, :, i]
        #     cross_attentions.append(header_cross_attention_i)
        #
        # cat_encoder_attentions = encoder_attentions[0]
        # for encoder_a in encoder_attentions[1:]:
        #     cat_encoder_attentions = torch.cat([cat_encoder_attentions, encoder_a],
        #                                        dim=0)
        #
        # cat_cross_attentions = cross_attentions[0]
        # for cross_a in cross_attentions[1:]:
        #     cat_cross_attentions = torch.cat([cat_cross_attentions, cross_a], dim=0)
        #
        # header_cross_attention_PIL = cvtOneTensorToPIL(cat_cross_attentions)
        # header_cross_attention_list.append(header_cross_attention_PIL)



        # cat_attention = torch.cat((cat_encoder_attentions, cat_cross_attentions),
        #                                                     dim=1)

        # cat_attention_PIL = cvtOneTensorToPIL(cat_attention)
        # header_cat_attention_list.append(cat_attention_PIL)

        # print("cat_encoder_attentions shape ", cat_encoder_attentions.shape)
        # print("cat_cross_attentions shape ", cat_cross_attentions.shape)
        # print("cat_attention shape ", cat_attention.shape)

    # data_header_dict['crossAttentions'] = header_cross_attention_list
    # # data_header_dict['catAttentions'] = header_cat_attention_list
    # data_header_dict['labels'] = data_labels
    # data_header_dict['tokens'] = header_tokens
    # return data_header_dict


def eval_sciGen(text_out_path, reference_path):
    bleu_info = eval_sacre_bleu(reference_path, text_out_path)
    # moverScore = eval_mover_score(reference_path, text_out_path)
    # return bleu_info, moverScore
    return bleu_info
def eval_sacre_bleu(ref_file, pred_file):
    try:
        refs = [get_lines(ref_file)]
        sys = get_lines(pred_file)
        bleu = scb.corpus_bleu(sys, refs)
        return bleu.score
    except:
        return 0

# def eval_mover_score(ref_file, pred_file):
#     try:
#         refs = get_lines(ref_file)
#         sys = get_lines(pred_file)
#         idf_dict_hyp = get_idf_dict(sys)
#         idf_dict_ref = get_idf_dict(refs)
#
#         scores = word_mover_score(refs, sys, idf_dict_ref, idf_dict_hyp, \
#                           stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
#         return round(np.mean(scores),3) , round(np.median(scores),3 )
#     except Exception as e:
#         print(e)
#         return 0, 0


def get_lines(fil):
    lines = []
    with open(fil, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
            else:
                lines.append('empty')
    return lines

