# -*- coding:utf-8 -*-
"""
Author: Zhixin Guo
Date: 2022/08/30
"""
import torch
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
from utlis import parse_config, map_cuda, map_cuda_
from scidataclass import Data
from evaluate import evaluate_bleu, evaluate_rouge, evaluate_meteor, evaluate_bert_score, \
    evaluate_bleurt, ctrlsum_eval
from generator import Generator
from CTRLEval.ctrleval import CTRLEval

import os


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    args = parse_config()
    device = args.gpu_id

    ckpt_output_dir = args.ckpt_path
    if os.path.exists(ckpt_output_dir):
        pass
    else:  # recursively construct directory
        os.makedirs(ckpt_output_dir, exist_ok=True)

    print('Start loading data...')
    test_dict = {}
    test_dict['table_text_path'] = args.test_table_text_path
    test_dict['reference_sentence_path'] = args.test_reference_sentence_path


    special_token_name = args.special_token_path

    print(args.special_token_path)
    data = Data(train_dict, dev_dict, args.max_table_len, args.max_content_plan_len, args.max_tgt_len,
                args.generator_model_name, args.special_token_path, args.min_slot_key_cnt)

    print('Data loaded...')
    print("args.max_decode_len", args.max_decode_len)
    model = Generator(model_name=args.generator_model_name, tokenizer=data.decode_tokenizer,
                      max_decode_len=args.max_decode_len, dropout=args.dropout)


    model.eval()

    test_num = data.dev_num
    batch_size = args.batch_size
    test_step_num = int(test_num / batch_size) + 1

    model.eval()
    test_output_list = []
    print('Start evaluation...')
    test_src_tensor_list, test_tgt_tensor_list = [],[]
    with torch.no_grad():
        for test_step in range(test_step_num):
            _, test_batch_src_item, test_batch_tgt_item, _, _ \
                = data.get_next_dev_batch(batch_size)
            test_batch_src_tensor, test_batch_src_mask = map_cuda(test_batch_src_item, device,
                                                                cuda_available)
            test_batch_tgt_tensor = map_cuda_(test_batch_tgt_item, device, cuda_available)

            test_src_tensor_list.append(test_batch_src_tensor)
            test_tgt_tensor_list.append(test_batch_tgt_tensor)



            metric = Perplexity(ignore_index=-100)
            metric(preds, target)

            # calculate perplexit

            test_output = model(test_batch_src_tensor, test_batch_src_mask, test_batch_tgt_tensor)
            decoded_result = model.generate(test_batch_src_tensor, test_batch_src_mask)
            test_output_text_list += decoded_result

        test_output_text_list = test_output_text_list[:test_num]
        test_text_out_path = \
            "../outputs/inference_results/" + args.inference_output_file

        print("test_text_out_path", test_text_out_path)

        with open(test_text_out_path, 'w', encoding='utf8') as o:
            for text in dev_output_text_list:
                o.writelines(text + '\n')


    ################################################################################################
    # basic evaluations

        print('Basic Evaluations (BLEU, ROUGE, METEOR, BERTScore, BLEURT)')
        # bleu scores
        bleu_info = ctrlsum_eval(dev_text_out_path, args.dev_reference_path)
        print('BLEU score: %.4f' % bleu_info)

        # rouge scores
        rouge_1, rouge_2, rouge_l = evaluate_rouge(predicted_list, reference_list)
        print('rouge-1 is %5f, rouge-2 is %5f, rouge-l is %5f' % (rouge_1, rouge_2, rouge_l))

        # meteor scores
        meteor_info = evaluate_meteor(predicted_list, reference_list)
        print('METEOR score: %.4f' % meteor_info)

        # bert scores
        bert_score_precision, bert_score_recall, bert_score_f1 = evaluate_bert_score(
            predicted_list, reference_list)
        print('bert_score precision is %5f, bert_score recall is %5f, bert_score f1 is %5f' % (
            bert_score_precision, bert_score_recall, bert_score_f1))

        # bleurt scores
        bleurt_info = evaluate_bleurt(predicted_list, reference_list)
        print('BLEURT score: %.4f' % bleurt_info)

        print('----------------------------------------------------------------')


    ################################################################################################
    # advande evaluations
        print('Advanced Evaluations (perplexity, CTRLEval)')

        # perplexity scores
        perplexity = evaluate_perplexity(test_src_tensor_list, test_tgt_tensor_list)
        print('perplexity score: %.4f' % perplexity)

        # CTRLEval scores
        task = 'topic'  # evaluation for sentiment-controlled text generation
        scorer = CTRLEval(iwf_dir='./CTRLEval/iwf_full.txt',
                          prompt_dir='./CTRLEval//prompt/prompt_{}.txt'.format(task),
                          verbal_dir='./CTRLEval//prompt/verbal_{}.txt'.format(task),
                          model_name_or_path='google/pegasus-large')
        prefix_list = []
        prefix_path = args.prefix_path
        with open(prefix_path, 'r', encoding='utf8') as f:
            for line in f:
                prefix_list.append(line.strip())

        # evaluation of coherence
        coherence_score = scorer.score(aspect='coh', data=predicted_list, batch_size=16)
        # evaluation of consistency
        consistency_score = scorer.score(aspect='cons', data=predicted_list, prefix=prefix_list,
                                 batch_size=16)
        print('CTRLEval coherence_score is %5f, CTRLEval consistency_score is %5f' % (
            coherence_score,consistency_score))



