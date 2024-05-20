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
    train_dict, dev_dict = {}, {}

    train_dict['table_text_path'] = args.train_table_text_path
    train_dict['reference_sentence_path'] = args.train_reference_sentence_path

    dev_dict['table_text_path'] = args.dev_table_text_path
    dev_dict['reference_sentence_path'] = args.dev_reference_sentence_path



    special_token_name = args.special_token_path

    print(args.special_token_path)
    data = Data(train_dict, dev_dict, args.max_table_len, args.max_content_plan_len, args.max_tgt_len,
                args.generator_model_name, args.special_token_path, args.min_slot_key_cnt)

    print('Data loaded...')
    print("args.max_decode_len", args.max_decode_len)
    model = Generator(model_name=args.generator_model_name, tokenizer=data.decode_tokenizer,
                      max_decode_len=args.max_decode_len, dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda(device)

    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = (args.total_steps // args.gradient_accumulation_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.total_steps)
    optimizer.zero_grad()

    train_num, dev_num = data.train_num, data.dev_num
    batch_size = args.batch_size
    train_step_num, dev_step_num = int(train_num / batch_size) + 1, int(dev_num / batch_size) + 1

    batches_processed = 0
    max_dev_score = 0.
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every

    train_loss_accumulated = 0.

    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        batches_processed += 1

        # return (batch_table_tensor, batch_table_mask), \
        #        (batch_src_tensor, batch_src_mask), \
        #        batch_tgt_tensor, \
        #        (batch_reference_text_list, batch_table_text_list, batch_src_text_list), \
        #        (batch_table_id_list, batch_src_id_list)

        _, train_batch_src_item, train_batch_tgt_item, _, _ = data.get_next_train_batch(
            batch_size)

        train_batch_src_tensor, train_batch_src_mask = map_cuda(train_batch_src_item, device,
                                                                cuda_available)

        train_batch_tgt_tensor = map_cuda_(train_batch_tgt_item, device, cuda_available)

        train_output = model(train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_tensor)
        train_loss = train_output.loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss_accumulated += train_loss.item()

        if (one_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if batches_processed % print_every == 0:
            curr_train_loss = train_loss_accumulated / print_every
            print('At epoch %d, batch %d, train loss %.5f, max combine score is %5f' %
                  (epoch, batches_processed, curr_train_loss, max_dev_score))
            train_loss_accumulated = 0.

        if batches_processed % eval_every == 0:
            model.eval()
            dev_loss_accumulated = 0.
            dev_output_text_list = []
            print('Start evaluation...')
            with torch.no_grad():
                # return (batch_table_tensor, batch_table_mask), \
                #        (batch_src_tensor, batch_src_mask), \
                #        batch_tgt_tensor, \
                #        (batch_reference_text_list, batch_table_text_list, batch_src_text_list), \
                #        (batch_table_id_list, batch_src_id_list)
                for dev_step in range(dev_step_num):
                    _, dev_batch_src_item, dev_batch_tgt_item, _, _ \
                        = data.get_next_dev_batch(batch_size)
                    dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device,
                                                                        cuda_available)
                    dev_batch_tgt_tensor = map_cuda_(dev_batch_tgt_item, device, cuda_available)

                    dev_output = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_tgt_tensor)
                    dev_loss = dev_output.loss
                    dev_loss_accumulated += dev_loss.item()
                    decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask)
                    dev_output_text_list += decoded_result

                dev_output_text_list = dev_output_text_list[:dev_num]
                dev_text_out_path = \
                    "../outputs/inference_results/"+args.inference_output_file

                print("dev_text_out_path", dev_text_out_path)

                with open(dev_text_out_path, 'w', encoding='utf8') as o:
                    for text in dev_output_text_list:
                        o.writelines(text + '\n')

                # predicted_list = dev_output_text_list[:dev_num]
                # reference_list = []
                # with open(args.dev_reference_path, 'r', encoding='utf8') as f:
                #     for line in f:
                #         reference_list.append(line.strip())

                bleu_info =ctrlsum_eval(dev_text_out_path, args.dev_reference_path)
                # _, _, _,  bleu_info = evaluate_bleu(predicted_list, reference_list)
                # _, _, rouge_info = evaluate_rouge(predicted_list, reference_list)
                # meteor_info = evaluate_meteor(predicted_list, reference_list)
                # _, _, bert_score_info = evaluate_bert_score(predicted_list, reference_list)
                # bleurt_info = evaluate_bleurt(predicted_list, reference_list)
                # #
                # one_dev_combine_score = bleu_info + rouge_info + meteor_info + bert_score_info + bleurt_info
                one_dev_combine_score = bleu_info
                one_dev_loss = dev_loss_accumulated / dev_step_num
                print('----------------------------------------------------------------')
                # print(
                #     'At epoch %d, batch %d, bleu_info is %5f, rouge_info is %5f, meteor_info is '
                #     '%5f,  bert_score_info is %5f, bleurt_info is %5f, one_dev_combine_score is %5f,'
                #     ' dev loss is %5f' \
                #     % (epoch, batches_processed, bleu_info, rouge_info, meteor_info,
                #        bert_score_info, bleurt_info, one_dev_combine_score, one_dev_loss))
                print(
                    'At epoch %d, batch %d, bleu_info is %5f, one_dev_combine_score is %5f, dev loss is %5f' \
                    % (epoch, batches_processed, bleu_info,one_dev_combine_score,
                    one_dev_loss))


                # save model
                save_name = args.generator_model_name.replace("/", "")+ "_sci_inference.ckpt"

                if one_dev_combine_score > max_dev_score:
                    torch.save({'model': model.state_dict()}, ckpt_output_dir + save_name)
                    max_dev_score = one_dev_combine_score
                else:
                    pass

                print("remove old model!!!")
                fileData = {}
                for fname in os.listdir(ckpt_output_dir):
                    fileData[fname] = os.stat(ckpt_output_dir + '/' + fname).st_mtime
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < 1:
                    pass
                else:
                    delete = len(sortedFiles) - 1
                    for x in range(0, delete):
                        os.remove(ckpt_output_dir + '/' + sortedFiles[x][0])
            model.train()
