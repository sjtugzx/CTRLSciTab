import sys
import json
import progressbar
import numpy as np
from nltk import word_tokenize
import nltk
import re
from difflib import SequenceMatcher
from nltk.corpus import stopwords
import argparse
stopword_set = set(stopwords.words('english'))

def load_special_tokens(special_token_path, min_cnt):
    special_token_list, special_token_dict = [], {}
    with open(special_token_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            token = content_list[0]
            cnt = int(content_list[1])
            if cnt >= min_cnt:
                special_token_list.append(token)
                special_token_dict[token] = 1
    print (len(special_token_list))
    return special_token_list, special_token_dict

def get_input_text(ordered_cell_list):
    input_text = ''
    for item in ordered_cell_list:
        one_text = item['slot_key'] + ' : ' + item['slot_value'] + ' ' + END_OF_SLOT
        input_text += one_text + ' '
    input_text = ' '.join(input_text.split()).strip()
    return input_text

def write_file(text_list, out_f):
    with open(out_f, 'w', encoding = 'utf8') as o:
        for text in text_list:
            o.writelines(text + '\n')

def clean_one_token(token):
    char_list = []
    for c in token:
        if c == '(':
            one_c = ''
        elif c == ')':
            one_c = ''
        else:
            one_c = c
        char_list.append(one_c)
    return ''.join(char_list)


def clean_text(text):
    res_list = []
    token_list = text.strip().split()
    for token in token_list:
        res_list.append(clean_one_token(token))
    return ' '.join(res_list).strip()

# def map_to_pure_content(content_sequence):
#     res_list = []
#     print(content_sequence)
#     for token in content_sequence.split():
#         if token.startswith('__') and token.endswith('__'):
#             res_list.append(token)
#         else:
#             pass
#     print(res_list)
#     return ' '.join(res_list).strip()

def transform_matching_string(match_string):
    special_char_set = set(list(r"!@#$%^&*()[]{};:,./<>?\|`~-=_+"))
    res_str = ''
    for one_char in match_string:
        if one_char in special_char_set:
            one_char = r"\\" + one_char
        else:
            one_char = one_char
        res_str += one_char
    return res_str

def return_valid_length(substring):
    '''
        return number of valid tokens exist in the matched string
        if the matched string only contains stopword or the overlapped
        length is too small, then we reject the replacement
    '''
    token_list = substring.strip().split()
    valid_len = 0
    for token in token_list:
        if token.lower() in stopword_set:
            pass
        else:
            valid_len += 1
    #if len(substring) < 3: # single letter matching
    if len(substring) < 3 and substring.isalpha():
        valid_len = 0
    elif len(substring) < 2:
        valid_len = 0
    else:
        pass
    return valid_len

def find_longest_common_substring(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    match_string = string1[match.a: match.a + match.size].strip()
    valid_len = return_valid_length(match_string)
    match_span = match.size
    return match_string, valid_len, match_span

    # case 3816 or 3846

def find_final_substring(tokenized_reference, slot_value_text):
    match_1, valid_len_1, span_1 = find_longest_common_substring(tokenized_reference, slot_value_text)
    match_2, valid_len_2, span_2 = find_longest_common_substring(slot_value_text, tokenized_reference)
    if valid_len_1 > valid_len_2:
        return match_1, valid_len_1, span_1
    else:
        return match_2, valid_len_2, span_2

def check_result(text):
    flag = True
    token_list = text.strip().split()
    for token in token_list:
        if token.startswith('__') and token.endswith('__'):
            try:
                assert len(token.strip('__').split('__')) == 2
            except:
                flag = False
                break
    return flag

def process_reference(tokenized_reference, cell_content):
    match_string, valid_len, _ = find_final_substring(tokenized_reference, cell_content)
    match_span = len(match_string)

    # print (match_string, valid_len, match_span)

    if match_span >= len(cell_content) * 0.4 and cell_content not in stopword_set:
        return True
    else:
        return False

def get_highlight_cells(tokenized_reference, table_content):
    res_text = clean_text(tokenized_reference)
    # table_content = clean_text(table_content)
    print(table_content)
    one_highlight_cell_list = []
    one_highlight_cell_pos_list = []

    for row_idx in range(len(table_content)):
        for col_idx in range(len(table_content[row_idx])):
            cell_content = table_content[row_idx][col_idx]
            cell_content = clean_text(cell_content)
            process_result = process_reference(res_text, cell_content)
            if process_result:
                one_highlight_cell_list.append(cell_content)
                one_highlight_cell_pos_list.append([row_idx, col_idx])

    return one_highlight_cell_list, one_highlight_cell_pos_list

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--special_token_min_cnt', type=int)
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--file_head_name', type=str)
    parser.add_argument('--dataset_mode', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    special_token_path = args.special_token_path
    special_token_min_cnt = args.special_token_min_cnt
    special_token_list, special_token_dict = load_special_tokens(special_token_path, special_token_min_cnt)

    print ('Start loading raw data...')
    json_dict_list = []

    with open(args.raw_data_path, 'r', encoding = 'utf8') as f:
        data = json.load(f)
        for one_json_dict in data.values():
            json_dict_list.append(one_json_dict)

    print('Raw data loaded.')


    dataset_mode = args.dataset_mode
    print ('Processing data...')
    if dataset_mode == 'train' or dataset_mode == 'dev':
        all_src_text_list, all_reference_list = [], []
        all_content_text_list = []
        p = progressbar.ProgressBar(len(json_dict_list))
        p.start()
        idx = 0
        #for one_json_dict in [json_dict_list[710]]:
        for one_json_dict in json_dict_list:
            p.update(idx + 1)
            idx += 1
            one_reference = one_json_dict['text']
            one_table_caption = one_json_dict['table_caption']

            one_table_column_names = one_json_dict['table_column_names']
            one_table_content_values = one_json_dict['table_content_values']
            one_background_information = one_json_dict['backgroundinformation']

            print('one_reference: ', one_reference)
            # cat_one_table = [one_table_column_names]
            # for i in one_table_content_values:
            #     cat_one_table.append(i)

            one_highlight_cells = []
            one_highlight_cells_pos = []

            # one_highlight_cells, one_highlight_cells_pos = get_highlight_cells(
            #     one_reference, one_table_content_values)
            # print('one_reference: ', one_reference)
            one_highlight_cell, one_highlight_cell_pos = get_highlight_cells(one_reference,
                                                             one_table_content_values)
            print('one_highlight_cell: ', one_highlight_cell)
            print('one_highlight_cell_pos: ', one_highlight_cell_pos)
            one_highlight_cells.append(one_highlight_cell)
            one_highlight_cells_pos.append(one_highlight_cell_pos)

            if idx == 10:
                break
            #if idx == 10:
            #    break
            # try:
            #     one_reference = value['text']
            #     one_table_caption = value['table_caption']
            #     one_table_column_names = value['table_column_names']
            #     one_table_content_values = value['table_content_values']
            #     one_background_information = value['backgroundinformation']
            #
            #     one_highlight_cells = []
            #     one_highlight_cells_pos = []
            #
            #     # one_highlight_cells, one_highlight_cells_pos = get_highlight_cells(
            #     #     one_reference, one_table_content_values)
            #     print("get highlight cells")
            #     get_highlight_cells(one_reference, one_table_content_values)
            #     break

                # one_content_text = process_one_instance(one_tokenized_reference, one_ordered_cell_list)
                # one_map_dict = map_content_to_order_dict(one_content_text)
                # one_input_text = get_input_text(one_ordered_cell_list)
                # all_src_text_list.append(one_input_text)
                # all_reference_list.append(one_original_reference)
                # one_content_text = restore_original_content_text(one_content_text)
                # all_content_text_list.append(one_content_text)
                # all_highlight_header_list.append(highlight_headers)


            # except:
            #     print ("exception, something wrong with this instance")
            #     break
                # pass
        p.finish()

        # head_name = args.file_head_name
        # table_file_name = head_name + '_table.txt'
        # write_file(all_src_text_list, table_file_name)
        #
        # content_plan_file = head_name + '_content_plan.txt'
        # write_file(all_content_text_list, content_plan_file)
        #
        # reference_file = head_name + '_reference.txt'
        # write_file(all_reference_list, reference_file)
        #
        # highlight_header_file = heade_name + '_highlight_headers.txt'
        # write_file(all_highlight_header_list, highlight_headers_file)

    elif dataset_mode == 'test':
        all_src_text_list = []
        all_highlight_header_list = []
        p = progressbar.ProgressBar(len(json_dict_list))
        p.start()
        idx = 0
        #for one_json_dict in [json_dict_list[710]]:
        for one_json_dict in json_dict_list:
            p.update(idx + 1)
            idx += 1
            #if idx == 10:
            #    break
            try:
                one_ordered_cell_list = parse_subtable_metastr(one_json_dict, special_token_dict)
                one_input_text = get_input_text(one_ordered_cell_list)
                one_table = one_json_dict['table']
                one_highlight_cells_pos = one_json_dict['highlighted_cells']
                # print (one_highlight_cells_pos)
                highlight_map = []
                highlight_tokens = []
                highlight_headers = []
                for pos in one_highlight_cells_pos:
                    row_pos = pos[0]
                    col_pos = pos[1]
                    hc = one_table[row_pos][col_pos]['value']
                    highlight_tokens.append(hc)

                # print(highlight_tokens)

                for hc in highlight_tokens:
                    for item in one_ordered_cell_list:
                        if item['slot_value'] == hc:
                            highlight_map.append(item)
                # print(highlight_map)
                for item in highlight_map:
                    highlight_headers.append(item['slot_key'])

                all_src_text_list.append(one_input_text)
                all_highlight_header_list.append(highlight_headers)

            except:
                print (idx-1)
                pass
        p.finish()
        head_name = args.file_head_name
        table_file_name = head_name + '_table.txt'
        highlight_headers_name = head_name + '_highlight_headers.txt'
        write_file(all_src_text_list, table_file_name)
        write_file(all_highlight_header_list, highlight_headers_file)
    else:
        raise Exception('Wrong Dataset Mode!!!')
    print ('Data processed.')
