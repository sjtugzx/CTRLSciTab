import json

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

def write_file(text_list, out_f):
    with open(out_f, 'w', encoding = 'utf8') as o:
        for text in text_list:
            o.writelines(text + '\n')

def load_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    reference_list = []
    table_caption_list = []
    table_column_list = []
    table_content_list = []
    backgroud_list = []

    for value in data.values():
        for keys in value.keys():

            reference = value['text']
            table_caption = value['table_caption']
            table_column_names = value['table_column_names']
            table_content_values = value['table_content_values']
            backgroundinformation = value['backgroundinformation']

            reference_list.append(reference)
            table_caption_list.append(table_caption)
            table_column_list.append(table_column_names)
            table_content_list.append(table_content_values)
            backgroud_list.append(backgroundinformation)

    print(len(reference_list))



if __name__ == '__main__':
    input_file = '../sciGen_data/dev/dev.json'
    extract_data(input_file)