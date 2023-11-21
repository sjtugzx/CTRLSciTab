from argparse import ArgumentParser
import json
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import scispacy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize
from tqdm import *
from multiprocessing import Pool
import math

ignored_words = nltk.corpus.stopwords.words('english')


def calculate_sentence_embeddings(candidate_sentences):
    print(len(candidate_sentences[0]))
    model = SentenceTransformer('output/tsdae_new-model')
    embeddings = model.encode(candidate_sentences)
    print(len(embeddings[0]))

    result_list = []
    print(util.cos_sim(embeddings[0], embeddings[1]))
    print(util.cos_sim(embeddings[0], embeddings[2]))
    print(util.cos_sim(embeddings[0], embeddings[3]))

    for pos in range(len(candidate_sentences) - 1):
        result = util.cos_sim(embeddings[0], embeddings[pos + 1])
        result_list.append(result.item())

    return result_list


def select_top_k_sentences(number, embedding_results):
    print('Selecting top {} sentences'.format(number))
    print("embedding results: ", embedding_results)
    if number < len(embedding_results):
        top_k = sorted(embedding_results, reverse=True)[:number]
        pos_list = []
        for tp in top_k:
            pos_list.append(embedding_results.index(tp))
    else:
        pos_list = [i for i in range(len(embedding_results))]
    return pos_list


def calculate_sentence_tf_idf(candidate_sentences):
    """
       vectorizer: TfIdfVectorizer model
       docs_tfidf: tfidf vectors for all docs
       query: query doc
       return: cosine similarity between query and all docs
       """

    query = candidate_sentences[0]
    allDocs = candidate_sentences[1:]
    vectorizer = TfidfVectorizer(stop_words='english')
    docs_tfidf = vectorizer.fit_transform(allDocs)
    query_tfidf = vectorizer.transform([query])
    tf_idf_results = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return tf_idf_results.tolist()


def select_top_k_sentences_tf_idf(number, tf_idf_results):
    print('Selecting top {} sentences'.format(number))
    print("tf idf results: ", tf_idf_results)
    if number < len(tf_idf_results):
        top_k = sorted(tf_idf_results, reverse=True)[:number]
        pos_list = []
        for tp in top_k:
            pos_list.append(tf_idf_results.index(tp))
    else:
        pos_list = [i for i in range(len(tf_idf_results))]
    return pos_list


def generate_output_files(data):
    # with open('{}.source'.format(out), 'w', encoding='utf-8') as source:
    #     print('{}.source'.format(out))
    #     with open('{}.target'.format(out), 'w', encoding='utf-8') as target:
    #         print('{}.target'.format(out))

    print("length of data: ", len(data))
    sourcelist = []
    targetlist = []
    for d in trange(len(data), desc="generate sentences:"):
        text = cell_separator.join(data[d]['table_highlight_cell']) + " " + highlight_separator + " "
        text += row_seperator + ' ' + cell_separator
        row_len = len(data[d]['table_column_names'])
        for i, c in enumerate(data[d]['table_column_names']):
            text += ' ' + c
            if i < row_len - 1:
                text += ' ' + cell_separator
        # for row in data[d]['table_content_values']:
        #     text += ' ' + row_seperator + ' ' + cell_separator
        #     for i, c in enumerate(row):
        #         text += ' ' + c
        #         if i < row_len - 1:
        #             text += ' ' + cell_separator

        initial_text = text
        nlp = spacy.load("en_core_sci_scibert")
        word = word_tokenize(data[d]['background_information'])
        filtered_word = [w for w in word if not w in ignored_words]
        try:
            background_s = nlp(" ".join(filtered_word))
            print("len of background sentences: {}".format(len(background_s)))
            background_sentences = list(background_s.sents)
            descp = data[d]['text'].replace('[CONTINUE]', '') + '\n'
            reference_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                'table_caption'] + ' ' + background_separator + ' ' + descp + '\n'

            candidate_sentences = [reference_text]
            for sentence in background_sentences:
                candidate_sentence = initial_text + ' ' + caption_separator + ' ' + \
                                     data[d]['table_caption'] + ' ' + \
                                     background_separator + ' ' + str(sentence) + '\n'

                candidate_sentences.append(candidate_sentence)
            print('calculating sentence embeddings')
            # calculate embeddings
            embedding_results = calculate_sentence_embeddings(candidate_sentences)
            pos_list = select_top_k_sentences(3, embedding_results)

            filtered_background_sentences = ""
            for pos in pos_list:
                filtered_background_sentences += str(background_sentences[
                                                         pos]) + " "

            print('writing to file')
            filtered_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                'table_caption'] + ' ' + background_separator + ' ' + \
                            filtered_background_sentences.strip()
            filtered_text = filtered_text.replace('\n', '').replace('\r', '')
            filtered_text = filtered_text + '\n'
            sourcelist.append(filtered_text)
            descp = descp.replace('\n', '').replace('\n', '')
            targetlist.append(descp)
            # source.write(filtered_text)
            # target.write(descp)
        except:
            print('error')
            pass
        # if sourcelist != []:
        #     break
    return sourcelist, targetlist


def list_slice(alist, size, index):
    size = math.ceil(len(alist) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(alist) else len(alist)
    return alist[start:end]


def save_to_json(paper_dict, json_file):
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(paper_dict, ensure_ascii=False, indent=4))
        f.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="The SciGen json file to be converted for pretrained models' input format",
                        required=True)
    parser.add_argument("-s", "--split",
                        help="Specify the corresponding split, i.e., train, dev, or test",
                        required=True)
    parser.add_argument("-i", "--index",
                        help="Specify the corresponding split, i.e., train, dev, or test",
                        required=True)

    args = parser.parse_args()
    file = args.file
    out = args.split
    index = int(args.index)

    row_seperator = '<R>'
    cell_separator = '<C>'
    caption_separator = '<CAP>'
    background_separator = '<BKG>'
    highlight_separator = '<H>'

    with open(file, 'r', encoding='utf-8') as f:
        ALLdata = json.load(f)
    nums = 10
    print(str(index), ' processor started!')
    dataslice = list_slice(ALLdata, nums, index)
    sourcelist, targetlist = generate_output_files(dataslice)
    save_to_json(sourcelist, 'tempdata/source/trainsource_0.json')
    save_to_json(targetlist, 'tempdata/target/traintarget_0.json')
    print('source length: ', len(sourcelist), ' target length: ', len(targetlist))
    with open('tempdata/source/{}_{}.source'.format(out, index), 'w', encoding='utf-8') as source:
        print('tempdata/source/{}_{}.source'.format(out, index))
        with open('tempdata/target/{}_{}.target'.format(out, index), 'w', encoding='utf-8') as target:
            print('tempdata/target/{}_{}.target'.format(out, index))
            for j in range(len(sourcelist)):
                source.write(sourcelist[j])
                target.write(targetlist[j])
