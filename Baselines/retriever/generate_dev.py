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


def generate_output_files(file):
    with open(file, 'r', encoding='utf-8') as f:
        with open('tsdae_without_bginfo_highlightcell/{}.source'.format(out), 'w', encoding='utf-8') as source:
            print('tsdae_without_bginfo_highlightcell/{}.source'.format(out))
            with open('tsdae_without_bginfo_highlightcell/{}.target'.format(out), 'w', encoding='utf-8') as target:
                print('tsdae_without_bginfo_highlightcell/{}.target'.format(out))

                data = json.load(f)
                print("length of data: ", len(data))

                sourcelist = []
                targetlist = []
                for d in trange(len(data)):
                    # text = cell_separator.join(data[d]['table_highlight_cell']) + " " + highlight_separator + " "
                    # text += row_seperator + ' ' + cell_separator
                    # row_len = len(data[d]['table_column_names'])
                    # for i, c in enumerate(data[d]['table_column_names']):
                    #     text += ' ' + c
                    #     if i < row_len - 1:
                    #         text += ' ' + cell_separator
                    text = ''
                    row_len = len(data[d]['table_column_names'])
                    for row in data[d]['table_content_values']:
                        text += ' ' + row_seperator + ' ' + cell_separator
                        for i, c in enumerate(row):
                            text += ' ' + c
                            if i < row_len - 1:
                                text += ' ' + cell_separator

                    initial_text = text
                    nlp = spacy.load("en_core_sci_scibert")
                    word = word_tokenize(data[d]['background_information'])
                    filtered_word = [w for w in word if not w in ignored_words]
                    try:

                        background_s = nlp(" ".join(filtered_word))
                        print("len of background sentences: {}".format(len(background_s)))
                        background_sentences = list(background_s.sents)

                        descp = data[d]['text'].replace('[CONTINUE]', '') + '\n'
                        # reference_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                        #     'table_caption'] + ' ' + background_separator + ' ' + descp + '\n'

                        reference_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                            'table_caption']
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

                        # raw format data
                        # filtered_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                        #     'table_caption'] + ' ' + background_separator + ' ' + \
                        #                 filtered_background_sentences.strip()

                        # without background information
                        # filtered_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                        #     'table_caption']

                        # without highlight cell
                        # filtered_text = initial_text + ' ' + caption_separator + ' ' + data[d][
                        #     'table_caption'] + ' ' + background_separator + ' ' + \
                        #                 filtered_background_sentences.strip()

                        # without highlight cell and background information
                        filtered_text = initial_text + ' ' + caption_separator + ' ' + data[d]['table_caption']

                        # rm useless '\n'
                        new_filtered_text = filtered_text.replace("\n", "") + ("\n")
                        new_descp = descp.replace("\n", "") + ("\n")
                        source.write(new_filtered_text)
                        target.write(new_descp)

                        sourcelist.append(new_filtered_text)
                        targetlist.append(new_descp)
                    except:
                        print('error')
                        pass
    return sourcelist, targetlist


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

    args = parser.parse_args()
    file = args.file
    out = args.split

    row_seperator = '<R>'
    cell_separator = '<C>'
    caption_separator = '<CAP>'
    background_separator = '<BKG>'
    highlight_separator = '<H>'

    sources, targets = generate_output_files(file)
    save_to_json(sources, 'tsdae_without_bginfo_highlightcell/{}_source.json'.format(out))
    save_to_json(targets, 'tsdae_without_bginfo_highlightcell/{}_target.json'.format(out))
