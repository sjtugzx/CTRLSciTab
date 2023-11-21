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
ignored_words = nltk.corpus.stopwords.words('english')







def generate_output_files(file):
    with open(file, 'r', encoding='utf-8') as f:
        with open('{}.target_new'.format(out), 'w', encoding='utf-8') as target:
            print('{}.target_new'.format(out))

            data = json.load(f)
            print("length of data: ", len(data))

            for d in range(len(data)):
                try:
                    descp = data[d]['text'].replace('[CONTINUE]', '') + '\n'
                    target.write(descp)
                except:
                    print('error')
                    pass

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

    generate_output_files(file)








