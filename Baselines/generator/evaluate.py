from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from torchmetrics.text.bert import BERTScore
from bleurt import score
import numpy as np
from multiprocessing import pool
import subprocess
import multiprocessing
from itertools import chain



def evaluate_one_bleu(predicted_sentence, reference_sentence):
    bleu_1 = sentence_bleu(reference_sentence, predicted_sentence, weights=(1,0,0,0))
    bleu_2 = sentence_bleu(reference_sentence, predicted_sentence, weights=(0.5,0.5,0,0))
    bleu_3 = sentence_bleu(reference_sentence, predicted_sentence, weights=(0.33,0.33,0.33,0))
    bleu_4 = sentence_bleu(reference_sentence, predicted_sentence, weights=(0.25,0.25,0.25,0.25))

    return bleu_1, bleu_2, bleu_3, bleu_4


def evaluate_bleu(predicted_sentence_list, reference_sentence_list):
    pool = multiprocessing.Pool(processes=30)
    zip_args = list(zip(predicted_sentence_list, reference_sentence_list))
    result = pool.starmap(evaluate_one_bleu, zip_args)
    pool.close()
    pool.join()
    new_result = np.array(result)
    bleu_1_result = new_result[:, 0]
    bleu_2_result = new_result[:, 1]
    bleu_3_result = new_result[:, 2]
    bleu_4_result = new_result[:, 3]

    return np.mean(bleu_1_result), np.mean(bleu_2_result), np.mean(bleu_3_result), np.mean(
        bleu_4_result)


# def evaluate_one_rouge(predicted_sentence, reference_sentence):
#     rouge = Rouge()
#     scores = rouge.get_scores(predicted_sentence, reference_sentence)
#     rouge_1 = scores[0]['rouge-1']['f']
#     rouge_2 = scores[0]['rouge-2']['f']
#     rouge_l = scores[0]['rouge-l']['f']
#     return rouge_1, rouge_2, rouge_l
#
# def evaluate_rouge(predicted_sentence_list, reference_sentence_list):
#     pool = multiprocessing.Pool(processes=30)
#     zip_args = list(zip(predicted_sentence_list, reference_sentence_list))
#     result = pool.starmap(evaluate_one_rouge, zip_args)
#     pool.close()
#     pool.join()
#     new_result = np.array(result)
#     rouge_1_result = new_result[:, 0]
#     rouge_2_result = new_result[:, 1]
#     rouge_l_result = new_result[:, 2]
#     return np.mean(rouge_1_result), np.mean(rouge_2_result), np.mean(rouge_l_result)

def evaluate_one_meteor(predicted_sentence, reference_sentence):
    score = meteor_score([word_tokenize(predicted_sentence)], word_tokenize(reference_sentence))
    return score

def evaluate_meteor(predicted_sentence_list, reference_sentence_list):
    pool = multiprocessing.Pool(processes=30)
    zip_args = list(zip(predicted_sentence_list, reference_sentence_list))
    result = pool.starmap(evaluate_one_meteor, zip_args)
    pool.close()
    pool.join()
    new_result = np.array(result)

    return np.mean(new_result)


def evaluate_bert_score(predicted_list, reference_list):
    print("calculate bert score")
    bertscore = BERTScore()
    preds = ["hello there", "general kenobi"]
    target = ["hello there", "master kenobi"]
    score = bertscore(preds, target)
    precision = score['precision']
    recall = score['recall']
    f1 = score['f1']

    return np.mean(precision), np.mean(recall), np.mean(f1)

def evaluate_bleurt(checkpoint_path, predicted_list, reference_list):
    checkpoint = checkpoint_path
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=reference_list, candidates=predicted_list)
    return np.mean(scores)


def ctrlsum_eval(prediction_path, target_path):
    command = "cat " + prediction_path+" | sacrebleu "+ target_path
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
    print(content_list[0])

    bleu = float(content_list[0].split()[2])

    return(bleu)

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def evaluate_ngram_distinct_n(predicted_list, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param reference_corpus: a list of sentence.
    :param translation_corpus: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return distinct_n_corpus_level(predicted_list, n)


