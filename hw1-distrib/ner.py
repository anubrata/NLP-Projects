# ner.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from models import *
from collections import Counter
from typing import List


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, HMM, CRF)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class BadNerModel(object):
    """
    NER model that simply assigns each word its most likely observed tag in training

    Attributes:
        words_to_tag_counters: dictionary where each word (string) is mapped to a Counter over tags representing
        counts observed in training
    """
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        pred_tags = []
        for tok in sentence_tokens:
            if tok.word in self.words_to_tag_counters:
                # [0] selects the top most common (tag, count) pair, the next [0] picks out the tag itself
                pred_tags.append(self.words_to_tag_counters[tok.word].most_common(1)[0][0])
            else:
                pred_tags.append("O")
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_bad_ner_model(training_set: List[LabeledSentence]) -> BadNerModel:
    """
    :param training_set: labeled NER sentences to extract a BadNerModel from
    :return: the BadNerModel based on counts collected from the training data
    """
    words_to_tag_counters = {}
    for sentence in training_set:
        tags = sentence.get_bio_tags()
        for idx in range(0, len(sentence)):
            word = sentence.tokens[idx].word
            if not word in words_to_tag_counters:
                words_to_tag_counters[word] = Counter()
            words_to_tag_counters[word][tags[idx]] += 1.0
    return BadNerModel(words_to_tag_counters)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train = read_data(args.train_path)
    dev = read_data(args.dev_path)
    # Here's a few sentences...
    print("Examples of sentences:")
    print(str(dev[1]))
    print(str(dev[3]))
    print(str(dev[5]))
    system_to_run = args.model
    # If set to True, runs your CRF on the test set to produce final output
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_ner_model(train)
        dev_decoded = [bad_model.decode(test_ex.tokens) for test_ex in dev]
    elif system_to_run == "HMM":
        hmm_model = train_hmm_model(train)
        ## TODO: Run for the whole dev set: HMM
        dev_decoded = [hmm_model.decode(test_ex.tokens) for test_ex in dev]
        # dev_decoded = [hmm_model.decode(dev[-1].tokens)]
    elif system_to_run == "CRF":
        ## TODO: When done debugging comment out the following line to train on the whole dataset
        # train = read_data("./data/eng.train.small")
        crf_model = train_crf_model(train)
        print("Data reading and training took %f seconds" % (time.time() - start_time))
        ## TODO: Comment out to get the dev scores: CRF
        # dev = read_data("./data/eng.train.small")
        dev_decoded = [crf_model.decode(test_ex.tokens) for test_ex in tqdm(dev)]
        print_evaluation(dev, dev_decoded)
        if args.run_on_test:
            print("Running on test")
            test = read_data(args.blind_test_path)
            test_decoded = [crf_model.decode(test_ex.tokens) for test_ex in test]
            print_output(test_decoded, args.test_output_path)
    else:
        raise Exception("Pass in either BAD, HMM, or CRF to run the appropriate system")
    # Print the evaluation statistics
    print_evaluation(dev, dev_decoded)
