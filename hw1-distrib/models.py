# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np
from scipy.misc import logsumexp


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def get_max_prev(self, current_state_idx, time_step, viterbi):
        prev_value_list = np.zeros(len(self.tag_indexer))

        for prev_state_idx in range(0, len(self.tag_indexer)):
            prev_value_list[prev_state_idx] = self.transition_log_probs[prev_state_idx, current_state_idx] + viterbi[time_step-1, prev_state_idx]

        max_prev = np.max(prev_value_list)
        max_from_prev = np.argmax(prev_value_list)

        return max_prev, max_from_prev


    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        viterbi = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        back_pointers = np.zeros((len(sentence_tokens), len(self.tag_indexer)))

        # print("Current Sentence: ", sentence_tokens)
        # Initial State
        word_index = self.word_indexer.index_of(sentence_tokens[0].word)
        if (word_index == -1):
            word_index = self.word_indexer.index_of("UNK")

        for curr_state_idx in range(0, len(self.tag_indexer)):
            # print("Current Word - Current State:", sentence_tokens[0].word, self.tag_indexer.get_object(curr_state_idx))
            viterbi[0, curr_state_idx] = self.init_log_probs[curr_state_idx] + self.emission_log_probs[curr_state_idx, word_index]

        # Viterbi lattice
        for time_step in range(1, len(sentence_tokens)):
            word_index = self.word_indexer.index_of(sentence_tokens[time_step].word)
            if(word_index == -1):
                word_index = self.word_indexer.index_of("UNK")
            for curr_state_idx in range(0, len(self.tag_indexer)):
                max_prev, arg_bck_ptr = self.get_max_prev(curr_state_idx, time_step, viterbi)
                viterbi[time_step, curr_state_idx] = self.emission_log_probs[curr_state_idx, word_index] + max_prev
                back_pointers[time_step, curr_state_idx] = arg_bck_ptr

        # print(viterbi)
        # print(back_pointers)

        # Best Final State
        best_final_state = np.argmax(viterbi[len(sentence_tokens)-1,])

        predicted_tags = []
        predicted_tags.append(self.tag_indexer.get_object(best_final_state))

        for rows in range(len(sentence_tokens) - 1, 0, -1):
            best_final_state = back_pointers[rows, int(best_final_state)]
            predicted_tags.append(self.tag_indexer.get_object(best_final_state))

        predicted_tags.reverse()
        # print(predicted_tags)

        predicted_chunks = chunks_from_bio_tag_seq(predicted_tags)
        return LabeledSentence(sentence_tokens, predicted_chunks)

def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)

## TODO: Implement the Feature Based Scorer
class FeatureBasedSequenceScorer(object):
    """
    Scoring function for sequence models based on features.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        trainsition_potentials: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_potentials: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, transition_features: np.ndarray, emission_potentials: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.trainsition_potentials = transition_potentials
        self.emission_potentials = emission_potentials

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.trainsition_potentials[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_potentials[tag_idx, word_idx]


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_matrix = np.zeros((len(self.tag_indexer), len(self.tag_indexer)))
        for tag_idx_r in range(0, len(self.tag_indexer)):
            for tag_idx_c in range(0, len(self.tag_indexer)):
                prev_state = self.tag_indexer.get_object(tag_idx_r)
                curr_state = self.tag_indexer.get_object(tag_idx_c)
                if curr_state.split("-")[0] == "I":
                    if prev_state.split("-")[-1] != curr_state.split("-")[-1]:
                        self.transition_matrix[tag_idx_r][tag_idx_c] = -np.Inf

    def get_max_prev(self, current_state_idx, time_step, viterbi):
        prev_value_list = np.ma.masked_array(np.zeros(len(self.tag_indexer)), mask = self.transition_matrix[ : ,current_state_idx].T)
        # prev_value_list = Counter()

        for prev_state_idx in range(0, len(self.tag_indexer)):
            if self.transition_matrix[prev_state_idx][current_state_idx] != -np.Inf:
                prev_value_list[prev_state_idx] = viterbi[time_step-1, prev_state_idx]
            # else:
            #     print("Invalid state, ignored", self.tag_indexer.get_object(prev_state_idx), self.tag_indexer.get_object(current_state_idx))

        max_prev = np.max(prev_value_list)
        max_from_prev = np.argmax(prev_value_list)
        # print("predicted transition", self.tag_indexer.get_object(max_from_prev),
        #       self.tag_indexer.get_object(current_state_idx))

        return max_prev, max_from_prev

    def decode(self, sentence_tokens):
        viterbi = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        back_pointers = np.zeros((len(sentence_tokens), len(self.tag_indexer)))

        # print("Current Sentence: ", sentence_tokens)

        # initial state
        for curr_state_idx in range(0, len(self.tag_indexer)):
            # print("Current Word - Current State:", sentence_tokens[0].word, self.tag_indexer.get_object(curr_state_idx))
            tag = self.tag_indexer.get_object(curr_state_idx)
            feat = extract_emission_features(sentence_tokens, 0, tag,self.feature_indexer, False)
            viterbi[0, curr_state_idx] = sum(self.feature_weights[feat])

        # Viterbi lattice
        for time_step in range(1, len(sentence_tokens)):
            for curr_state_idx in range(0, len(self.tag_indexer)):
                max_prev, arg_bck_ptr = self.get_max_prev(curr_state_idx, time_step, viterbi)
                tag = self.tag_indexer.get_object(curr_state_idx)
                feat = extract_emission_features(sentence_tokens, time_step, tag, self.feature_indexer, False)
                viterbi[time_step, curr_state_idx] = sum(self.feature_weights[feat]) + max_prev
                back_pointers[time_step, curr_state_idx] = arg_bck_ptr

        # print(viterbi)
        # print(back_pointers)

        # Best Final State
        best_final_state = np.argmax(viterbi[len(sentence_tokens) - 1,])

        predicted_tags = []
        predicted_tags.append(self.tag_indexer.get_object(best_final_state))

        for rows in range(len(sentence_tokens) - 1, 0, -1):
            best_final_state = back_pointers[rows, int(best_final_state)]
            predicted_tags.append(self.tag_indexer.get_object(best_final_state))

        predicted_tags.reverse()
        print(predicted_tags)

        predicted_chunks = chunks_from_bio_tag_seq(predicted_tags)
        return LabeledSentence(sentence_tokens, predicted_chunks)

        # raise Exception("IMPLEMENT ME")



def phi_e(tag_idx, word_idx, sentence_idx, w, feature_cache):
    phi_value = 0
    feat = feature_cache[sentence_idx][word_idx][tag_idx]
    for i in feat:
        phi_value += w[i]
    return phi_value

## TODO: Implement Forward backward
def forward_backward_crf(sentence_tokens, sentence_idx, tag_indexer, feature_cache, weights, transition_matrix):

    ## Forward pass
    alpha = np.zeros((len(sentence_tokens), len(tag_indexer)))

    for word_idx in range(0, len(sentence_tokens)):
        for tag_idx in range(0, len(tag_indexer)):
            if word_idx == 0:
                ## Real space
                # alpha[0][tag_idx] = np.exp(phi_e(tag_idx, 0, sentence_idx, weights, feature_cache))
                ## log space
                alpha[0][tag_idx] = phi_e(tag_idx, 0, sentence_idx, weights, feature_cache)
            else:
                prev_alpha = []
                for prev_tag_idx in range(0, len(tag_indexer)):
                    ## Real Space
                    ## TODO: Real_space vs log_space alpha calculation
                    # alpha[word_idx][tag_idx] += alpha[word_idx - 1][prev_tag_idx]* np.exp(phi_e(tag_idx, word_idx, sentence_idx, weights, feature_cache)) * transition_matrix[prev_tag_idx][tag_idx]
                    ## TODO: Handling transition at inference for alpha
                    if transition_matrix[prev_tag_idx][tag_idx] != 0:
                        prev_alpha.append(alpha[word_idx-1][prev_tag_idx] + phi_e(tag_idx, word_idx, sentence_idx, weights, feature_cache))
                # alpha[word_idx][tag_idx] = sum(prev_alpha) + phi_e(tag_idx, word_idx, sentence_idx, weights, feature_cache)
                alpha[word_idx][tag_idx] = logsumexp(prev_alpha)

    # Backward pass
    beta = np.zeros((len(sentence_tokens), len(tag_indexer)))
    for word_idx in range(len(sentence_tokens)-1, -1, -1):
        for tag_idx in range(0, len(tag_indexer)):
            if word_idx == len(sentence_tokens)-1:
                ## Real space
                # beta[word_idx][tag_idx] = 1
                ## log space
                beta[word_idx][tag_idx] = np.log(1)
            else:
                beta_next = []
                for next_tag_idx in range(0,len(tag_indexer)):
                    ## real space
                    # beta[word_idx][tag_idx] += beta[word_idx+1][next_tag_idx] \
                    #                            * np.exp(phi_e(next_tag_idx, word_idx+1, sentence_idx, weights, feature_cache)) \
                    #                            * transition_matrix[tag_idx][next_tag_idx]

                    ## log space
                    ## TODO: Handling transition at inference for beta
                    if transition_matrix[tag_idx, next_tag_idx] != 0:
                        beta_next.append(beta[word_idx + 1][next_tag_idx] + phi_e(next_tag_idx, word_idx + 1, sentence_idx, weights, feature_cache))
                beta[word_idx][tag_idx] = logsumexp(beta_next)
                # beta[word_idx][tag_idx] = (beta_next)

    return alpha, beta

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)

    # Transition validity
    print("Getting valid transitions")
    transitional_feature_indexer = Indexer()
    transition_matrix = np.ones((len(tag_indexer), len(tag_indexer)))
    for tag_idx_r in range(0,len(tag_indexer)):
        for tag_idx_c in range (0, len(tag_indexer)):
            prev_state = tag_indexer.get_object(tag_idx_r)
            curr_state = tag_indexer.get_object(tag_idx_c)
            if curr_state.split("-")[0] == "I":
                if prev_state.split("-")[-1] == curr_state.split("-")[-1]:
                    # transition_matrix[tag_idx_r][tag_idx_c] = 0
                    transitional_feature_indexer.add_and_get_index(prev_state+":to:"+curr_state)
                else:
                    transition_matrix[tag_idx_r][tag_idx_c] = 0
            else:
                transitional_feature_indexer.add_and_get_index(prev_state + ":to:" + curr_state)
    print(transition_matrix)

    # Constant transitional features
    f_t = np.ones(len(transitional_feature_indexer))

    epochs = 5

    print("Training")
    feature_size = len(feature_indexer)
    # feature_size = len(feature_indexer) + len(transitional_feature_indexer)
    optimizer = UnregularizedAdagradTrainer(np.zeros(feature_size), eta=1.0)
    # optimizer = L1RegularizedAdagradTrainer(np.zeros(feature_size), lamb=1e-8, eta=1.0)

    for epoch in range(0, epochs):
        for sentence_idx in range(0, len(sentences)):
            sentence_tokens = sentences[sentence_idx].tokens
            bio_tags = sentences[sentence_idx].get_bio_tags()
            weights = optimizer.get_final_weights()
            alpha, beta = forward_backward_crf(sentence_tokens, sentence_idx, tag_indexer, feature_cache, weights, transition_matrix)
            # print(alpha)
            # print(beta)

            # Compute denominator for the posterior and check for stability
            # Since all values are same for the marginals, the z value for the first word
            # should do.
            ## TODO: Sum or logsum?
            z = logsumexp(alpha[0] + beta[0])
            # print(z)
            # if z == 0:
            #     print("divided by 0 error")
            #     print(sentence_tokens)
            #     z = logsum(alpha[0] + beta[0])
            #     print(z)
            ## Commented out code for stability check
            # for word_idx in range(0, len(sentence_tokens)):
            #     z_test = []
            #     z = logsumexp(alpha[word_idx] + beta[word_idx])
            #     z_test.append(z)
            #     print(z)
            # if len(set(z_test)) != 1:
            #     print("Unstable marginal!")

            gold_counter = Counter()
            marginal_counter = Counter()
            for word_idx in range(0, len(sentence_tokens)):
                for feat in feature_cache[sentence_idx][word_idx][tag_indexer.index_of(bio_tags[word_idx])]:
                    gold_counter[feat] += 1
                for tag_idx in range(0, len(tag_indexer)):
                    # print("Computing marginal for state {0}".format(tag_indexer.get_object(tag_idx)))
                    ## TODO: Sum or logsum?
                    # numerator = logsumexp(alpha[word_idx][tag_idx], beta[word_idx][tag_idx])
                    numerator = alpha[word_idx][tag_idx] + beta[word_idx][tag_idx]
                    marginal_value = numerator - z
                    # Convert the marginals back to real_value
                    # print("log - marginal", marginal_value)
                    marginal_value = np.exp(marginal_value)
                    # print("marginals", marginal_value)
                    for feat in feature_cache[sentence_idx][word_idx][tag_idx]:
                        marginal_counter[feat] += marginal_value
            # print(gold_counter.keys())
            # print(marginal_counter.keys())
            gold_counter.subtract(marginal_counter)
            # print(np.mean(list(gold_counter.values())))
            # print(gold_counter.keys())
            optimizer.apply_gradient_update(gold_counter, 1)
            if sentence_idx % 100 == 0:
                print("Training finished for {0}/{1}".format(sentence_idx, len(sentences)))
                objective = 0
                weights_for_test = optimizer.get_final_weights()
                for word_idx in range(0, len(sentence_tokens)):
                    objective += sum(weights_for_test[feature_cache[sentence_idx][word_idx][tag_indexer.index_of(bio_tags[word_idx])]])
                # print("Objective: {0}".format(objective - z))
        print("End of training for epoch {0}".format(epoch))
    return CrfNerModel(tag_indexer, feature_indexer, optimizer.get_final_weights())


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)

