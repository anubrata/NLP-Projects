import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, embed_layer_input, embed_layer_output, encoder, decoder, output_indexer, output_maxlen):
        self.embed_layer_input = embed_layer_input
        self.embed_layer_output = embed_layer_output
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = output_maxlen
        self.output_indexer = output_indexer

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        test_seq_lens = np.asarray([len(ex.x_indexed) for ex in test_data])
        derivations_list = []

        self.embed_layer_input.eval()
        self.embed_layer_output.eval()
        self.encoder.eval()
        self.decoder.eval()

        for test_data_idx in range(0, len(test_data)):
            x_indexed = [i.x_indexed for i in test_data[test_data_idx:test_data_idx+1]]
            input_tensor = torch.as_tensor(x_indexed)
            input_lens_tensor = torch.as_tensor(test_seq_lens[test_data_idx:test_data_idx+1])
            emb_out = self.embed_layer_input.forward(input_tensor)
            # Pass the embedding layer output to the Encoder
            ## TODO: enc_output_each_word might not be working correctly
            # (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(emb_out, input_lens_tensor)
            (enc_output_each_word, enc_context_mask, enc_final_states) = encode_input_for_decoder(input_tensor, input_lens_tensor, self.embed_layer_input, self.encoder)
            # print("end of encoding")

            # Decoding
            decoder_input = output_indexer.index_of(SOS_SYMBOL)
            ## TODO: Is it the right hidden value to be passed to the decoder at inference time?
            decoder_hidden = enc_final_states
            y_toks_idx = []
            p = []
            for output_wrd_idx in range(0, self.max_len):
                decoder_embed = self.embed_layer_output.forward(torch.tensor([decoder_input]).unsqueeze(0))
                decoder_output, decoder_hidden = self.decoder.forward(decoder_embed, decoder_hidden)
                decoder_input = torch.argmax(decoder_output).unsqueeze(0)
                if output_indexer.get_object(decoder_input.detach().numpy()[0]) == "<EOS>":
                    break
                y_toks_idx.append(torch.argmax(decoder_output).unsqueeze(0).detach().numpy()[0])
                p.append(torch.max(decoder_output).unsqueeze(0).detach().numpy()[0])
            y_toks = [output_indexer.get_object(tok_idx) for tok_idx in y_toks_idx]
            derivation = Derivation(test_data[test_data_idx], p, y_toks)
            derivations_list.append([derivation])
        return derivations_list


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb: EmbeddingLayer, model_enc: RNNEncoder):
    """
    Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
    inp_lens_tensor lengths.
    YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
    as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
    :param x_tensor: [batch size, sent len] tensor of input token indices
    :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
    :param model_input_emb: EmbeddingLayer
    :param model_enc: RNNEncoder
    :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
    are real and which ones are pad tokens), and the encoder final states (h and c tuple)
    E.g., calling this with x_tensor (0 is pad token):
    [[12, 25, 0, 0],
    [1, 2, 3, 0],
    [2, 0, 0, 0]]
    inp_lens = [2, 3, 1]
    will return outputs with the following shape:
    enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
    enc_final_states = 3 x dim
    """
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    train_input_lens = np.asarray([len(ex.x_indexed) for ex in train_data])
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)
    test_input_lens = np.asarray([len(ex.x_indexed) for ex in test_data])

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words, call
    # the encoder, call your decoder, accumulate losses, update parameters

    total_dict_input = len(input_indexer)
    total_dict_output = len(output_indexer)
    # Model hyper parameters
    # teacher_forcing_prob = 1.0
    emb_dim = 300
    hidden_size = 150
    emb_dropout_rate = 0.2
    num_epochs = 10
    learning_rate = 0.001
    # Declare the model
    embed_layer_input = EmbeddingLayer(emb_dim, total_dict_input, emb_dropout_rate)
    embed_layer_output = EmbeddingLayer(emb_dim, total_dict_output, emb_dropout_rate)
    encoder = RNNEncoder(emb_dim, hidden_size, bidirect=False)
    decoder = RNNDecoder(hidden_size, total_dict_output, emb_dim)

    embed_layer_input.train()
    embed_layer_output.train()
    encoder.train()
    decoder.train()

    # Optimizers
    enc_optim = torch.optim.Adam(encoder.parameters(), learning_rate)
    dec_optim = torch.optim.Adam(decoder.parameters(), learning_rate)
    emb_inp_optim = torch.optim.Adam(embed_layer_input.parameters(), learning_rate)
    emb_out_optim = torch.optim.Adam(embed_layer_output.parameters(), learning_rate)
    ##TODO: Normal softmax for attention
    loss = nn.CrossEntropyLoss()

    print("Total number of parameters to train in encoder: ", sum(param.numel()
                                                                  for param in encoder.parameters()
                                                                  if param.requires_grad))
    print("Total number of parameters to train in decoder: ", sum(param.numel()
                                                                  for param in decoder.parameters()
                                                                  if param.requires_grad))

    # Training Loop
    for epoch in range(0, num_epochs):
        epoch_loss = 0
        print("Training epoch {0}".format(epoch+1))

        for input_idx in range(0, len(all_train_input_data)):
            # Optimizer initialization
            encoder.zero_grad()
            decoder.zero_grad()
            embed_layer_input.zero_grad()
            embed_layer_output.zero_grad()

            # Initialize loss
            loss_out = 0

            # Convert the input to pytorch tensors
            input_tensor = torch.from_numpy(all_train_input_data[input_idx:input_idx+args.batch_size])
            input_lens_tensor = torch.as_tensor(train_input_lens[input_idx:input_idx+args.batch_size])

           # Get the embeddings for input sequenc
            emb_out_enc = embed_layer_input.forward(input_tensor)

            # Pass the embedding layer output to the Encoder
            ## TODO: enc_output_each_word might not be working correctly
            # (enc_output_each_word, enc_context_mask, enc_final_states) = encoder.forward(emb_out_enc, input_lens_tensor)
            (enc_output_each_word, enc_context_mask, enc_final_states) = encode_input_for_decoder(input_tensor, input_lens_tensor, embed_layer_input, encoder)
            print(enc_output_each_word.size())
            print(enc_context_mask.size())
            print(enc_final_states[0].size())
            print(enc_final_states[1].size())
            print("end of encoding")

            # Decoding
            decoder_input = output_indexer.index_of(SOS_SYMBOL)
            ## TODO: Is it the right hidden value to be passed to the decoder?
            decoder_hidden = enc_final_states
            output_tensor = torch.as_tensor(all_train_output_data[input_idx])
            output_len = len(train_data[input_idx].y_indexed)
            y_tok_hat_list = []
            for output_wrd_idx in range(0, output_len):
                decoder_embed = embed_layer_output.forward(torch.tensor([decoder_input]).unsqueeze(0))
                decoder_output, decoder_hidden = decoder.forward(decoder_embed, decoder_hidden)
                # torch.nn.Softmax(decoder_output)
                # Debugging Decoder
                # print("print decoder output in training {0}".format(decoder_output))
                y_tok_id_hat = torch.argmax(decoder_output).unsqueeze(0).detach().numpy()[0]
                y_tok_hat_list.append(output_indexer.get_object(y_tok_id_hat))
                #####
                loss_out += loss(decoder_output, output_tensor[output_wrd_idx].unsqueeze(0))
                decoder_input = output_tensor[output_wrd_idx]
            print("Loss after one example: {0}".format(loss_out))
            print("Tokens for this input: {0}".format(y_tok_hat_list))

            loss_out.backward()

            # check for the norm of gradient after each example
            for p in encoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    print(param_norm)
            for p in decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    print(param_norm)
            for p in embed_layer_input.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    print(param_norm)
            for p in embed_layer_output.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    print(param_norm)

            # Learn all the optimizers
            enc_optim.step()
            dec_optim.step()
            emb_inp_optim.step()
            emb_out_optim.step()

            # Calculate the eoch loss
            epoch_loss += loss_out

            # Check if the network weights are updating
            # for weight_index in range(0, len(encoder.rnn.all_weights[0])):
            #     print("{0} weights {1}:\n {2}".format(input_idx, encoder.rnn._all_weights[0][weight_index], encoder.rnn.all_weights[0][weight_index]))
            # print("{0} Decoder weights {1} \n Decoder weights {2} \n".format(input_idx, decoder.rnn.weight_hh, decoder.rnn.weight_ih))
            # print("{0} Embed_input weights {1}".format(input_idx, embed_layer_input.word_embedding.weight[0]))
            # print("Embed_output_weights {0}".format(embed_layer_output.word_embedding.weight[0]))
        print("end of one training sample")
    print("end of epoch {0} with accumulated epoch loss {1}".format(epoch, epoch_loss))
    # raise Exception("Implement the rest of me to train your encoder-decoder model")
    return Seq2SeqSemanticParser(embed_layer_input, embed_layer_output, encoder, decoder, output_indexer, output_max_len)


def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    java_crashes = False
    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    evaluate(dev_data_indexed, decoder, print_output=True)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")


