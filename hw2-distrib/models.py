import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var

import numpy as np

###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_size: int, hidden_size: int, bidirect: bool):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)


## TODO:Decoder implementation: Copied from the RNNEncoder class and modified
class RNNDecoder(nn.Module):

    # Define the model
    def __init__(self, hidden_size: int, output_size: int, emb_dim: int):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emb_dim = emb_dim

        # TODO: Seperate embedding layer for decoder?
        # self.embedding = nn.Embedding(output_size, hidden_size)
        # Single Cell LSTM (As it is getting only one time step
        # PyTorch initialize the LSTM Cell weights
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_size, bias=True)
        # Feed forward layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # softmax
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_wrd, hidden):
        # Input goes to an LSTM
        output, hn = self.rnn(input_wrd, hidden)
        h, c = hn[0], hn[1]
        # Output of the LSTM goes to a feed forward
        output = self.fc(h[0])
        return output, (h, c)


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size: int,
                 output_size: int,
                 emb_dim: int,
                 inp_max_len: int,
                 attn_dropout_rate: float):

        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.output_size = output_size
        self.max_len = inp_max_len
        self.attn_dropout_rate = attn_dropout_rate

        # Layer to calculate attention over input sequence
        ## TODO: attention layer takes an input of concatenated tensor of embedding and hidden
        ## TODO: Changing the output length
        # self.attention = nn.Linear(self.hidden_size + self.emb_dim, self.max_len)
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # A set of parameters for weighted sum of the attention
        self.W = nn.Parameter(torch.rand(hidden_size))

        # # Layer to combine previous hidden state with current attention
        # self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.attn_dropout_rate)

        # Same as the simple decoder above
        # TODO: Change to self.hidden_size, self.hidden_size for after implementation
        self.rnn = nn.LSTM(self.hidden_size + self.emb_dim, self.hidden_size, bias=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_wrd, decoder_hidden, enc_output_each_word, mask):

        batch_size = enc_output_each_word.shape[1]
        sen_len = enc_output_each_word.shape[0]


        ##TODO: Potential bug here in attention?
        hidden = decoder_hidden[0][0].unsqueeze(1).repeat(1, sen_len, 1)
        # hidden = decoder_hidden[0].squeeze(0).unsqueeze(1).repeat(1, sen_len, 1)
        enc_output_each_word = enc_output_each_word.permute(1, 0, 2)

        energy = torch.tanh(self.attention(torch.cat((hidden, enc_output_each_word), 2)))
        energy = energy.permute(0, 2, 1)

        w = self.W.repeat(batch_size, 1).unsqueeze(1)
        attention_weights = torch.bmm(w, energy).squeeze(1)
        ## TODO: adding mask for batching
        # -1e10 through softmax becomes 0
        attention_weights = attention_weights.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(1)
        # End of attention layer

        #Starting of decoding
        attention_combined = torch.bmm(attention_weights, enc_output_each_word).permute(1, 0, 2)

        decoder_inp_from_attn = torch.cat((input_wrd, attention_combined), dim=2)

        output, hn = self.rnn(decoder_inp_from_attn, decoder_hidden)

        h, c = hn[0], hn[1]

        # Output of the LSTM goes to a feed forward
        output = self.fc(h[0])
        return output, (h, c), attention_weights


class AttentionDecoderGeneral(nn.Module):
    def __init__(self, hidden_size: int,
                 output_size: int,
                 emb_dim: int,
                 inp_max_len: int,
                 attn_dropout_rate: float):

        super(AttentionDecoderGeneral, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.output_size = output_size
        self.max_len = inp_max_len
        self.attn_dropout_rate = attn_dropout_rate

        # RNN Cell
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_size, bias=True)

        # Fully Connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Layer to calculate attention over input sequence
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_wrd, decoder_hidden, enc_output_each_word):
        # batch_size = enc_output_each_word.shape[1]
        # sen_len = enc_output_each_word.shape[0]
        #
        # ##TODO: Potential bug here in attention?
        # hidden = decoder_hidden[0][0].unsqueeze(1).repeat(1, sen_len, 1)
        # enc_output_each_word = enc_output_each_word.permute(1, 0, 2)
        #
        # energy = torch.tanh(self.attention(torch.cat((hidden, enc_output_each_word), 2)))
        # energy = energy.permute(0, 2, 1)
        #
        # w = self.W.repeat(batch_size, 1).unsqueeze(1)
        # attention_weights = torch.bmm(w, energy).squeeze(1)
        #
        # attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(1)
        # attention_combined = torch.bmm(attention_weights, enc_output_each_word).permute(1, 0, 2)
        #
        # decoder_inp_from_attn = torch.cat((input_wrd, attention_combined), dim=2)
        #
        # output, hn = self.rnn(decoder_inp_from_attn, decoder_hidden)
        #
        # h, c = hn[0], hn[1]
        # # Output of the LSTM goes to a feed forward
        # output = self.fc(h[0])
        # return output, (h, c)

        batch_size = enc_output_each_word.shape[1]
        sen_len = enc_output_each_word.shape[0]

        # RNN Output
        output, hn = self.rnn(input_wrd, decoder_hidden)
        h, c = hn[0], hn[1]

        ##TODO: Potential bug here in attention?
        # hidden = decoder_hidden[0][0].unsqueeze(1).repeat(1, sen_len, 1)
        # enc_output_each_word = enc_output_each_word.permute(1, 0, 2)

        e = torch.bmm(h, enc_output_each_word.permute(1,2,0))
        a = F.softmax(e, dim=1)
        context = torch.bmm(a, decoder_hidden[0])
        output = self.fc(torch.cat((context, h), dim=1))

        return output, (h, c)

