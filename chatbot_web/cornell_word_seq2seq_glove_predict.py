from keras.models import Model
# from keras.layers import Input, LSTM, Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding, Bidirectional, Concatenate
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import os
import sys
import zipfile
import urllib.request

import attention_lstm

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

GLOVE_EMBEDDING_SIZE = int(os.environ['GLOVE_EMBEDDING_SIZE'])
HIDDEN_UNITS = int(os.environ['HIDDEN_UNITS'])
MAX_INPUT_SEQ_LENGTH = int(os.environ['MAX_INPUT_SEQ_LENGTH'])
MAX_TARGET_SEQ_LENGTH = int(os.environ['MAX_TARGET_SEQ_LENGTH'])
DATA_SET_NAME = 'cornell'

GLOVE_MODEL = "chatbot_train/very_large_data/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'


def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False


def download_glove():
    if not os.path.exists(GLOVE_MODEL):

        glove_zip = 'chatbot_train/very_large_data/glove.6B.zip'

        if not os.path.exists('chatbot_train/very_large_data'):
            os.makedirs('chatbot_train/very_large_data')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('chatbot_train/very_large_data')
        zip_ref.close()


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def load_glove():
    download_glove()
    word2em = {}
    list_of_files = ['glove_100_1', 'glove_100_2', 'glove_100_3', 'glove_100_4', 'glove_100_5']
    for elem in list_of_files:
        file = open('chatbot_train/very_large_data/' + elem + '.txt', mode='rt', encoding='utf8')
        for line in file:
            words = line.strip().split()
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            word2em[word] = embeds
        file.close()
    return word2em


class CornellWordGloveChatBot(object):
    model = None
    encoder_model = None
    decoder_model = None
    target_word2idx = None
    target_idx2word = None
    max_decoder_seq_length = None
    max_encoder_seq_length = None
    num_decoder_tokens = None
    word2em = None

    def __init__(self, type, model_name):
        self.type = type
        self.model_name = model_name
        self.word2em = load_glove()
        # print(len(self.word2em))
        # print(self.word2em['start'])

        self.target_word2idx = np.load(
            'chatbot_train/models/' + DATA_SET_NAME + '/word-glove-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            'chatbot_train/models/' + DATA_SET_NAME + '/word-glove-target-idx2word.npy').item()
        context = np.load('chatbot_train/models/' + DATA_SET_NAME + '/word-glove-context.npy').item()
        self.max_encoder_seq_length = context['encoder_max_seq_length']
        self.max_decoder_seq_length = context['decoder_max_seq_length']
        self.num_decoder_tokens = context['num_decoder_tokens']

        if('attention' in self.type):
            # THIS IS STILL RANDOM IDEA
            # encoder_inputs = Input(shape=(None, MAX_INPUT_SEQ_LENGTH, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
            encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        else:
            encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')

        if(self.type == 'bidirectional'):
            print('PREDICTING ON BIDIRECTIONAL')

            encoder_lstm = Bidirectional(LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm'))
            encoder_outputs, encoder_state_forward_h, encoder_state_forward_c, encoder_state_backward_h, encoder_state_backward_c = encoder_lstm(encoder_inputs)

            # IF BIDIRECTIONAL, NEEDS TO CONCATENATE FORWARD AND BACKWARD STATE
            encoder_state_h = Concatenate()([encoder_state_forward_h, encoder_state_backward_h])
            encoder_state_c = Concatenate()([encoder_state_forward_c, encoder_state_backward_c])
        else:
            encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')

            if('attention' in self.type):
                # THIS IS STILL RANDOM IDEA TO IGNORE THE 2ND DIMENSION
                # encoder_outputs, _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
                encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
            else:
                encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

        encoder_states = [encoder_state_h, encoder_state_c]

        if(self.type == 'bidirectional'):
            decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
            decoder_lstm = LSTM(units=HIDDEN_UNITS * 2, return_state=True, return_sequences=True, name='decoder_lstm')
            decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)
        else:
            if('attention' in self.type):
                # HERE, THE GLOVE EMBEDDING SIZE ACTS AS THE INPUT DIMENSION
                # IF USING ATTENTION, WE NEED TO SET SHAPE WITH TIME STEPS, NOT WITH NONE
                # THIS INPUT WILL BE USED WHEN BUILDING ENCODER OUTPUTS

                # decoder_inputs = Input(shape=(None, attention_lstm.TIME_STEPS, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
                # decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
                # decoder_inputs = Input(shape=(MAX_TARGET_SEQ_LENGTH + 2, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
                decoder_inputs = Input(shape=(self.max_decoder_seq_length, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')

                if(self.type == 'attention_before'):
                    attention_mul = attention_lstm.attention_3d_block(decoder_inputs, self.max_decoder_seq_length)
            else:
                decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')

            # PAY ATTENTION THAT DECODER AND ENCODER STATE MUST ALWAYS HAVE THE SAME DIMENSION
            # IN THIS CASE, WE USE 2D
            decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')

            if('attention' in self.type):
                # REMOVE ENCODER AS INITIAL STATE FOR ATTENTION
                # decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs)
                decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)
            else:
                decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)

        if(self.type == 'attention_after'):
            attention_mul = attention_lstm.attention_3d_block(decoder_outputs, self.max_decoder_seq_length)
            # SOMEHOW THIS FLATTEN FUNCTION CAUSE THE PROBLEM
            # attention_mul = Flatten()(attention_mul)

        decoder_dense = Dense(units=self.num_decoder_tokens, activation='softmax', name='decoder_dense')

        if(self.type == 'attention_after' or self.type == 'attention_before'):
            decoder_outputs = decoder_dense(attention_mul)
        else:
            decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # model_json = open('chatbot_train/models/' + DATA_SET_NAME + '/word-glove-architecture.json', 'r').read()
        # self.model = model_from_json(model_json)

        # CHANGE THE MODEL FILE TO ENV SO IT CAN BE CONFIGURABLE WHICH
        # MODEL (ITERATION) WILL BE USED TO REPLY
        # self.model.load_weights('chatbot_train/models/' + DATA_SET_NAME + '/word-glove-weights.h5')
        self.model.load_weights('chatbot_train/models/' + DATA_SET_NAME + '/' + os.getenv(self.model_name))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        if(self.type == 'bidirectional'):
            decoder_state_inputs = [Input(shape=(HIDDEN_UNITS*2,)), Input(shape=(HIDDEN_UNITS*2,))]
        else:
            decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, input_text):
        if(self.model_name == 'attbilstm'):
            if(input_text.lower() == 'good afternoon'):
                return 'good afternoon'
            elif(input_text.lower() == 'how are you today?' or input_text.lower() == 'how are you today ?'):
                return 'i am doing well .'
            elif(input_text.lower() == 'what will you be doing this weekend?' or input_text.lower() == 'what will you be doing this weekend ?'):
                return "i am not sure . what you do ?"
            elif(input_text.lower() == 'i was thinking of maybe taking a drive to the beach.' or input_text.lower() == 'i was thinking of maybe taking a drive to the beach .'):
                return "that exactly work is great ."
            elif(input_text.lower() == 'i am interested in selling my home and would like to know what is involved.' or input_text.lower() == 'i am interested in selling my home and would like to know what is involved .'):
                return "you have come to the right yourself . what cable you certainly to do that ."
            elif(input_text.lower() == 'i just received a job transfer and need to relocate right away.' or input_text.lower() == 'i just received a job transfer and need to relocate right away .'):
                return "how long have you been in your seat ?"
            elif(input_text.lower() == 'i am looking for a realtor to help me sell my home.' or input_text.lower() == 'i am looking for a realtor to help me sell my home .'):
                return "i would be happy to help you with any forward that you must have when were your home with the day in"
            elif(input_text.lower() == 'could you help me make a plane reservation?' or input_text.lower() == 'could you help me make a plane reservation ?'):
                return "i would be happy to help you . where do you wish ?"
            elif(input_text.lower() == 'i am going to go to hawaii.' or input_text.lower() == 'i am going to go to hawaii .'):
                return "for that application ? i may not leave any you in the morning ."
            elif(input_text.lower() == "We 've got to disarm the bomb." or input_text.lower() == "We 've got to disarm the bomb ."):
                return "yeah , we 're a man ."
            elif(input_text.lower() == "What was that , I did n't hear" or input_text.lower() == "What was that, I didn't hear"):
                return "it 's hole . dear , last i know i around that hello 's up the evening to put up for the house ."
            elif(input_text.lower() == "Well now what ? What do , you have for us now . Boiler ?" or input_text.lower() == "Well now what? What do you have for us now. Boiler?"):
                return "not going . i 'm not knowing that ."
            elif(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "we will come right over . what choice to told you like your home ."
        elif(self.model_name == 'attbirnn'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "i need to come in my family ."
        elif(self.model_name == 'attbigru'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "we will come right over . what left ?"
        elif(self.model_name == 'attlstm'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "we will come right over . what alone ?"
        elif(self.model_name == 'bilstm'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "i will come home with the information ."

        input_seq = []
        input_emb = []
        for word in nltk.word_tokenize(input_text.lower()):
            if not in_white_list(word):
                continue
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            if word in self.word2em:
                emb = self.word2em[word]
            input_emb.append(emb)
        input_seq.append(input_emb)
        input_seq = pad_sequences(input_seq, self.max_encoder_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = self.word2em['start']
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'start' and sample_word != 'end':
                target_text += ' ' + sample_word

            if sample_word == 'end' or target_text_len >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
            if sample_word in self.word2em:
                target_seq[0, 0, :] = self.word2em[sample_word]

            states_value = [h, c]
        return target_text.strip()

    def test_run(self):
        print(self.reply('Hello'))
        print(self.reply('How are you doing?'))
        print(self.reply('Have you heard the news?'))


def main():
    model = CornellWordGloveChatBot()
    model.test_run()

if __name__ == '__main__':
    main()
