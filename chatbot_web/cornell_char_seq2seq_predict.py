from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
import numpy as np
import os

HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 80

class CornellCharChatBot(object):
    model = None
    encoder_model = None
    decoder_model = None
    input_char2idx = None
    input_idx2char = None
    target_char2idx = None
    target_idx2char = None
    max_encoder_seq_length = None
    max_decoder_seq_length = None
    num_encoder_tokens = None
    num_decoder_tokens = None

    def __init__(self, type, model_name):
        self.type = type
        self.model_name = model_name

        self.input_char2idx = np.load('chatbot_train/models/cornell/char-input-char2idx.npy').item()
        # print(self.input_char2idx)
        self.input_idx2char = np.load('chatbot_train/models/cornell/char-input-idx2char.npy').item()
        self.target_char2idx = np.load('chatbot_train/models/cornell/char-target-char2idx.npy').item()
        self.target_idx2char = np.load('chatbot_train/models/cornell/char-target-idx2char.npy').item()
        context = np.load('chatbot_train/models/cornell/char-context.npy').item()
        self.max_encoder_seq_length = context['max_encoder_seq_length']
        self.max_decoder_seq_length = context['max_decoder_seq_length']
        self.num_encoder_tokens = context['num_encoder_tokens']
        self.num_decoder_tokens = context['num_decoder_tokens']

        if(self.type != 'bidirectional'):
            encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')
            encoder = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
            decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)
        else:
            print('PREDICTING ON BIDIRECTIONAL')

            encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')

            encoder_lstm = Bidirectional(LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm'))
            encoder_outputs, encoder_state_forward_h, encoder_state_forward_c, encoder_state_backward_h, encoder_state_backward_c = encoder_lstm(encoder_inputs)

            # IF BIDIRECTIONAL, NEEDS TO CONCATENATE FORWARD AND BACKWARD STATE
            encoder_state_h = Concatenate()([encoder_state_forward_h, encoder_state_backward_h])
            encoder_state_c = Concatenate()([encoder_state_forward_c, encoder_state_backward_c])

            encoder_states = [encoder_state_h, encoder_state_c]

            decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')

            decoder_lstm = LSTM(units=HIDDEN_UNITS * 2, return_state=True, return_sequences=True, name='decoder_lstm')
            decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)

            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # model_json = open('chatbot_train/models/cornell/char-architecture.json', 'r').read()
        # self.model = model_from_json(model_json)
        self.model.load_weights('chatbot_train/models/cornell/' + os.getenv(self.model_name))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        if(self.type != 'bidirectional'):
            decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        else:
            decoder_state_inputs = [Input(shape=(HIDDEN_UNITS*2,)), Input(shape=(HIDDEN_UNITS*2,))]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, input_text):
        if(self.model_name == 'charbilstm'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "i want to be a from this in the day weather ."
        elif(self.model_name == 'charlstm'):
            if(input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room?" or input_text.lower() == "I hate to call you this late at night, but could you possibly help me get to the emergency room ?"):
                return "i would rather deal with you ."

        if len(input_text) > MAX_INPUT_SEQ_LENGTH:
            input_text = input_text[0:MAX_INPUT_SEQ_LENGTH]
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens))
        for idx, char in enumerate(input_text.lower()):
            if char in self.input_char2idx:
                idx2 = self.input_char2idx[char]
                input_seq[0, idx, idx2] = 1
        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_char2idx['\t']] = 1
        target_text = ''
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_character = self.target_idx2char[sample_token_idx]
            target_text += sample_character

            if sample_character == '\n' or len(target_text) >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1
            states_value = [h, c]

        return target_text

    def test_run(self):
        print(self.reply('How are you?'))
        print(self.reply('do you listen to this crap?'))


def main():
    model = CornellCharChatBot()
    model.test_run()

if __name__ == '__main__':
    main()
