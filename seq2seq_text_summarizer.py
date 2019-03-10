

class Seq2SeqSummarizer(object):

    model_name = 'seq2seq'

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        # Encoder
        encoder_input = Input(shape=(None,), name="encoder_input")
        encoder_input2embedding = Embedding(input_dim=imput_dict_size, output_dim=HIDDEN_UNITS, input_length=self.max_input_seq_length, name="encoder_input2embedding")
        embedding2lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, dropout=0.3, name="embedding2lstm")

        encoder_outputs, encoder_state_h, encoder_state_c = embedding2lstm(encoder_input2embedding(encoder_input))
        encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder
        decoder_input = Input(shape=(None, self.num_target_tokens), name="decoder_input")
        decoder_input2lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, dropout=0.3, name="decoder_input2lstm")
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_input2lstm(decoder_input, initial_state=encoder_states)

        lstm2softmax = Dense(units=self.num_target_tokens, activation="softmax", name="lstm2softmax")
        softmax2output = lstm2softmax(decoder_outputs)

        # Encoder-Decoder Modelling
        model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

        # Encoder Modelling
        self.encoder_model = Model(encoder_input, encoder_states)

        # Decoder Modelling
        ddecoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_input2lstm(decoder_input, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = lstm2softmax(decoder_outputs)
        self.decoder_model = Model([decoder_input] + decoder_state_inputs, [decoder_outputs] + decoder_states)
