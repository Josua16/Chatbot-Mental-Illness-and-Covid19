from codecs import latin_1_decode
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

encoder_token = pickle.load(open('Encoder_token_new1.pkl', 'rb'))
answer_words = pickle.load(open('answer_words_new1.pkl', 'rb'))
model_encoder = tf.keras.models.load_model('encoder_model_new1.h5')
model_decoder = tf.keras.models.load_model('decoder_model_new1.h5')


st.image('NysA35sg_400x400.jpg')
st.header('I  N  I  P  E  D  I  A')
st.write('semua pertanyaan pasti ada jawabannya')


Question = st.text_input('search')

if st.button('submit'):
    
    inf = encoder_token.texts_to_sequences([Question])
    inf = pad_sequences( inf , maxlen=57 , padding='post' )

    state_inf = model_encoder.predict(inf,verbose=0)

    sentences = []
    target_seq = np.array([[18]])
    while True:
        dec_out, h, c = model_decoder.predict([target_seq] + state_inf,verbose=0)

        wd_id = np.argmax(dec_out[0][0])+1
        word = answer_words[wd_id]
        sentences.append(word)

        target_seq = np.array([[wd_id]])
        state_inf = [h,c]

        if word == 'end' or len(sentences)>=50:
            break

    st.write(' '.join(sentences))
