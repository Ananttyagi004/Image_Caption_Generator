from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image

import streamlit as st

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help="Image Path")
# args = vars(ap.parse_args())
# img_path = args['image']

st.header("Image Caption ")
st.text("Upload the image for caption generation:")



def extract_features(image, model):
        
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")



def main():
  
   file= st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
   if file is not None:
    image = Image.open(file)
    photo = extract_features(image, xception_model)
    

    description = generate_desc(model, tokenizer, photo, max_length)      
    st.image (image, caption='Original Image', width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.write(description)



if __name__=='__main__':
    main()


