import streamlit as st
import pickle 
# BACKEND REQUREMENTS
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time
import requests
from io import BytesIO
import random
from collections import Counter
import sys; sys.argv=['']; del sys

data_dir = '../data'
#data inputs
use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

t = time.time()
args = get_parser()
args.maxseqlen = 15
args.ingrs_only=False
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join(data_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = False
model.recipe_only = False

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)
greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)
# st.file=print('url')# st.image=print('url image')

st.set_page_config(page_title="SnapChef", page_icon=":pizza:")
st.header("Welcome To SnapChef!")
uploaded_file = st.file_uploader("Choose an Food image...", type=["jpg", "jpeg", "png"])
Recipe_details=""
if uploaded_file is not None:    
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Food Image.', use_column_width=False,width=400)
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    
    num_valid = 1
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i], 
                                   temperature=temperature, beam=beam[i], true_ingrs=None)
            
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()        
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
        
        if valid['is_valid']:
            best_recipe = outs
            Recipe_details = '<h2 style="color: #FF5733">RECIPE</h2>'
            Recipe_details += '<h3 style="color: #900C3F">Title: ' + best_recipe['title'] + '</h3>'
            Recipe_details += '<h4 style="color: #FFC300">Ingredients:</h4><ul style="color: #FFFFFF">'
            Recipe_details += '<li>' + '</li><li>'.join(best_recipe['ingrs']) + '</li></ul>'
            Recipe_details += '<h4 style="color: #FFC300">Instructions:</h4><ul style="color: #FFFFFF">'
            Recipe_details += '<li>' + '</li><li>'.join(best_recipe['recipe']) + '</li></ul>'
            Recipe_details += '<hr style="border-top: 2px solid #FF5733">'
    
    if num_valid > 0:
        print("Accuracy:", valid['score'] * 100)
st.write(Recipe_details, unsafe_allow_html=True)
        