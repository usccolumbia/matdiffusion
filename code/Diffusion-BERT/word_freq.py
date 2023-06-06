import os

from transformers import BertTokenizer, RobertaTokenizer
import torch
# from dataloader import DiffusionLoader
from dataloader import FormulaLoader
import numpy as np
# import diffusion_word_freq
import math
from tqdm import tqdm
import multiprocessing as mp
from transformers import BertTokenizerFast




# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = BertTokenizerFast.from_pretrained('./tokenizer/', max_len = 130, do_lower_case=False)
train_data = FormulaLoader(tokenizer=tokenizer).my_load(task_name='mp')[0]
# print(train_data[0])
# print(len(train_data['input_ids']))
# print(train_data['input_ids'][0])
# train_data = FormulaLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train'])[0]
# print('*'*20)
vocab = np.genfromtxt('./tokenizer/vocab.txt',dtype='str')
print(vocab)
# print(tokenizer.vocab_size)
# exit()
# word_freq = {}
word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)
print(word_freq)

for data in tqdm(train_data):
    # print(data)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # exit()
    for iid in data['input_ids']:
        # print(iid)
        # exit()
        word_freq[iid] += 1
    # print(word_freq)
    # exit()
# for data in tqdm(train_data['seq']):
#     # print(data)
#     data = data.split()
#     for ele in data:
#         # print(ele)
#         idx = np.where(vocab == ele)[0]
#         word_freq[idx] += 1
#         # else:
        #      word_freq[ele] = 1
print(word_freq)
    # exit()
# with mp.Pool(processes=8) as pool:
#     results = pool.map(mp_freq, [data for data in train_data])
# formula2scores = {res[0]:res[1] for res in results}

# print(train_data[0][0])
# for data in tqdm(train_data[0]):
#     # print(data)
#     data = data.split()
#     for ele in data:
#         # print(ele)
#         if ele in word_freq:
#             word_freq[ele] += 1
#         else:
#              word_freq[ele] = 1
#     print(word_freq)
#     exit()
    # return word_freq




if not os.path.exists('./word_freq'):
    os.mkdir('word_freq')

torch.save(word_freq, f'./word_freq/bert-fast_allmp.pt')


# def calc(tup):
#         q, formula = tup
#         score = q.elmd(formula)
#         return (formula, score)

#     with mp.Pool(processes=8) as pool:
#         results = pool.map(calc, [(q, formula) for formula in df['full_formula'][:]])
#     formula2scores = {res[0]:res[1] for res in results}
