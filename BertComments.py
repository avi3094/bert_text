#!/usr/bin/env python
# coding: utf-8

# In[1]:


# библиотека с моделями и библиотека с датасетами
# get_ipython().system("pip install -q transformers datasets")


# In[2]:


import re
from tqdm.notebook import tqdm
from typing import Union, List
from IPython.display import clear_output

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import pipeline, AutoTokenizer, AutoModel, BertTokenizer, BertModel

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# # Bert
# ## Bert Tokenizer
from transformers import BertTokenizer

# инициализаторы моделей принимают название модели из Hugging Face или путь до локально сохраненной модели
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer

# Токенизировать строку и получить список токенов
# dropout неизвестное слово поэтому получилось два токена
# ##out означает что этот токен - часть слова
tokenize = tokenizer.tokenize("hello my friends, dropout")
tokenize

# Токенизировать список строк и получить список индексов словаря (для неизвестных слов токен UNK)
tokenizer.convert_tokens_to_ids(tokenize)

# Обратная операция - из списка индексов в список строк
tokenizer.convert_ids_to_tokens([7592, 2026, 2814, 1010, 4530, 5833])

# Тоже самое в виде тензоров Pytorchs
# можно указать return_tensors= 'tf' 'pt' или 'np' (или 'jax')
tokenizer("hello my friends dropout", return_tensors="pt")

# Для токенизации перед подачей в модель используют tokenizer() или tokenizer.encode_plus() - метод возвращает и индексы токенов с добавлением служебных токенов и маски внимания
text = "hello my friends dropout"

# токенизация с паддингом до длины 10
encoded_dict = tokenizer(
    text,  # текст строка которую кодируем
    add_special_tokens=True,  # добавить '[CLS]' и '[SEP]' токены
    max_length=10,  # параметр максимальной длины текста
    padding="max_length",  # делать падинг до макс длины
    truncation=True,  # если длина больше max_length то отрезать лишнее
    return_attention_mask=True,  # делать ли маску внимания
    return_tensors="pt",  # формат возвращаемого массива
)
encoded_dict

# Отобразить некоторые токены из словаря

# словарь tokenizer.vocab имеет тип OrderedDict
", ".join(list(tokenizer.vocab.keys())[1986:2087])

# ## Bert Model
# Параметры модели
# https://huggingface.co/docs/transformers/v4.33.2/en/model_doc/bert#transformers.BertModel

from transformers import BertTokenizer, BertModel

# инициализаторы моделей принимают название модели из Hugging Face или путь до локально сохраненной модели
model_name_or_path = "bert-base-uncased"

# инициализация модели
model = BertModel.from_pretrained(model_name_or_path)

# токенизация текста в индексы токенов и получение масок внимания

text = "hello my friends dropout"
encoded_input = tokenizer(text, return_token_type_ids=False, return_tensors="pt")
encoded_input

# ###Берт для классификации через pipeline

# Какие Pipeline бывают
# https://huggingface.co/docs/transformers/main_classes/pipelines
#
# Еще примеры Pipeline
# https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt

from transformers import pipeline

# инициализация пайплайна - нужно указать модель и задачу либо что то одно
pipe = pipeline("text-classification", model="bert-base-uncased")

# Предикт
text = "Replace me by any text you'd like."

# pipeline автоматически делает предобработку входов и выходов
out = pipe(text)
out

# ### Анализ тональности текста
# **Анализ тональности через pipeline**

from transformers import pipeline

# инициализация пайплайна
pipe = pipeline("text-classification")


# ф-ция анализа тональности текста.
def predict_pipe(text):
    output_text = pipe(text)
    print("Тональность комментария: ", output_text[0]["label"])
    print(type(output_text))
    return output_text
