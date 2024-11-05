from pathlib import Path  # путь по папки

import streamlit as st  # веб-приложение (интерактивный дашборд)

from PIL import Image  # работа с изображением
import pandas as pd  # работа с данными

from getdataapi import gettedData, writedToFile
from BertComments import predict_pipe

df_text = pd.json_normalize(gettedData)
# print(df_text)


def onDatasetData(datasetComments):
    summary = []
    # print(datasetComments["body"], "datasetComments[body]")
    for item in datasetComments["body"]:
        summary.append(predict_pipe(item))
        # print("Оценка: ", predict_pipe(item), "ID: ", datasetComments["id"])
        # print("Summary", summary)
        print(summary)
    return summary


lolkek = onDatasetData(df_text)
print(type(lolkek))
print(lolkek)
# datasetData = onDatasetData(df_text)

# print(datasetData)

# summary = []
# for item in df_text["body"]:
#     summary.append(predict_pipe(item))
#     # print("Оценка: ", predict_pipe(item), "ID: ", datasetComments["id"])
#     # print("Summary", summary)
#     print(summary, "Summary")
