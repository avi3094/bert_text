from pathlib import Path  # путь по папки

import streamlit as st  # веб-приложение (интерактивный дашборд)

from PIL import Image  # работа с изображением
import pandas as pd  # работа с данными
import torch

from getdataapi import gettedData, writedToFile
from BertComments import predict_pipe, predict_pipe_custom
from itertools import chain

# from modelBertTextSentAnalysis import pipe

# ====================== главная страница ============================

# параметры главной страницы
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Demo TextAnalysis",
    page_icon="./berticon.jpg",
)

# ----------- функции -------------------------------------


# функция для загрузки картики с диска
# кэшируем иначе каждый раз будет загружатся заново
@st.cache_data
def load_image(image_path):
    image = Image.open(image_path)
    # обрезка до нужного размера с сохранинием пропорций
    MAX_SIZE = (600, 400)
    image.thumbnail(MAX_SIZE)
    return image


# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки
image_path = Path.cwd() / "sentiment_analysis.jpg"
image = load_image(image_path)


# отрисовка картинки на странице
c1, c2 = st.columns(2)

with c2:
    st.image(image)

# ---------- отрисовка текста и картинки ------------------------
with c1:
    st.write("""# Анализ тональности комментариев""")

if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button_data():
    st.session_state.clicked = True


if st.session_state.clicked:
    # The message and nested widget will remain on the page
    gettedData
    writedToFile

# Убрал кнопку, т.к. не могу сделать state - состояние приложения.
# st.button("Получить данные", on_click=click_button_data)

# ====================== боковое меню для ввода данных ===============
st.sidebar.header("Входные данные пользователя:")

st.sidebar.title("Введите комментарий для оценки")
new_comment = st.sidebar.text_input(
    "Введите комментарий на английском",
    key="simple_comment",
)
text = new_comment
# print(text)
# print(predict_pipe(text))
print(predict_pipe_custom(text))
print("Кастомный:", predict_pipe_custom(text))

answer = predict_pipe(text)
answer_custom = predict_pipe_custom(text)

st.sidebar.title("Результат:")
st.sidebar.write("Тональность: ", answer[0]["label"])
st.sidebar.write("Точность оценки: ", answer[0]["score"])

st.sidebar.title("Результат:")
st.sidebar.write("Тональность: ", answer_custom[0]["label"])
st.sidebar.write("Точность оценки: ", answer_custom[0]["score"])


# Обработка json в pandas датафрейм.
df_text = pd.json_normalize(gettedData)

# =========== вывод входных данных и предсказания модели ==========
##
st.write("## Таблица комментариев")
st.write(df_text)

# предикт моделью входных данных, на выходе вероятность МКБ
# mkb_prob = diabet_model.predict_proba(df.values)[0, 1]


# вывести предсказание модели МКБ
st.write("## Оценка тональности комментариев из датасета:")


def onDatasetData(datasetComments):
    summary = []
    # print(datasetComments["body"], "datasetComments[body]")
    for item in datasetComments["body"]:
        summary.append(predict_pipe(item))
        # print("Оценка: ", predict_pipe(item), "ID: ", datasetComments["id"])
        # print("Summary", summary)
        print(summary)
    return summary


pred_datasetData = onDatasetData(df_text)
df_pred_data = pd.DataFrame(list(chain.from_iterable(pred_datasetData)))
print(type(pred_datasetData))


st.button("Анализ тональности данных из датасета")
st.write(df_pred_data)
