from pathlib import Path  # путь по папки
import pickle  # архиватор модели

import streamlit as st  # веб-приложение (интерактивный дашборд)
from PIL import Image  # работа с изображением
import pandas as pd  # работа с данными
from pandas import json_normalize
import catboost

from getdataapi import gettedData, writedToFile

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


# функция загрузки модели
# кэшируем иначе каждый раз будет загружатся заново
@st.cache_data
def load_model(model_path):
    # загрузка сериализованной модели
    with open(model_path, "rb") as f:
        model = pickle.load(f)  # упакованная модель catboost
    return model


# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки
image_path = Path.cwd() / "sentiment_analysis.jpg"
image = load_image(image_path)

# путь до модели
# model_path = Path.cwd() / "model.pkl"
# diabet_model = load_model(model_path)

# путь до модели из примера.
model_bert_path = Path.cwd() / "modelBertTextSentAnalysis.pkl"
text_analysis_model = load_model(model_bert_path)

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
    gettedData
    writedToFile


if st.session_state.clicked:
    # The message and nested widget will remain on the page
    gettedData
    writedToFile

st.button("Получить данные", on_click=click_button_data)

# ====================== боковое меню для ввода данных ===============

placeholder_text = "Eng"

st.sidebar.header("Входные данные пользователя:")

st.sidebar.title("Введите комментарий для оценки")
new_comment = st.sidebar.text_input(
    "Введите комментарий на английском", key="simple_comment", value=""
)
st.write(new_comment)

st.sidebar.title("Введите комменатрий, чтобы добавить его в датасет")
new_comment_to_dataset = st.sidebar.text_input(
    "Введите комментарий на английском", key="simple_comment_to_dataset", value=""
)
st.write(new_comment_to_dataset)

# словарь с названиями признаков и описанием для удобства
# features_dict = {
#     "gravity": "Удельный вес (плотность) мочи",
#     "ph": "Уровень PH",
#     "osmo": "Осмолярность (мосм)",
#     "cond": "Проводимость (мМхо)",
#     "urea": "Коцентрация мочевины (ммоль/л)",
#     "calc": "Концентрация кальция (ммоль/л)",
# }

# кнопки - слайдеры для ввода дынных человека
# gravity = st.sidebar.slider(
#     features_dict["gravity"], min_value=1.005, max_value=1.04, value=0.0, step=0.001
# )
# ph = st.sidebar.slider(
#     features_dict["ph"], min_value=4.76, max_value=7.94, value=100.0, step=0.1
# )
# osmo = st.sidebar.slider(
#     features_dict["osmo"], min_value=187, max_value=1236, value=80, step=1
# )
# cond = st.sidebar.slider(
#     features_dict["cond"], min_value=5.1, max_value=38.0, value=20.0, step=1.0
# )
# urea = st.sidebar.slider(
#     features_dict["urea"], min_value=10, max_value=620, value=300, step=1
# )
# calc = st.sidebar.slider(
#     features_dict["calc"], min_value=0.17, max_value=14.34, value=30.0, step=0.1
# )


# записать входные данные в словарь и в датафрейм
# data = {
#     "gravity": gravity,
#     "ph": ph,
#     "osmo": osmo,
#     "cond": cond,
#     "urea": urea,
#     "calc": calc,
# }
# df = pd.DataFrame(data, index=[0])

df_text = pd.json_normalize(gettedData)


# =========== вывод входных данных и предсказания модели ==========

##
st.write("## Комментарии")
st.write(df_text)

# предикт моделью входных данных, на выходе вероятность МКБ
# mkb_prob = diabet_model.predict_proba(df.values)[0, 1]

text = new_comment
methods = dir(text_analysis_model)
analysis = pipe(text)

print(text_analysis_model)
print(text)

# вывести предсказание модели МКБ
st.write("## Оценка тональности комментария")
st.write(f"{analysis:.2f}")
