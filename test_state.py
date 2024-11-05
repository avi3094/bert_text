import streamlit as st
from getdataapi import gettedData, writedToFile

st.header("Counter")

if "counter" not in st.session_state:
    st.session_state.counter = 0

if "data" not in st.session_state:
    st.session_state.data = "Загрузите данные"

button = st.button("Increment")
button_data = st.button("Добавить Данные")

if button:
    st.session_state.counter += 1

if button_data:
    st.session_state.data = gettedData
    writedToFile

st.write("Counter = ", st.session_state.counter)
st.write(st.session_state.data)
