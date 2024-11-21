Учебное веб-приложение "Анализ тональности комментариев" по курсу "Анализ данных и машинное обучение" в Университете Иннополис.     
Приложение написано на python, на фреймворке "streamlit" с моделью анализа текста "BERT" с huggingface.    
Файлы проекта и структура:

1. app.py - главный файл проекта. В него импортируются все остальные файлы  и ф-ции.
2. BertComments.py - файл с моделью BERT, дообученной. Брал версию "BERT" ('bert-based-uncased' и дообучал (тюнил) на датасете тоналности комментариев с kaggle).  
3. getdatapi.py - файл, в котором принимаются данные (комментарии, в формате json) с API "dummyJSON" и затем записываются в файл comments.json.
  
Подробнее в презентации:  
https://docs.google.com/presentation/d/1qsLZz-d3QXSaXDnDVDKOoW82_7n5GHD9/edit?usp=sharing&ouid=107388121429638394178&rtpof=true&sd=true

Ссылка на приложение:  
https://bertdemoapp.streamlit.app/  
