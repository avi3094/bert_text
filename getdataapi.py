import requests, pickle, sys

###Получение данных с API dummyjson.
URL_C = "https://dummyjson.com/comments"


# функция получения json комментариев
def getData(url):
    r = requests.get(url)
    if r.status_code == 200:
        r.status_code
        data_json = r.json()
        dataAPI = data_json["comments"]
    else:
        print("Error! Data not Found. Status Code:", r.status_code)
    return dataAPI


# Поскольку, я так и не смог сделать видимой переменную dataAPI из области функции в глобальную, запишу её так в переменную.
gettedData = getData(URL_C)
print("Полученные данные:", gettedData)


# функция записи в json файл
def writeToFile(data):
    file = open("comments.json", "w")
    file.write(str(data))
    file.close()


writedToFile = writeToFile(gettedData)
