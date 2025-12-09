import requests 
import json 
import time 
import os 

def getPage(page = 0): 
    """
    Создаем метод для получения страницы со списком вакансий.
    Аргументы:
        page - Индекс страницы, начинается с 0. Значение по умолчанию 0,
    т.е. первая страница
    """
    
    # Справочник для параметров GET-запроса
    params = {
        'text': 'NAME:Программист',  # Текст фильтра. Отбираем вакансии по слову "Программист"
        'area': 1,  # Поиск осуществляется по вакансиям в г. Москве
        'page': page,  # Индекс страницы поиска на hh.ru
        'per_page': 100  # Кол-во вакансий на 1 странице
    }

    req = requests.get('https://api.hh.ru/vacancies', params)
    data = req.content.decode()
    req.close()
    return data

# Ключевые слова из резюме для фильтрации вакансий по описанию
RESUME_KEYWORDS = [
    'Django', 'PostgreSQL',
    'Django REST', 'Python',
    'HTML5', "базы данных"
]

def contains_resume_keywords(description):
    """Проверяет, содержит ли описание вакансии ключевые слова из резюме"""
    if not description:
        return False
    
    import re
    clean_description = re.sub('<[^<]+?>', ' ', description)
    clean_description = clean_description.lower()
    
    found_keywords = []
    for keyword in RESUME_KEYWORDS:
        if keyword.lower() in clean_description:
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"Найдены ключевые слова: {found_keywords}")
        return True
    return False

# Создаем папки для сохранения данных
if not os.path.isdir("pagination_desc"):
    os.mkdir("pagination_desc")

if not os.path.isdir("vacancies_desc"):
    os.mkdir("vacancies_desc")

# Собираем страницы с вакансиями
for page in range(0, 4):  # Первые 4 страницы

    try:
        # Преобразуем текст ответа запроса в справочник
        jsObj = json.loads(getPage(page))

        # Сохраняем файлы в папку pagination
        nextFileName = './pagination_desc/page_{}.json'.format(page)

        # Создаем новый документ, записываем в него ответ запроса, после закрываем
        with open(nextFileName, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsObj, ensure_ascii=False))

        print(f'Сохранена страница {page}')

        # Проверка на последнюю страницу
        if (jsObj['pages'] - page) <= 1:
            print(f'Достигнута последняя страница. Всего страниц: {jsObj["pages"]}')
            break

        # Задержка, чтобы не нагружать сервисы hh.ru
        time.sleep(0.2)
        
    except Exception as e:
        print(f"Ошибка при обработке страницы {page}: {e}")
        break

print('Страницы поиска собраны. Далее получаем список вакансий...')

# Счетчики для статистики
total_vacancies = 0
matched_vacancies = 0

# Собираем детальную информацию по вакансиям
for fl in os.listdir('./pagination_desc'):
    try:
        # Открываем файл, читаем его содержимое
        with open('./pagination_desc/{}'.format(fl), encoding='utf-8') as f:
            jsonText = f.read()

        # Преобразуем полученный текст в объект справочника
        jsonObj = json.loads(jsonText)

        # Получаем и проходимся по непосредственно списку вакансий
        for v in jsonObj['items']:
            total_vacancies += 1
            
            # Обращаемся к API и получаем детальную информацию по конкретной вакансии
            req = requests.get(v['url'])
            data = req.content.decode()
            req.close()

            # Преобразуем данные вакансии в JSON
            vacancy_data = json.loads(data)
            
            # Проверяем, содержит ли описание вакансии ключевые слова из резюме
            description = vacancy_data.get('description', '')
            
            if contains_resume_keywords(description):
                matched_vacancies += 1
                # Сохраняем только те вакансии, которые соответствуют критериям
                fileName = './vacancies_desc/{}.json'.format(v['id'])
                with open(fileName, 'w', encoding='utf-8') as f:
                    f.write(data)
                print(f'Сохранена подходящая вакансия: {v["name"]}')

            time.sleep(0.25)
            
    except Exception as e:
        print(f"Ошибка при обработке файла {fl}: {e}")
        continue

print(f'Сбор вакансий завершен!')
print(f'Всего обработано вакансий: {total_vacancies}')
print(f'Сохранено подходящих вакансий: {matched_vacancies}')
