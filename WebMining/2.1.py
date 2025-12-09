import requests
import json
from collections import Counter

token = 'ad3d8622ad3d8622ad3d86220bae00744daad3dad3d8622c40a4abea2950801c448cea8'
base_url = ' https://api.vk.com/method/'
group_id = 'ip_telephone'

resp = requests.post(
    base_url + 'groups.getMembers',  
    data={
        'group_id': group_id, 
        'count': 1,
        'v': '5.199', 
        'access_token': token
        }
)

data = resp.json()
total_count = data['response']['count']
print("Всего участников:", total_count)

all_members = []
offset = 0
batch_size = 1000  # максимум за один запрос

while offset < total_count:
    resp = requests.post(
        base_url + 'groups.getMembers',
        data={
            'group_id': group_id,
            'lang': 0,
            'count': batch_size,
            'offset': offset,
            'fields': 'online,sex,city,universities',
            'v': '5.199',
            'access_token': token
        }
    )
    data = resp.json()
    if "response" in data:
        all_members.extend(data['response']['items'])
    else:
        print("Ошибка:", data)
        break
    offset += batch_size
    print(f"Загружено {len(all_members)} участников...")

# Подсчёты
# По полу
male_count = sum(1 for m in all_members if m.get('sex') == 2)
female_count = sum(1 for m in all_members if m.get('sex') == 1)
no_count = sum(1 for m in all_members if m.get('sex') == 0)

# Онлайн
online_count = sum(1 for m in all_members if m.get('online') == 1)

# Распределение по городам
cities = Counter(
    m['city']['title'] for m in all_members if 'city' in m
)

# Распределение по индивидуально выбранному полю
universities = Counter()
for m in all_members:
    if 'universities' in m and isinstance(m['universities'], list):
        for uni in m['universities']:
            name = uni.get('name')
            if name:
                universities[name] += 1
    else:
        universities['Не указано'] += 1

# Вывод
print("\nПервые 10 участников:")
print(json.dumps(all_members[:10], ensure_ascii=False, indent=2))
print("\nУчастники женщины:", female_count)
print("Участники мужчины:", male_count)
print("Участники (не указали пол):", no_count)
print("\nОнлайн:", online_count)
print("\nРаспределение по городам:")
for city, count in cities.most_common(10): # топ-10
    print(f"  {city}: {count}")

print("\nРаспределение по университетам:")
for uni, count in universities.most_common(10):  # топ-10
    print(f"  {uni}: {count}")


