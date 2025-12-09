import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------
# VK Client: загрузка участников всей группы
# --------------------------
class VKClient:
    """
    Клиент для скачивания участников группы VK по batch'ам.
    Предполагается, что токен валиден и имеет права для groups.getMembers.
    """
    BASE_URL = 'https://api.vk.com/method/'

    def __init__(self, token: str, api_version: str = '5.199', verbose: bool = True):
        self.token = token
        self.v = api_version
        self.verbose = verbose

    def _post(self, method: str, data: dict) -> dict:
        url = self.BASE_URL + method
        resp = requests.post(url, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_group_member_count(self, group_id: str) -> int:
        data = {
            'group_id': group_id,
            'count': 1,
            'v': self.v,
            'access_token': self.token
        }
        resp = self._post('groups.getMembers', data)
        if 'response' not in resp:
            raise RuntimeError(f'VK API error: {resp}')
        return resp['response']['count']

    def get_all_members(self, group_id: str, fields: str = 'online,sex,city,country,universities,bdate', batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Возвращает список словарей — всех участников группы.
        batch_size: <= 1000
        """
        total = self.get_group_member_count(group_id)
        if self.verbose:
            print(f"Всего участников: {total}")
        offset = 0
        all_items = []
        while offset < total:
            data = {
                'group_id': group_id,
                'count': batch_size,
                'offset': offset,
                'fields': fields,
                'v': self.v,
                'access_token': self.token
            }
            resp = self._post('groups.getMembers', data)
            if 'response' not in resp:
                # логируем ошибку и прерываем сбор
                print("Ошибка при получении партии:", resp)
                break
            items = resp['response']['items']
            all_items.extend(items)
            offset += batch_size
            if self.verbose:
                print(f"Загружено {len(all_items)}/{total}")
            # чтобы не превышать лимиты VK, небольшая пауза
            time.sleep(0.34)
        return all_items

# --------------------------
# DataLoader: загрузка/сохранение
# --------------------------
class DataLoader:
    # Загрузчик/сейвер для JSON и преобразование в pandas.DataFrame
    @staticmethod
    def save_json(path: str, data: Any):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> Any:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def to_dataframe(members: List[Dict[str, Any]]) -> pd.DataFrame:
        # Преобразует "сырой" список участников в DataFrame, извлекая общие поля и оставляя оригинал.
        rows = []
        for m in members:
            row = {
                'id': m.get('id'),
                'first_name': m.get('first_name'),
                'last_name': m.get('last_name'),
                'sex': m.get('sex'),  # 1 - female, 2 - male
                'online': m.get('online', 0),
                'is_closed': m.get('is_closed', False),
                'can_access_closed': m.get('can_access_closed', False),
                'city_id': None,
                'city_title': None,
                'country_id': None,
                'country_title': None,
                'universities': None,
                'bdate': m.get('bdate')
            }
            if 'city' in m and isinstance(m['city'], dict):
                row['city_id'] = m['city'].get('id')
                row['city_title'] = m['city'].get('title')
            if 'country' in m and isinstance(m['country'], dict):
                row['country_id'] = m['country'].get('id')
                row['country_title'] = m['country'].get('title')
            if 'universities' in m and isinstance(m['universities'], list):
                # сохраним список университетов как список dict'ов
                row['universities'] = m['universities']
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

# --------------------------
# Preprocessor: очистка / фичеинжиниринг
# --------------------------
class Preprocessor:
    # Методы предобработки: нормализация города, извлечение возраста из bdate, агрегация университетов.
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def _extract_age(bdate: Optional[str], reference_year: Optional[int] = None) -> Optional[int]:
        # bdate может быть 'DD.MM.YYYY' или 'DD.MM'. Если нет года — вернётся None.
        if not isinstance(bdate, str):
            return None
        parts = bdate.split('.')
        if len(parts) == 3:
            try:
                year = int(parts[2])
            except ValueError:
                return None
            if reference_year is None:
                reference_year = pd.Timestamp.now().year
            age = reference_year - year
            if 4 <= age <= 120:
                return age
            return None
        return None

    def add_age(self):
        self.df['age'] = self.df['bdate'].apply(self._extract_age)
        return self

    def unify_city(self):
        # Заменяем пустые/NaN на 'Не указано'
        self.df['city_title'] = self.df['city_title'].fillna('Не указано')
        return self

    def extract_university_features(self):
        # Извлекаем признаки из universities: название, число, выпуск, факультет, статус.
        def extract(unis):
            if not isinstance(unis, list) or len(unis) == 0:
                return {
                    'primary_university': 'Не указано',
                    'n_universities': 0,
                    'graduation_year': None,
                    'faculty_name': 'Не указано',
                    'education_status': 'Не указано'
                }
            u = unis[0]
            return {
                'primary_university': u.get('name') or u.get('faculty_name') or u.get('chair_name') or 'Не указано',
                'n_universities': len(unis),
                'graduation_year': u.get('graduation'),
                'faculty_name': u.get('faculty_name', 'Не указано'),
                'education_status': u.get('education_status', 'Не указано')
            }

        features = self.df['universities'].apply(extract)
        for col in ['primary_university', 'n_universities', 'graduation_year', 'faculty_name', 'education_status']:
            self.df[col] = features.apply(lambda x: x[col])
        return self

    def fill_na_for_analysis(self):
        self.df['sex'] = self.df['sex'].fillna(0).astype(int)
        self.df['online'] = self.df['online'].fillna(0).astype(int)
        self.df['city_title'] = self.df['city_title'].fillna('Не указано')
        self.df['primary_university'] = self.df['primary_university'].fillna('Не указано')
        self.df['faculty_name'] = self.df['faculty_name'].fillna('Не указано')
        self.df['education_status'] = self.df['education_status'].fillna('Не указано')
        return self

    def run_all(self):
        return (self.add_age()
                    .unify_city()
                    .extract_university_features()
                    .fill_na_for_analysis()
                    .df)

# --------------------------
# Analyzer: агрегаты и сводки
# --------------------------
class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def gender_counts(self) -> Dict[str, int]:
        # VK: 1 female, 2 male, 0 unknown
        c = Counter(self.df['sex'])
        return {'female': c.get(1, 0), 'male': c.get(2, 0), 'unknown': c.get(0, 0)}

    def online_count(self) -> int:
        return int(self.df['online'].sum())

    def top_cities(self, n: int = 10) -> List[Tuple[str, int]]:
        c = Counter(self.df['city_title'])
        return c.most_common(n)

    def top_universities(self, n: int = 10) -> List[Tuple[str, int]]:
        c = Counter(self.df['primary_university'])
        return c.most_common(n)

    def age_stats(self) -> Dict[str, float]:
        ages = self.df['age'].dropna()
        if len(ages) == 0:
            return {}
        return {
            'count': int(ages.count()),
            'mean': float(ages.mean()),
            'median': float(ages.median()),
            'std': float(ages.std())
        }

    def summary_table(self) -> pd.DataFrame:
        # Простая сводка по городам и полу
        table = self.df.pivot_table(index='city_title', columns='sex', values='id', aggfunc='count', fill_value=0)
        # Переименуем колонки
        table = table.rename(columns={0: 'unknown', 1: 'female', 2: 'male'})
        table['total'] = table.sum(axis=1)
        table = table.sort_values('total', ascending=False)
        return table

# --------------------------
# Classifier: подготовка признаков, обучение, оценка
# --------------------------
class ClassifierModel:
    # Pipeline для классификации пола на основе расширенных признаков.
    def __init__(self, df: pd.DataFrame, target_col: str = 'sex'):
        self.df = df.copy()
        self.target_col = target_col
        self.pipeline = None

    def prepare_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        features = [
            'city_title',
            'primary_university',
            'faculty_name',
            'education_status',
            'graduation_year',
            'n_universities',
            'online',
            'age'
        ]
        X = self.df[features].copy()
        y = self.df[self.target_col].copy()
        mask = y.isin([1, 2])
        X = X[mask]
        y = y[mask]
        return X, y

    def build_pipeline(self):
        numeric_features = ['age', 'n_universities', 'graduation_year', 'online']
        categorical_features = ['city_title', 'primary_university', 'faculty_name', 'education_status']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        encoder_kwargs = {'handle_unknown': 'ignore'}
        sig = inspect.signature(OneHotEncoder)
        if 'sparse_output' in sig.parameters:
            encoder_kwargs['sparse_output'] = False
        else:
            encoder_kwargs['sparse'] = False

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Не указано')),
            ('ohe', OneHotEncoder(**encoder_kwargs))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])

    def train_and_evaluate(self, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        X, y = self.prepare_X_y()
        if len(X) < 50:
            raise ValueError("Недостаточно данных для обучения (требуется >=50 примеров с меткой пола).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['female(1)', 'male(2)'], zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2])
        return {'accuracy': acc, 'report_text': report, 'confusion_matrix': cm, 'y_test': y_test, 'y_pred': y_pred}

# --------------------------
# Visualizer: графики и сохранение таблиц
# --------------------------
class Visualizer:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_bar(self, items: List[Tuple[str, int]], title: str, filename: str, top_n: int = None):
        if top_n:
            items = items[:top_n]
        labels, counts = zip(*items) if items else ([], [])
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        return path

    def plot_pie_gender(self, gender_counts: Dict[str, int], filename: str):
        labels = list(gender_counts.keys())
        sizes = list(gender_counts.values())
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Gender distribution')
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        return path

    def save_table(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.output_dir, filename)
        # сохраняем в excel для удобного просмотра
        df.to_excel(path, index=True)
        return path

    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], filename: str):
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        return path

# --------------------------
# End-to-end pipeline runner (пример использования)
# --------------------------
def run_pipeline_from_file(json_path: str, save_output: bool = True):
    # 1) Загрузка
    raw = DataLoader.load_json(json_path)
    # raw ожидается: dict с ключом 'response' -> 'items' или список участников
    if isinstance(raw, dict) and 'response' in raw and 'items' in raw['response']:
        members = raw['response']['items']
    elif isinstance(raw, list):
        members = raw
    else:
        raise ValueError("Не распознан формат JSON. Ожидается список или {response: {items: [...]}}.")

    df = DataLoader.to_dataframe(members)

    # 2) Предобработка
    prep = Preprocessor(df)
    df_processed = prep.run_all()

    # 3) Анализ
    analyzer = Analyzer(df_processed)
    gender = analyzer.gender_counts()
    online = analyzer.online_count()
    top_cities = analyzer.top_cities(20)
    top_unis = analyzer.top_universities(20)
    age_stats = analyzer.age_stats()
    summary_table = analyzer.summary_table()

    # 4) Классификация (если есть достаточно меток)
    classifier_results = None
    try:
        clf = ClassifierModel(df_processed)
        classifier_results = clf.train_and_evaluate()
    except Exception as e:
        classifier_results = {'error': str(e)}

    # 5) Визуализации и сохранение
    vis = Visualizer()
    gender_path = vis.plot_pie_gender(gender, 'gender_pie.png')
    cities_path = vis.plot_bar(top_cities, 'Top cities', 'top_cities.png', top_n=20)
    unis_path = vis.plot_bar(top_unis, 'Top universities', 'top_universities.png', top_n=20)
    table_path = vis.save_table(summary_table, 'summary_by_city.xlsx')
    cm_path = None
    if isinstance(classifier_results, dict) and 'confusion_matrix' in classifier_results and not classifier_results.get('error'):
        cm = classifier_results['confusion_matrix']
        cm_path = vis.plot_confusion_matrix(cm, ['female(1)','male(2)'], 'confusion_matrix.png')

    # 6) Output (словари, пути файлов)
    out = {
        'gender_counts': gender,
        'online_count': online,
        'top_cities': top_cities,
        'top_universities': top_unis,
        'age_stats': age_stats,
        'summary_table_path': table_path,
        'gender_chart': gender_path,
        'cities_chart': cities_path,
        'universities_chart': unis_path,
        'classifier': classifier_results,
        'confusion_matrix_chart': cm_path
    }

    # сохранить полный dataframe и сырой members
    if save_output:
        DataLoader.save_json(os.path.join(vis.output_dir, 'members_processed.json'), df_processed.to_dict(orient='records'))
        df_processed.to_excel(os.path.join(vis.output_dir, 'members_processed.xlsx'), index=False)

    return out

# --------------------------
# CLI / пример использования
# --------------------------
if __name__ == '__main__':
    # Пример 1: если уже есть members.json 
    example_json = 'members.json'
    if os.path.exists(example_json):
        print("Запуск pipeline по файлу:", example_json)
        results = run_pipeline_from_file(example_json)
        print("Результаты анализа (кратко):")
        print("Gender:", results['gender_counts'])
        print("Online:", results['online_count'])
        print("Top cities (top5):", results['top_cities'][:5])
        print("Top unis (top5):", results['top_universities'][:5])
        print("Age stats:", results['age_stats'])
        if isinstance(results['classifier'], dict) and results['classifier'].get('error'):
            print("Классификация не выполнена:", results['classifier']['error'])
        else:
            print("Accuracy:", results['classifier']['accuracy'])
            print("Classification report:\n", results['classifier']['report_text'])
    else:
        # Пример 2: загрузка напрямую из VK
        token = 'vk1.a.6pAK9OQR1azUgnvaoS82AKqzgeHUnP0eEC0Z6bb_vneCS2rue85MiSTKch1b2-kowvk7DqNN9FESIHVKXK1IwsHjHcXtQSZczVyub7DQWgQXw5IcTtVoc47xWTHTgjIXd66HrMvsBymbQ7GJA2nMtfY05vFeiMnO2x9ezL5IYhEmFlamu8A4guCRje0BkuZUEtszvxEfy0EHY-xJSbhHjA' # заменить на свой
        group_id = 'ip_telephone'
        if token.startswith('<'):
            print("Файл members.json не найден. Чтобы скачать из VK, укажите реальный token в коде.")
        else:
            vk = VKClient(token, verbose=True)
            members = vk.get_all_members(group_id, fields='online,sex,city,country,universities,bdate')
            DataLoader.save_json('members.json', {'response': {'count': len(members), 'items': members}})
            print("Сохранено members.json. Запустите скрипт снова для анализа.")
