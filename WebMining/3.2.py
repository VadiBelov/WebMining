import sys
import subprocess
import os

# Проверяем наличие файла ee.html
print("Текущая директория:", os.getcwd())
print("Проверка файла ee.html...")

if os.path.exists('ee.html'):
    print("✓ Файл ee.html найден")
else:
    print("✗ Файл ee.html не найден")
    print("Доступные файлы:", os.listdir('.'))


# Функция для установки необходимых библиотек
def install_packages():
    required_packages = [
        'beautifulsoup4',
        'lxml',
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]

    for package in required_packages:
        try:
            if '-' in package:
                import_name = package.replace('-', '')
            else:
                import_name = package
            __import__(import_name)
            print(f"✓ {package} уже установлен")
        except ImportError:
            print(f"⏳ Устанавливаю {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} установлен")
            except Exception as e:
                print(f"✗ Ошибка при установке {package}: {e}")


# Устанавливаем необходимые библиотеки
print("\nПроверка и установка необходимых библиотек...")
install_packages()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re


# Функция для извлечения навыков из HTML-резюме Django разработчика
def extract_skills_from_django_resume(html_file='ee.html'):
    """Извлекает навыки из HTML резюме Django-разработчика"""

    if not os.path.exists(html_file):
        print(f"Ошибка: Файл {html_file} не найден!")
        print(f"Текущая директория: {os.getcwd()}")
        print(f"Файлы в директории: {os.listdir('.')}")
        return {'soft_skills': [], 'hard_skills': []}

    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Файл {html_file} успешно загружен ({len(content)} символов)")

        skills = {'soft_skills': [], 'hard_skills': []}

        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # 1. Извлекаем личные качества
        all_sections = soup.find_all('section', {'class': 'skills-section'})
        print(f"Найдено секций: {len(all_sections)}")

        for i, section in enumerate(all_sections):
            section_text = section.get_text(strip=True)[:100]
            print(f"Секция {i}: {section_text}...")

            # Проверяем содержание секции
            section_content = section.get_text()

            # Личные качества
            if 'Личные качества' in section_content:
                print("Найдена секция 'Личные качества'")
                qualities_items = section.find_all('div', {'class': 'quality-item'})
                print(f"Найдено качеств: {len(qualities_items)}")
                for item in qualities_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 5:
                        skills['soft_skills'].append(text)

            # Увлечения
            elif 'Увлечения' in section_content:
                print("Найдена секция 'Увлечения'")
                hobbies_items = section.find_all('div', {'class': 'hobby-item'})
                print(f"Найдено увлечений: {len(hobbies_items)}")
                for item in hobbies_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 5:
                        # Убираем 'Стендовый моделизм - ' и подобные префиксы
                        if ' - ' in text:
                            skill_name = text.split(' - ')[0]
                            skills['soft_skills'].append(skill_name)
                        else:
                            skills['soft_skills'].append(text)

            # Языки
            elif 'Владение языками' in section_content:
                print("Найдена секция 'Владение языками'")
                language_items = section.find_all('div', {'class': 'language-item'})
                print(f"Найдено языков: {len(language_items)}")
                for item in language_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 5:
                        skills['hard_skills'].append(text)

            # Профессиональные навыки
            elif 'Профессиональная специализация' in section_content:
                print("Найдена секция 'Профессиональная специализация'")
                skill_categories = section.find_all('div', {'class': 'skill-category'})
                print(f"Найдено категорий навыков: {len(skill_categories)}")
                for category in skill_categories:
                    # Извлекаем название категории
                    category_title = category.find('h3')
                    if category_title:
                        category_text = category_title.get_text(strip=True)
                        if category_text and len(category_text) > 2:
                            skills['hard_skills'].append(category_text)

                    # Извлекаем отдельные навыки
                    skill_items = category.find_all('div', {'class': 'skill-item'})
                    for item in skill_items:
                        text = item.get_text(strip=True)
                        if text and len(text) > 5:
                            # Извлекаем текст после иконки
                            text_parts = text.split(' ', 1)
                            if len(text_parts) > 1:
                                skills['hard_skills'].append(text_parts[1])
                            else:
                                skills['hard_skills'].append(text)

            # Курсы и образование
            elif 'Образование и курсы' in section_content or 'Курсы' in section_content:
                print("Найдена секция 'Образование и курсы'")
                course_items = section.find_all('div', {'class': 'course-item'})
                print(f"Найдено курсов: {len(course_items)}")
                for item in course_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 10:
                        # Извлекаем название курса (первая строка)
                        lines = text.split('\n')
                        if lines:
                            course_name = lines[0].strip()
                            if course_name:
                                skills['hard_skills'].append(course_name)

        # 6. Извлекаем информацию из заголовка
        header = soup.find('header')
        if header:
            subtitle = header.find('p', {'class': 'subtitle'})
            if subtitle:
                subtitle_text = subtitle.get_text(strip=True)
                if subtitle_text:
                    skills['hard_skills'].append(subtitle_text)

            tagline = header.find('p', {'class': 'tagline'})
            if tagline:
                tagline_text = tagline.get_text(strip=True)
                if tagline_text:
                    skills['soft_skills'].append(tagline_text)

        # Очистка и фильтрация навыков
        def clean_skill(skill):
            # Удаляем лишние пробелы и символы
            skill = skill.strip()
            # Удаляем смайлики и специальные символы, но сохраняем буквы, цифры, пробелы и основные знаки препинания
            skill = re.sub(r'[^\w\s\-.,()/:]', ' ', skill, flags=re.UNICODE)
            # Убираем лишние пробелы
            skill = re.sub(r'\s+', ' ', skill)
            return skill

        skills['soft_skills'] = [clean_skill(skill) for skill in skills['soft_skills']
                                 if skill and len(skill) > 2]
        skills['hard_skills'] = [clean_skill(skill) for skill in skills['hard_skills']
                                 if skill and len(skill) > 2]

        # Убираем дубликаты
        skills['soft_skills'] = list(set(skills['soft_skills']))
        skills['hard_skills'] = list(set(skills['hard_skills']))

        # Фильтруем слишком короткие навыки
        skills['soft_skills'] = [skill for skill in skills['soft_skills']
                                 if len(skill.split()) > 0]
        skills['hard_skills'] = [skill for skill in skills['hard_skills']
                                 if len(skill.split()) > 0]

        print(f"\nИзвлечение завершено:")
        print(f"  Soft Skills: {len(skills['soft_skills'])} навыков")
        print(f"  Hard Skills: {len(skills['hard_skills'])} навыков")

        return skills

    except Exception as e:
        print(f"Ошибка при обработке файла {html_file}: {e}")
        import traceback
        traceback.print_exc()
        return {'soft_skills': [], 'hard_skills': []}


# Функция для расчета семантического сходства
def calculate_semantic_similarity(text1, text2):
    """Вычисляет семантическое сходство между двумя текстами"""
    if not text1 or not text2:
        return 0.0

    # Создаем TF-IDF векторайзер с русскими стоп-словами
    russian_stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со',
                          'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да',
                          'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
                          'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
                          'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'ли',
                          'или', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять',
                          'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
                          'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для',
                          'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
                          'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
                          'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
                          'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
                          'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда',
                          'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
                          'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше',
                          'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
                          'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо',
                          'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том',
                          'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно',
                          'всю', 'между']

    vectorizer = TfidfVectorizer(stop_words=russian_stop_words, min_df=1)

    try:
        # Преобразуем тексты в TF-IDF векторы
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Вычисляем косинусное сходство
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
    except Exception as e:
        print(f"Ошибка при вычислении сходства: {e}")
        return 0.0


# Основная функция для анализа сходства
def analyze_vacancies_similarity(vacancies_folder='vacancies_desc', resume_skills=None):
    """Анализирует сходство резюме с вакансиями"""

    if resume_skills is None:
        resume_skills = {'soft_skills': [], 'hard_skills': []}

    results = []

    # Создаем текстовые представления навыков
    soft_skills_text = ' '.join(resume_skills['soft_skills'])
    hard_skills_text = ' '.join(resume_skills['hard_skills'])
    all_skills_text = soft_skills_text + ' ' + hard_skills_text

    print(f"\nТекст для анализа:")
    print(f"Soft skills: {soft_skills_text[:200]}..." if len(
        soft_skills_text) > 200 else f"Soft skills: {soft_skills_text}")
    print(f"Hard skills: {hard_skills_text[:200]}..." if len(
        hard_skills_text) > 200 else f"Hard skills: {hard_skills_text}")

    # Проверяем существование папки с вакансиями
    if not os.path.isdir(vacancies_folder):
        print(f"\nОшибка: Папка '{vacancies_folder}' не найдена!")
        print(f"Текущая директория: {os.getcwd()}")
        print(f"Доступные папки: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        return pd.DataFrame(results)

    # Получаем список файлов вакансий
    vacancy_files = [f for f in os.listdir(vacancies_folder) if f.endswith('.json')]
    print(f"\nНайдено файлов вакансий: {len(vacancy_files)}")

    # Обрабатываем каждую вакансию
    for i, vacancy_file in enumerate(vacancy_files[:50]):  # Ограничиваем для теста
        try:
            with open(os.path.join(vacancies_folder, vacancy_file), 'r', encoding='utf-8') as f:
                vacancy_data = json.load(f)

            vacancy_description = vacancy_data.get('description', '')
            vacancy_name = vacancy_data.get('name', '')
            vacancy_id = vacancy_data.get('id', '')

            if vacancy_description:
                # Очищаем описание от HTML тегов
                soup = BeautifulSoup(vacancy_description, 'html.parser')
                clean_description = soup.get_text()

                # Вычисляем сходство
                soft_similarity = calculate_semantic_similarity(clean_description, soft_skills_text)
                hard_similarity = calculate_semantic_similarity(clean_description, hard_skills_text)
                all_similarity = calculate_semantic_similarity(clean_description, all_skills_text)

                # Рассчитываем общее сходство как среднее
                overall_similarity = (soft_similarity + hard_similarity + all_similarity) / 3

                results.append({
                    'vacancy_id': vacancy_id,
                    'vacancy_name': vacancy_name,
                    'soft_similarity': soft_similarity,
                    'hard_similarity': hard_similarity,
                    'all_similarity': all_similarity,
                    'overall_similarity': overall_similarity,
                    'description_length': len(clean_description),
                    'file_name': vacancy_file
                })

                if (i + 1) % 10 == 0:
                    print(f"Обработано {i + 1} вакансий...")

        except Exception as e:
            print(f"Ошибка при обработке вакансии {vacancy_file}: {e}")
            continue

    print(f"\nОбработка завершена. Всего обработано: {len(results)} вакансий")
    return pd.DataFrame(results)


# Функция для визуализации результатов
def visualize_results(df, resume_skills):
    """Визуализирует результаты анализа"""

    if df.empty:
        print("Нет данных для визуализации")
        return

    # Создаем графики
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Распределение сходства
    axes[0, 0].hist(df['soft_similarity'], bins=20, alpha=0.7, label='Soft Skills', color='skyblue')
    axes[0, 0].hist(df['hard_similarity'], bins=20, alpha=0.7, label='Hard Skills', color='lightcoral')
    axes[0, 0].hist(df['overall_similarity'], bins=20, alpha=0.5, label='Overall', color='lightgreen')
    axes[0, 0].set_title('Распределение семантического сходства', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Сходство (%)', fontsize=10)
    axes[0, 0].set_ylabel('Количество вакансий', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Сравнение soft vs hard skills
    axes[0, 1].scatter(df['soft_similarity'], df['hard_similarity'],
                       alpha=0.6, c=df['overall_similarity'], cmap='viridis', s=50)
    axes[0, 1].set_title('Soft Skills vs Hard Skills сходство', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Soft Skills сходство (%)', fontsize=10)
    axes[0, 1].set_ylabel('Hard Skills сходство (%)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Топ вакансий по общему сходству
    top_vacancies = df.nlargest(10, 'overall_similarity')
    bars = axes[0, 2].barh(range(len(top_vacancies)), top_vacancies['overall_similarity'],
                           color=plt.cm.Set3(range(len(top_vacancies))))
    axes[0, 2].set_yticks(range(len(top_vacancies)))
    axes[0, 2].set_yticklabels([f"{name[:25]}..." if len(name) > 25 else name
                                for name in top_vacancies['vacancy_name']], fontsize=8)
    axes[0, 2].set_title('Топ-10 вакансий по сходству', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Сходство (%)', fontsize=10)
    axes[0, 2].invert_yaxis()  # Самый высокий сверху
    axes[0, 2].grid(True, alpha=0.3, axis='x')

    # 4. Зависимость сходства от длины описания
    scatter = axes[1, 0].scatter(df['description_length'], df['overall_similarity'],
                                 c=df['all_similarity'], cmap='plasma', alpha=0.7, s=50)
    axes[1, 0].set_title('Зависимость сходства от длины описания', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Длина описания (символы)', fontsize=10)
    axes[1, 0].set_ylabel('Общее сходство (%)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='All Skills Similarity')

    # 5. Boxplot сходства по типам навыков
    similarity_data = [df['soft_similarity'], df['hard_similarity'], df['overall_similarity']]
    bp = axes[1, 1].boxplot(similarity_data, labels=['Soft Skills', 'Hard Skills', 'Overall'],
                            patch_artist=True)
    # Цвета для boxplot
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_title('Распределение сходства по типам навыков', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Сходство (%)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Heatmap корреляции
    corr_matrix = df[['soft_similarity', 'hard_similarity', 'overall_similarity', 'description_length']].corr()
    sns.heatmap(corr_matrix, annot=True, ax=axes[1, 2], cmap='coolwarm',
                center=0, fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
    axes[1, 2].set_title('Матрица корреляции', fontsize=12, fontweight='bold')

    plt.suptitle('Анализ семантического сходства для Django-разработчика',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Сохраняем график
    output_file = 'django_semantic_similarity_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен как: {output_file}")
    plt.show()

    # Вывод статистики
    print("\n" + "=" * 60)
    print("СТАТИСТИКА АНАЛИЗА СЕМАНТИЧЕСКОГО СХОДСТВА")
    print("=" * 60)
    print(f"Всего проанализировано вакансий: {len(df)}")

    print(f"\nSoft Skills из резюме Django-разработчика ({len(resume_skills['soft_skills'])}):")
    for i, skill in enumerate(resume_skills['soft_skills'][:10], 1):
        print(f"  {i}. {skill}")
    if len(resume_skills['soft_skills']) > 10:
        print(f"  ... и еще {len(resume_skills['soft_skills']) - 10} навыков")

    print(f"\nHard Skills из резюме Django-разработчика ({len(resume_skills['hard_skills'])}):")
    for i, skill in enumerate(resume_skills['hard_skills'][:10], 1):
        print(f"  {i}. {skill}")
    if len(resume_skills['hard_skills']) > 10:
        print(f"  ... и еще {len(resume_skills['hard_skills']) - 10} навыков")

    print("\n" + "-" * 60)
    print("Статистика сходства (%):")
    print("-" * 60)
    print(f"Soft Skills:  Среднее = {df['soft_similarity'].mean():.2f}%, "
          f"Макс = {df['soft_similarity'].max():.2f}%, "
          f"Мин = {df['soft_similarity'].min():.2f}%")
    print(f"Hard Skills:  Среднее = {df['hard_similarity'].mean():.2f}%, "
          f"Макс = {df['hard_similarity'].max():.2f}%, "
          f"Мин = {df['hard_similarity'].min():.2f}%")
    print(f"Overall:      Среднее = {df['overall_similarity'].mean():.2f}%, "
          f"Макс = {df['overall_similarity'].max():.2f}%, "
          f"Мин = {df['overall_similarity'].min():.2f}%")

    # Топ-5 наиболее подходящих вакансий
    print("\n" + "=" * 60)
    print("ТОП-5 НАИБОЛЕЕ ПОДХОДЯЩИХ ВАКАНСИЙ")
    print("=" * 60)
    top_5 = df.nlargest(5, 'overall_similarity')
    for idx, row in top_5.iterrows():
        print(f"\n{row['vacancy_name']}")
        print(f"ID: {row['vacancy_id']}")
        print(f"Файл: {row['file_name']}")
        print(f"Общее сходство: {row['overall_similarity']:.2f}%")
        print(f"Soft Skills: {row['soft_similarity']:.2f}%")
        print(f"Hard Skills: {row['hard_similarity']:.2f}%")
        print(f"Длина описания: {row['description_length']} символов")


if __name__ == "__main__":
    print("=" * 60)
    print("АНАЛИЗ СЕМАНТИЧЕСКОГО СХОДСТВА ДЛЯ DJANGO-РАЗРАБОТЧИКА")
    print("=" * 60)

    # Извлекаем навыки из резюме Django разработчика
    print("\n1. Извлечение навыков из резюме Django-разработчика...")
    resume_skills = extract_skills_from_django_resume('ee.html')

    if not resume_skills['soft_skills'] and not resume_skills['hard_skills']:
        print("Не удалось извлечь навыки. Завершаем работу.")
    else:
        print(
            f"\nИзвлечено {len(resume_skills['soft_skills'])} soft skills и {len(resume_skills['hard_skills'])} hard skills")

        # Анализируем сходство вакансий
        print("\n2. Анализ семантического сходства с вакансиями...")
        results_df = analyze_vacancies_similarity('vacancies_desc', resume_skills)

        if not results_df.empty:
            # Сохраняем результаты
            output_excel = 'django_semantic_similarity_results.xlsx'
            results_df.to_excel(output_excel, index=False)
            print(f"\nРезультаты сохранены в файл: {output_excel}")

            # Сортируем по сходству
            results_df = results_df.sort_values('overall_similarity', ascending=False)

            # Сохраняем топ вакансий
            top_output_excel = 'django_top_similar_vacancies.xlsx'
            top_results = results_df.head(20)
            top_results.to_excel(top_output_excel, index=False)
            print(f"Топ-20 вакансий сохранены в файл: {top_output_excel}")

            # Визуализация
            print("\n3. Создание визуализации...")
            visualize_results(results_df, resume_skills)

            print("\n" + "=" * 60)
            print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
            print("=" * 60)
            print(f"\nСозданные файлы:")
            print(f"1. {output_excel} - полные результаты анализа")
            print(f"2. {top_output_excel} - топ-20 наиболее подходящих вакансий")
            print(f"3. django_semantic_similarity_analysis.png - графики анализа")
        else:
            print("\nНе удалось проанализировать вакансии. Проверьте папку 'vacancies_desc'.")