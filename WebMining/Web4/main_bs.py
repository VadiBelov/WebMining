import requests
import time
import json
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import re


class HHParser:
    """
    Парсер для сайта HeadHunter для поиска вакансий, связанных с программированием.
    Использует BeautifulSoup для парсинга HTML.
    """

    def __init__(self):
        self.base_url = "https://hh.ru"
        self.search_url = "https://hh.ru/search/vacancy"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search_vacancies(self, text: str, area: int = 1, page: int = 0) -> Optional[str]:
        """
        Поиск вакансий по текстовому запросу.

        Args:
            text (str): Текст для поиска (напр., "Python разработчик").
            area (int): ID региона (1 - Москва, 2 - СПб, 113 - Россия).
            page (int): Номер страницы (начинается с 0).

        Returns:
            str: HTML-код страницы или None в случае ошибки.
        """
        params = {
            'text': text,
            'area': area,
            'page': page,
            'items_on_page': 50,
            'hhtmFrom': 'vacancy_search_list'
        }

        try:
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    def parse_vacancy_card(self, card) -> Optional[Dict[str, Any]]:
        """
        Парсит информацию о вакансии из карточки.

        Args:
            card: BeautifulSoup объект карточки вакансии.

        Returns:
            Dict[str, Any]: Словарь с информацией о вакансии или None.
        """
        try:
            # Находим заголовок и ссылку
            title_tag = card.find('a', class_='serp-item__title')
            if not title_tag:
                return None

            title = title_tag.text.strip()
            link = title_tag['href']

            # Находим информацию о зарплате
            salary_tag = card.find('span', {'data-qa': 'vacancy-serp__vacancy-compensation'})
            salary = salary_tag.text.strip() if salary_tag else "Зарплата не указана"

            # Находим название компании
            company_tag = card.find('a', {'data-qa': 'vacancy-serp__vacancy-employer'})
            company = company_tag.text.strip() if company_tag else "Компания не указана"

            # Находим местоположение
            location_tag = card.find('div', {'data-qa': 'vacancy-serp__vacancy-address'})
            location = location_tag.text.strip() if location_tag else "Местоположение не указано"

            # Находим требования/описание
            requirement_tag = card.find('div', {'data-qa': 'vacancy-serp__vacancy_snippet_requirement'})
            requirement = requirement_tag.text.strip() if requirement_tag else ""

            # Находим ответственность
            responsibility_tag = card.find('div', {'data-qa': 'vacancy-serp__vacancy_snippet_responsibility'})
            responsibility = responsibility_tag.text.strip() if responsibility_tag else ""

            # Собираем полное описание
            description = f"{requirement} {responsibility}".strip()

            vacancy_data = {
                'title': title,
                'link': link,
                'salary': salary,
                'company': company,
                'location': location,
                'description': description,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            return vacancy_data

        except Exception as e:
            print(f"Ошибка при парсинге карточки вакансии: {e}")
            return None

    def parse_vacancies_page(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Парсит все вакансии на странице.

        Args:
            html_content (str): HTML-код страницы с вакансиями.

        Returns:
            List[Dict[str, Any]]: Список спарсенных вакансий.
        """
        vacancies = []

        if not html_content:
            return vacancies

        soup = BeautifulSoup(html_content, 'html.parser')

        # Находим все карточки вакансий
        vacancy_cards = soup.find_all('div', class_='serp-item')

        print(f"Найдено карточек вакансий на странице: {len(vacancy_cards)}")

        for card in vacancy_cards:
            vacancy = self.parse_vacancy_card(card)
            if vacancy:
                vacancies.append(vacancy)

        return vacancies

    def parse_vacancies(self, search_query: str, pages_to_parse: int = 5) -> List[Dict[str, Any]]:
        """
        Парсит несколько страниц с вакансиями.

        Args:
            search_query (str): Запрос для поиска.
            pages_to_parse (int): Количество страниц для парсинга.

        Returns:
            List[Dict[str, Any]]: Список спарсенных вакансий.
        """
        all_vacancies = []

        for page in range(pages_to_parse):
            print(f"Парсинг страницы {page + 1}...")

            html_content = self.search_vacancies(search_query, page=page)

            if html_content is None:
                print(f"Не удалось получить данные для страницы {page}. Прерывание.")
                break

            vacancies = self.parse_vacancies_page(html_content)

            if not vacancies:
                print("Больше вакансий нет. Прерывание.")
                break

            all_vacancies.extend(vacancies)
            print(f"На странице {page + 1} найдено вакансий: {len(vacancies)}")

            # Проверяем наличие следующей страницы
            soup = BeautifulSoup(html_content, 'html.parser')
            next_button = soup.find('a', {'data-qa': 'pager-next'})

            if not next_button:
                print("Больше страниц нет.")
                break

            # Уважаем сайт и добавляем задержку
            time.sleep(2)

        print(f"Всего спарсено вакансий: {len(all_vacancies)}")
        return all_vacancies

    def get_detailed_vacancy_info(self, vacancy_url: str) -> Optional[Dict[str, Any]]:
        """
        Получает подробную информацию о вакансии.

        Args:
            vacancy_url (str): URL вакансии.

        Returns:
            Dict[str, Any]: Подробная информация о вакансии.
        """
        try:
            response = self.session.get(vacancy_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Получаем полное описание
            description_tag = soup.find('div', {'data-qa': 'vacancy-description'})
            full_description = description_tag.text.strip() if description_tag else ""

            # Получаем ключевые навыки
            skills_tags = soup.find_all('span', {'data-qa': 'bloko-tag__text'})
            skills = [tag.text.strip() for tag in skills_tags]

            # Получаем тип занятости
            employment_tag = soup.find('p', {'data-qa': 'vacancy-view-employment-mode'})
            employment = employment_tag.text.strip() if employment_tag else ""

            # Получаем опыт работы
            experience_tag = soup.find('span', {'data-qa': 'vacancy-experience'})
            experience = experience_tag.text.strip() if experience_tag else ""

            detailed_info = {
                'full_description': full_description,
                'skills': skills,
                'employment_type': employment,
                'experience_required': experience,
                'detailed_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            return detailed_info

        except Exception as e:
            print(f"Ошибка при получении подробной информации: {e}")
            return None

    def save_to_json(self, data: List[Dict[str, Any]], filename: str = "hh_vacancies.json"):
        """Сохраняет данные в JSON файл."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в {filename}")


if __name__ == "__main__":
    parser = HHParser()

    search_queries = [
        "Data Scientist",
        "ML Engineer",
        "Аналитик данных",
        "Python разработчик",
        "Разработчик Java"
    ]

    all_vacancies_data = []

    for query in search_queries:
        print(f"\n{'=' * 60}")
        print(f"Поиск вакансий для: '{query}'")
        print('=' * 60)

        vacancies = parser.parse_vacancies(search_query=query, pages_to_parse=2)

        # Получаем дополнительную информацию для первых 3 вакансий
        for i, vacancy in enumerate(vacancies[:3]):
            print(f"Получение подробной информации для вакансии {i + 1}: {vacancy['title']}")
            detailed_info = parser.get_detailed_vacancy_info(vacancy['link'])
            if detailed_info:
                vacancy.update(detailed_info)
            time.sleep(1)

        all_vacancies_data.extend(vacancies)

        print(f"Найдено вакансий для '{query}': {len(vacancies)}")

    parser.save_to_json(all_vacancies_data, "hh_programming_vacancies.json")

    # Статистика
    print(f"\n{'=' * 60}")
    print("СТАТИСТИКА СБОРА ДАННЫХ")
    print('=' * 60)
    print(f"Всего собрано вакансий: {len(all_vacancies_data)}")

    # Группировка по запросам
    from collections import Counter

    query_counts = Counter()
    for vacancy in all_vacancies_data:
        for query in search_queries:
            if query.lower() in vacancy['title'].lower() or query.lower() in vacancy['description'].lower():
                query_counts[query] += 1
                break

    print("\nРаспределение по запросам:")
    for query, count in query_counts.items():
        print(f"  {query}: {count} вакансий")
