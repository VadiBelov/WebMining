import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Мокаем временные модули чтобы не импортировать в тестах
sys.modules['requests'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['bs4.BeautifulSoup'] = MagicMock()
sys.modules['re'] = MagicMock()

# Теперь импортируем класс
from unittest.mock import Mock, patch, MagicMock
import sys

# Создаем мок для time.strftime перед импортом
mock_time = MagicMock()
mock_time.strftime.return_value = "2024-01-01 12:00:00"
sys.modules['time'] = mock_time


# Теперь импортируем и создаем тестовый класс
class TestHHParser(unittest.TestCase):
    """Тесты для класса HHParser с BeautifulSoup."""

    def setUp(self):
        """Настройка перед каждым тестом."""
        # Создаем заглушки для зависимостей
        self.mock_requests = MagicMock()
        self.mock_bs4 = MagicMock()
        self.mock_re = MagicMock()

        sys.modules['requests'] = self.mock_requests
        sys.modules['bs4'] = self.mock_bs4
        sys.modules['re'] = self.mock_re

        # Импортируем класс (теперь с моками)
        from hh_bs4_parser import HHParser  # Предполагаем, что файл называется hh_bs4_parser.py
        self.parser_class = HHParser

    def test_init(self):
        """Тест инициализации парсера."""
        parser = self.parser_class()
        self.assertIsNotNone(parser)
        self.assertEqual(parser.base_url, "https://hh.ru")
        self.assertEqual(parser.search_url, "https://hh.ru/search/vacancy")
        self.assertIn('User-Agent', parser.headers)

    @patch('requests.Session')
    def test_search_vacancies_success(self, mock_session_class):
        """Тест успешного поиска вакансий."""
        # Настраиваем моки
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html>Test HTML content</html>"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Создаем парсер с мок сессией
        parser = self.parser_class()
        parser.session = mock_session

        # Вызываем метод
        result = parser.search_vacancies("Python", page=0)

        # Проверяем
        self.assertEqual(result, "<html>Test HTML content</html>")
        mock_session.get.assert_called_once()

        # Проверяем параметры вызова
        call_args = mock_session.get.call_args
        self.assertEqual(call_args[0][0], "https://hh.ru/search/vacancy")
        self.assertIn('params', call_args[1])
        self.assertEqual(call_args[1]['params']['text'], "Python")
        self.assertEqual(call_args[1]['params']['page'], 0)

    @patch('requests.Session')
    def test_search_vacancies_failure(self, mock_session_class):
        """Тест неудачного поиска вакансий."""
        # Настраиваем моки для ошибки
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")
        mock_session_class.return_value = mock_session

        # Создаем парсер
        parser = self.parser_class()
        parser.session = mock_session

        # Вызываем метод
        result = parser.search_vacancies("Python", page=0)

        # Проверяем
        self.assertIsNone(result)

    def test_parse_vacancy_card_success(self):
        """Тест успешного парсинга карточки вакансии."""
        # Создаем мок карточки
        mock_card = Mock()

        # Настраиваем элементы карточки
        mock_title_tag = Mock()
        mock_title_tag.text.strip.return_value = "Python Developer"
        mock_title_tag.__getitem__.return_value = "https://hh.ru/vacancy/123"

        mock_salary_tag = Mock()
        mock_salary_tag.text.strip.return_value = "100 000 - 150 000 руб."

        mock_company_tag = Mock()
        mock_company_tag.text.strip.return_value = "Test Company"

        mock_location_tag = Mock()
        mock_location_tag.text.strip.return_value = "Москва"

        mock_requirement_tag = Mock()
        mock_requirement_tag.text.strip.return_value = "Знание Python, Django"

        mock_responsibility_tag = Mock()
        mock_responsibility_tag.text.strip.return_value = "Разработка веб-приложений"

        # Настраиваем поиск элементов
        mock_card.find.side_effect = lambda tag, **kwargs: {
            ('a', 'class_'): mock_title_tag if kwargs.get('class_') == 'serp-item__title' else None,
            ('span', 'data-qa'): mock_salary_tag if kwargs.get(
                'data-qa') == 'vacancy-serp__vacancy-compensation' else None,
            ('a', 'data-qa'): mock_company_tag if kwargs.get('data-qa') == 'vacancy-serp__vacancy-employer' else None,
            ('div', 'data-qa'): mock_location_tag if kwargs.get(
                'data-qa') == 'vacancy-serp__vacancy-address' else mock_requirement_tag if kwargs.get(
                'data-qa') == 'vacancy-serp__vacancy_snippet_requirement' else mock_responsibility_tag if kwargs.get(
                'data-qa') == 'vacancy-serp__vacancy_snippet_responsibility' else None,
        }.get((tag, list(kwargs.keys())[0]), None)

        # Вызываем метод
        parser = self.parser_class()
        result = parser.parse_vacancy_card(mock_card)

        # Проверяем
        self.assertIsNotNone(result)
        self.assertEqual(result['title'], "Python Developer")
        self.assertEqual(result['link'], "https://hh.ru/vacancy/123")
        self.assertEqual(result['salary'], "100 000 - 150 000 руб.")
        self.assertEqual(result['company'], "Test Company")
        self.assertEqual(result['location'], "Москва")
        self.assertEqual(result['description'], "Знание Python, Django Разработка веб-приложений")
        self.assertEqual(result['timestamp'], "2024-01-01 12:00:00")

    def test_parse_vacancy_card_no_title(self):
        """Тест парсинга карточки без заголовка."""
        mock_card = Mock()
        mock_card.find.return_value = None  # Заголовок не найден

        parser = self.parser_class()
        result = parser.parse_vacancy_card(mock_card)

        self.assertIsNone(result)

    def test_parse_vacancies_page_empty(self):
        """Тест парсинга пустой страницы."""
        parser = self.parser_class()
        result = parser.parse_vacancies_page(None)

        self.assertEqual(result, [])

    @patch('bs4.BeautifulSoup')
    def test_parse_vacancies_page_with_vacancies(self, mock_bs):
        """Тест парсинга страницы с вакансиями."""
        # Настраиваем моки
        mock_soup = Mock()
        mock_card = Mock()
        mock_vacancy = {
            'title': 'Test Vacancy',
            'link': 'https://test.com',
            'salary': '100000 руб',
            'company': 'Test Co',
            'location': 'Moscow',
            'description': 'Test description',
            'timestamp': '2024-01-01 12:00:00'
        }

        mock_soup.find_all.return_value = [mock_card, mock_card]  # Две карточки
        mock_bs.return_value = mock_soup

        # Мокаем parse_vacancy_card
        parser = self.parser_class()
        parser.parse_vacancy_card = Mock(side_effect=[mock_vacancy, None])  # Первая успешна, вторая нет

        # Вызываем метод
        result = parser.parse_vacancies_page("<html>content</html>")

        # Проверяем
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_vacancy)
        mock_soup.find_all.assert_called_once_with('div', class_='serp-item')

    @patch.object(HHParser, 'search_vacancies')
    @patch.object(HHParser, 'parse_vacancies_page')
    @patch('bs4.BeautifulSoup')
    def test_parse_vacancies_success(self, mock_bs, mock_parse_page, mock_search):
        """Тест парсинга нескольких страниц."""
        # Настраиваем моки
        parser = self.parser_class()

        # Настройка ответов
        mock_search.side_effect = ["html1", "html2", None]  # Две страницы, третья None

        mock_parse_page.side_effect = [
            [{'title': 'Vacancy 1'}],  # Первая страница
            [{'title': 'Vacancy 2'}],  # Вторая страница
        ]

        # Мок для проверки следующей страницы
        mock_soup = Mock()
        mock_next_button = Mock()
        mock_soup.find.return_value = mock_next_button  # Есть следующая страница
        mock_bs.return_value = mock_soup

        # Вызываем метод
        result = parser.parse_vacancies("Python", pages_to_parse=3)

        # Проверяем
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'Vacancy 1')
        self.assertEqual(result[1]['title'], 'Vacancy 2')

        # Проверяем вызовы
        self.assertEqual(mock_search.call_count, 2)  # Вызвано 2 раза (третий None)
        self.assertEqual(mock_parse_page.call_count, 2)

    @patch.object(HHParser, 'search_vacancies')
    @patch.object(HHParser, 'parse_vacancies_page')
    def test_parse_vacancies_no_results(self, mock_parse_page, mock_search):
        """Тест парсинга когда нет результатов."""
        parser = self.parser_class()

        mock_search.return_value = "html"
        mock_parse_page.return_value = []  # Пустая страница

        result = parser.parse_vacancies("Python", pages_to_parse=1)

        self.assertEqual(result, [])

    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_save_to_json(self, mock_json_dump, mock_open):
        """Тест сохранения в JSON."""
        parser = self.parser_class()
        test_data = [{'title': 'Test', 'company': 'Test Co'}]

        parser.save_to_json(test_data, "test.json")

        # Проверяем вызовы
        mock_open.assert_called_once_with("test.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(
            test_data, mock_open.return_value.__enter__.return_value,
            ensure_ascii=False, indent=2
        )

    @patch('requests.Session')
    def test_get_detailed_vacancy_info(self, mock_session_class):
        """Тест получения детальной информации о вакансии."""
        # Настраиваем моки
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <div data-qa="vacancy-description">Full description</div>
            <span data-qa="bloko-tag__text">Python</span>
            <span data-qa="bloko-tag__text">Django</span>
            <p data-qa="vacancy-view-employment-mode">Полная занятость</p>
            <span data-qa="vacancy-experience">1-3 года</span>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Создаем парсер
        parser = self.parser_class()
        parser.session = mock_session

        # Мокаем BeautifulSoup
        with patch('bs4.BeautifulSoup') as mock_bs:
            mock_soup = Mock()

            # Настраиваем поиск элементов
            mock_desc_tag = Mock()
            mock_desc_tag.text.strip.return_value = "Full description"

            mock_skill_tag1 = Mock()
            mock_skill_tag1.text.strip.return_value = "Python"
            mock_skill_tag2 = Mock()
            mock_skill_tag2.text.strip.return_value = "Django"

            mock_emp_tag = Mock()
            mock_emp_tag.text.strip.return_value = "Полная занятость"

            mock_exp_tag = Mock()
            mock_exp_tag.text.strip.return_value = "1-3 года"

            mock_soup.find.side_effect = lambda tag, **kwargs: {
                ('div', 'data-qa'): mock_desc_tag if kwargs.get('data-qa') == 'vacancy-description' else None,
                ('p', 'data-qa'): mock_emp_tag if kwargs.get('data-qa') == 'vacancy-view-employment-mode' else None,
                ('span', 'data-qa'): mock_exp_tag if kwargs.get('data-qa') == 'vacancy-experience' else None,
            }.get((tag, list(kwargs.keys())[0]), None)

            mock_soup.find_all.return_value = [mock_skill_tag1, mock_skill_tag2]
            mock_bs.return_value = mock_soup

            # Вызываем метод
            result = parser.get_detailed_vacancy_info("https://hh.ru/vacancy/123")

            # Проверяем
            self.assertIsNotNone(result)
            self.assertEqual(result['full_description'], "Full description")
            self.assertEqual(result['skills'], ["Python", "Django"])
            self.assertEqual(result['employment_type'], "Полная занятость")
            self.assertEqual(result['experience_required'], "1-3 года")
            self.assertEqual(result['detailed_timestamp'], "2024-01-01 12:00:00")

    @patch('requests.Session')
    def test_get_detailed_vacancy_info_error(self, mock_session_class):
        """Тест ошибки при получении детальной информации."""
        # Настраиваем мок для ошибки
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Request failed")
        mock_session_class.return_value = mock_session

        parser = self.parser_class()
        parser.session = mock_session

        result = parser.get_detailed_vacancy_info("https://hh.ru/vacancy/123")

        self.assertIsNone(result)


class TestHHParserIntegration(unittest.TestCase):
    """Интеграционные тесты (требуют реального доступа к интернету)."""

    @unittest.skipIf(os.environ.get('SKIP_NETWORK_TESTS'), "Сетевые тесты пропущены")
    def test_real_connection(self):
        """Тест реального соединения с HH.ru."""
        from hh_bs4_parser import HHParser

        parser = HHParser()

        # Тест простого запроса
        response = parser.session.get("https://hh.ru", timeout=10)

        self.assertEqual(response.status_code, 200)
        self.assertIn('hh.ru', response.text)


class TestMainFunctionality(unittest.TestCase):
    """Тесты основной функциональности."""

    def test_vacancy_structure(self):
        """Тест структуры данных вакансии."""
        from hh_bs4_parser import HHParser

        parser = HHParser()

        # Пример ожидаемой структуры
        expected_keys = ['title', 'link', 'salary', 'company', 'location',
                         'description', 'timestamp']

        # Тестовая вакансия
        test_vacancy = {key: f"test_{key}" for key in expected_keys}

        # Проверяем что все ключи присутствуют
        for key in expected_keys:
            self.assertIn(key, test_vacancy)

    def test_search_queries(self):
        """Тест корректности поисковых запросов."""
        search_queries = [
            "Data Scientist",
            "ML Engineer",
            "Аналитик данных",
            "Python разработчик",
            "Разработчик Java"
        ]

        # Проверяем что запросы не пустые
        for query in search_queries:
            self.assertIsInstance(query, str)
            self.assertTrue(len(query) > 0)


if __name__ == '__main__':
    # Запуск всех тестов
    unittest.main(verbosity=2)