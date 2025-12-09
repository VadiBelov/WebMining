using System.Text.Json;
using PuppeteerSharp;

class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== ПАРСЕР ВАКАНСИЙ HH.RU ===\n");
        
        try
        {
            // Укажите путь к установленному Chrome (если есть)
            var chromePath = @"C:\Program Files\Google\Chrome\Application\chrome.exe";
            
            // 1. Настройка браузера
            Console.WriteLine("[1/4] Настройка браузера...");
            var launchOptions = new LaunchOptions
            {
                Headless = true,
                Args = new[] 
                { 
                    "--no-sandbox", 
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-web-security" // Отключаем CORS для теста
                },
                IgnoredDefaultArgs = new[] { "--disable-extensions" },
                Timeout = 60000, // 60 секунд на запуск браузера
                // Используем существующий Chrome, если есть
                ExecutablePath = File.Exists(chromePath) ? chromePath : null
            };
            
            // 2. Запускаем браузер
            Console.WriteLine("[2/4] Запуск браузера...");
            using var browser = await Puppeteer.LaunchAsync(launchOptions);
            
            // 3. Создаем страницу
            Console.WriteLine("[3/4] Создание страницы...");
            using var page = await browser.NewPageAsync();
            
            // Устанавливаем User-Agent и таймауты
            await page.SetUserAgentAsync("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            page.DefaultTimeout = 60000; // 60 секунд таймаут по умолчанию
            
            // 4. Переходим на упрощенную версию или альтернативный URL
            Console.WriteLine("[4/4] Загрузка страницы...");
            
            // Пробуем несколько вариантов URL
            string[] urlsToTry = {
                "https://hh.ru", // Главная страница
                "https://m.hh.ru", // Мобильная версия (быстрее)
                "https://hh.ru/search/vacancy?text=C%23", // Без сложных параметров
                "https://google.com" // Для теста подключения
            };
            
            bool loaded = false;
            foreach (var url in urlsToTry)
            {
                try
                {
                    Console.WriteLine($"Пробуем: {url}");
                    await page.GoToAsync(url, new NavigationOptions 
                    { 
                        WaitUntil = new[] { WaitUntilNavigation.DOMContentLoaded },
                        Timeout = 30000 
                    });
                    loaded = true;
                    Console.WriteLine($"✓ Успешно загружено: {url}");
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"✗ Ошибка загрузки {url}: {ex.Message}");
                }
            }
            
            if (!loaded)
            {
                Console.WriteLine("❌ Не удалось загрузить ни одну страницу");
                return;
            }
            
            // 5. Для HH.ru пытаемся найти вакансии
            if (page.Url.Contains("hh.ru"))
            {
                try
                {
                    Console.WriteLine("\nПоиск вакансий...");
                    
                    // Если на главной странице, вводим поиск
                    if (page.Url == "https://hh.ru/" || page.Url == "https://hh.ru")
                    {
                        // Ждем поле поиска
                        await page.WaitForSelectorAsync("[data-qa='search-input']", new WaitForSelectorOptions { Timeout = 10000 });
                        
                        // Вводим запрос
                        await page.TypeAsync("[data-qa='search-input']", "C# разработчик", new TypeOptions { Delay = 100 });
                        
                        // Нажимаем кнопку поиска
                        await page.ClickAsync("[data-qa='search-button']");
                        
                        // Ждем загрузки результатов
                        await page.WaitForSelectorAsync(".vacancy-serp-item", new WaitForSelectorOptions { Timeout = 15000 });
                    }
                    
                    // Ждем немного для полной загрузки
                    await Task.Delay(3000);
                    
                    // 6. Извлекаем данные
                    var vacanciesData = await page.EvaluateFunctionAsync<string>(@"
                        () => {
                            try {
                                const vacancies = [];
                                
                                // Пробуем несколько селекторов
                                const selectors = [
                                    '[data-qa=""vacancy-serp__vacancy""]',
                                    '.vacancy-serp-item',
                                    '.serp-item',
                                    '.vacancy-card'
                                ];
                                
                                let cards = [];
                                for (const selector of selectors) {
                                    cards = document.querySelectorAll(selector);
                                    if (cards.length > 0) break;
                                }
                                
                                console.log('Найдено карточек: ' + cards.length);
                                
                                cards.forEach((card, index) => {
                                    try {
                                        // Пробуем разные селекторы для заголовка
                                        let titleElem = card.querySelector('[data-qa=""vacancy-serp__vacancy-title""]') ||
                                                       card.querySelector('.serp-item__title') ||
                                                       card.querySelector('.vacancy-card__title') ||
                                                       card.querySelector('a.bloko-link');
                                        
                                        if (titleElem && titleElem.textContent) {
                                            const vacancy = {
                                                id: index + 1,
                                                title: titleElem.textContent.trim(),
                                                url: titleElem.href || '',
                                                // Пробуем найти зарплату
                                                salary: (() => {
                                                    const salaryElem = card.querySelector('[data-qa=""vacancy-serp__vacancy-compensation""]') ||
                                                                      card.querySelector('.vacancy-serp-item__compensation') ||
                                                                      card.querySelector('.bloko-header-section-3');
                                                    return salaryElem ? salaryElem.textContent.trim() : 'Не указана';
                                                })(),
                                                // Пробуем найти компанию
                                                company: (() => {
                                                    const companyElem = card.querySelector('[data-qa=""vacancy-serp__vacancy-employer""]') ||
                                                                       card.querySelector('.vacancy-serp-item__meta-info-company') ||
                                                                       card.querySelector('.bloko-link_secondary');
                                                    return companyElem ? companyElem.textContent.trim() : 'Не указана';
                                                })()
                                            };
                                            vacancies.push(vacancy);
                                        }
                                    } catch (e) {
                                        console.log('Ошибка при обработке карточки ' + index + ': ' + e.message);
                                    }
                                });
                                
                                return JSON.stringify({
                                    success: true,
                                    count: vacancies.length,
                                    vacancies: vacancies
                                });
                            } catch (error) {
                                return JSON.stringify({
                                    success: false,
                                    error: error.message,
                                    count: 0,
                                    vacancies: []
                                });
                            }
                        }
                    ");
                    
                    // Парсим результат
                    var result = JsonSerializer.Deserialize<ParseResult>(vacanciesData);
                    
                    if (result != null && result.Success)
                    {
                        Console.WriteLine("\n" + new string('=', 50));
                        Console.WriteLine($"НАЙДЕНО ВАКАНСИЙ: {result.Count}");
                        Console.WriteLine(new string('=', 50) + "\n");
                        
                        foreach (var vacancy in result.Vacancies.Take(10))
                        {
                            Console.WriteLine($"[{vacancy.Id}] {vacancy.Title}");
                            Console.WriteLine($"    💼 Компания: {vacancy.Company}");
                            Console.WriteLine($"    💰 Зарплата: {vacancy.Salary}");
                            Console.WriteLine($"    🔗 Ссылка: {(vacancy.Url.Length > 50 ? vacancy.Url.Substring(0, 50) + "..." : vacancy.Url)}");
                            Console.WriteLine();
                        }
                        
                        // Сохраняем в файл
                        if (result.Vacancies.Count > 0)
                        {
                            var options = new JsonSerializerOptions 
                            { 
                                WriteIndented = true,
                                Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                            };
                            
                            await File.WriteAllTextAsync("vacancies.json", 
                                JsonSerializer.Serialize(result.Vacancies, options));
                            Console.WriteLine($"✅ Данные сохранены в vacancies.json");
                        }
                    }
                    else
                    {
                        Console.WriteLine("Не удалось найти вакансии на странице");
                        Console.WriteLine("Текущий HTML сохранен в page.html для отладки");
                        
                        // Сохраняем HTML для отладки
                        var content = await page.GetContentAsync();
                        await File.WriteAllTextAsync("page.html", content);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка при поиске вакансий: {ex.Message}");
                }
            }
            
            await browser.CloseAsync();
            Console.WriteLine("\n✓ Парсер завершил работу");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Критическая ошибка: {ex.Message}");
            
            // Более подробная информация
            Console.WriteLine($"Тип ошибки: {ex.GetType().Name}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Внутренняя ошибка: {ex.InnerException.Message}");
            }
        }
        
        Console.WriteLine("\nНажмите любую клавишу для выхода...");
        Console.ReadKey();
    }
    
    public class ParseResult
    {
        public bool Success { get; set; }
        public string Error { get; set; } = string.Empty;
        public int Count { get; set; }
        public List<Vacancy> Vacancies { get; set; } = new List<Vacancy>();
    }
    
    public class Vacancy
    {
        public int Id { get; set; }
        public string Title { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public string Salary { get; set; } = string.Empty;
        public string Company { get; set; } = string.Empty;
    }
}