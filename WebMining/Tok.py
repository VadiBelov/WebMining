import webbrowser
import urllib.parse


def get_vk_token():
    params = {
        'client_id': '54391407',  # ID вашего приложения
        'display': 'page',
        'redirect_uri': 'https://oauth.vk.com/blank.html',
        'scope': 'friends,groups',
        'response_type': 'token',
        'v': '5.199'
    }

    url = "https://oauth.vk.com/authorize?" + urllib.parse.urlencode(params)
    print(f"Перейдите по ссылке: {url}")
    print("\nПосле авторизации скопируйте токен из адресной строки браузера")
    print("(часть после access_token= до &)")
    webbrowser.open(url)

    token = input("\nВведите токен: ").strip()
    return token


if __name__ == "__main__":
    token = get_vk_token()
    print(f"\nПолучен токен: {token[:20]}...")
    print("\nВставьте его в основной код:")
    print(f"token = '{token}'")