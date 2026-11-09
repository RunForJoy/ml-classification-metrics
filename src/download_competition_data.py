import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    print("Авторизация в Kaggle...")
    api = KaggleApi()
    api.authenticate()

    target_dir = "./datasets"

    print("Начинаю скачивание данных соревнования")
    api.competition_download_files("DontGetKicked", path=target_dir)

    print("\nРаспаковка архива...")
    zip_path = os.path.join(target_dir, "DontGetKicked.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)

    print(f"Файлы успешно распакованы в папку: {target_dir}")
    print("Внутри папки лежат:", os.listdir(target_dir))

except Exception as e:
    print("\n Произошла ошибка при скачивании:")
    print(e)
    if "Unauthorized" in str(e) or "Could not find" in str(e):
        print("\n Похоже, у вас не настроен Kaggle API токен.")
        print(
            "Пожалуйста, ознакомьтесь с инструкцией в файле README.md, чтобы настроить доступ!"
        )
