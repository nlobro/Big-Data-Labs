# Big Data Labs

Лабораторные работы по дисциплине «Большие данные» (1.1–3.2).

## Структура
- `chapter_1_2/` — `ex1.py`, `ex2.py`, `ex3.py`, `ex21.py`, `ex22.py`, `ex23.py` и все исходные данные: `Customers.csv`, `Clusters.csv`, `Analysis.csv`, `Rural_prices.csv`, `FinCond.csv`.
- `chapter_3/` — `ex31.py` (Streamlit), `ex32.py` (Gradio) и данные для трёх дополнительных моделей.
- `requirements.txt` — необходимые пакеты Python.

Все Python-файлы сохраняют основную и самостоятельную части заданий.

Для `FinCond` используется CSV-формат вместо исходного XLSX: содержимое таблицы сохранено, а код адаптирован к `FinCond.csv`.

## Работа в Visual Studio / VS Code
Установить зависимости: `pip install -r requirements.txt`.

Лабораторные 1.1–2.3 открываются как обычные Python-файлы. Они читают данные относительно своего расположения.

Streamlit: `python -m streamlit run chapter_3/ex31.py`.

Gradio: `python chapter_3/ex32.py`.

Исходный `Rural_prices.csv` получен из публичного учебного репозитория:
https://github.com/Alenushka2013/Python_Data_Science
