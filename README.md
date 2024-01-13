# Отчет по задаче


## Решение ML стеком

Обработка данных производится функцией preprocessing из скрипта preprocessing.py

## Решение с помощью LLM


В качестве LLM использовалась GPT4all 

Установка GPT4all:

```
pip install gpt4all

mkdir models

wget https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf -O models/mistral-7b-openorca.Q4_0.gguf
```


Краткое описание файлов:

В скрипте templates содержатся контексты для пользовательских промптов

В скрипте examples лежат примеры для few-shot learning



 Использовался подход Automatic Chain of Thought Prompting:
 https://github.com/amazon-science/auto-cot

Промпты, состоящие из нескольких подвопросов, декомпозировались

Агент general_knowledge_agent находил общую информацию по теме,
чтобы помочь другому ответу ответить на конкретный вопрос


Сгенерированный код для ответов на промпты лежит в папке generated_code

Как запускать: