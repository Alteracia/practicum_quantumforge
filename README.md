# Quantum Forge Software
## Задание 1 - Подбор конфигурации
### Сравнение моделей, облако vs локальные
| Критерий                   | Локальные модели                                                                                   | Облачные модели (OpenAI / YandexGPT)                                                     |
|----------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Качество ответов**       | В целом слабее для комплексных задач;<br> хороши для стандартных задач (особенно после дообучения) | Высокие результаты в логике, длинных контекстах, мультимодальности                       |
| **Скорость работы**        | Зависит от железа;<br> при нехватке VRAM и квантовании скорость падает                             | Стабильная низкая задержка за счёт оптимизированной инфраструктуры;<br> Сетевые задержки |
| **Стоимость**              | Разовые затраты на железо + операционные расходы;<br> выгодны при больших объёмах запросов         | Оплата по токенам;<br> удобно для старта, но затраты растут линейно с нагрузкой          |
| **Удобство развёртывания** | Требуют технических навыков и настройки                                                            | Подключение через API;<br> не нужно управлять инфраструктурой                            |

### Сравнение моделей эмбеддингов, облако vs локальные
| Критерий                           | Локальные (BGE-M3 E5-Large-v2 GTE-Large)                                                                             | Облачные OpenAI Embeddings                                                                                                                                        |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Скорость создания индекса          | Высокая при наличии GPU;<br> на CPU — заметно ниже.                                                                  | Быстрая генерация на стороне сервиса, но общая скорость ограничена сетевой задержкой и пропускной способностью API — может быть узким местом при больших объёмах. |
| Качество поиска                    | Хорошее, но в сложных сценариях зачастую уступает топовым облачным решениям;<br> сильно зависит от выбранной модели. | Высокое: стабильно показывает сильные результаты в бенчмарках                                                                                                     |
| Стоимость владения и использования | Затраты на инфраструктуру (GPU/CPU)                                                                                  | Плата за токен; стоимость растёт пропорционально объёму данных                                                                                                    |

### Сравнение векторных баз ChromaDB и FAISS
| Критерий                        | ChromaDB                                                             | FAISS                                                                                       |
|---------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Скорость поиска и индексации    | Хорошая для небольших и средних объёмов (до нескольких млн векторов) | Высокая, особенно на больших объёмах                                                        |
| Сложность внедрения и поддержки | Низкая: простая установка через pip                                  | Низкая, но только если не нужны доп фичи: БД, персистентность, фильтрация, мониторинг и др. |
| Удобство в работе               | Удобно есть meta, БД, персистентность, фильтрация из коробки         | Больше контроля и тонкой настройки;<br> сложнее если нужны доп фичи или масштабирование     |
| Стоимость владения              | Низкие                                                               | Потенциально ниже при больших нагрузках за счёт оптимизации;<br> выше при дополнении        |

### Варианты конфигураций
|                 | Cloud                                                                                                                              | Hybrid                                                                             | Локальная ChromaDB                                                          | Локальная FAISS                                                                |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| LLM-model       | OpenAI                                                                                                                             | Локальная                                                                          | Локальная                                                                   | Локальная                                                                      |
| Embedding-model | OpenAI                                                                                                                             | OpenAI                                                                             | Локальная                                                                   | Локальная                                                                      |
| Vector DB       | ChromaDB                                                                                                                           | FAISS                                                                              | ChromaDB                                                                    | FAISS                                                                          |
| CPU             | 8 cores                                                                                                                            | 12 cors                                                                            | 16 cors                                                                     | 16 cors                                                                        |
| GPU             | None                                                                                                                               | 16 VRAM                                                                            | 24 VRAM                                                                     | 24 VRAM                                                                        |
| RAM             | 32 GB                                                                                                                              | 48 GB                                                                              | 64 GB                                                                       | 48 GB                                                                          |
| Плюсы           | Low server cost;<br> Высокое качество моделей;<br> Готовая, все функции из коробки                                                 | Низкие расходы per-token;<br> Модель можно доучить;<br> Приватность запросов к LLM | Нет расходов per-token;<br> Полная приватность;<br> Не требует доработки DB | Нет расходов per-token;<br> Полная приватность;<br> Высокая скорость поиска DB |
| Минусы          | Оплата per token, может сильно вырасти при больших объемах;<br> Network latency;<br> Низкая приватность;<br> Модель не доучиваемая | Расходы на сервер выше;<br> Эмбеддинг не приватный;<br> Требуется разработка       | Высокие расходы на сервер;<br> Скорость поиска DB ниже чем у FAISS          | Высокие расходы на сервер;<br> Требует доработки DB                            |

### Итоговая рекомендация
- Local Hugging Face LLM with LoRA/QLoRA fine-tuning.
- Local embedding model BGE-M3, лучшее качество, средняя скорость.
- FAISS, с добавлением фильтрации. С учетом требования по масштабируемости.
- Linux server

| Компонент | Рекомендация            |
|-----------|-------------------------|
| CPU       | 16 physical cores       |
| RAM       | 64 GB                   |
| GPU       | NVIDIA RTX 24 GB VRAM   |
| Storage   | 2×2 TB NVMe SSD (RAID1) |
| Network   | 1 Gbps                  |

## Задание 2 - Подготовка данных
Для сбора данных взята вселенная Doctor WHO [Tardis](https://tardis.fandom.com/wiki/Doctor_Who_Wiki).<br> 
Замена с помощью скрипта [rename_terms.py](/knowledge_base/rename_terms.py) на основе [словаря замен](/knowledge_base/terms_map.json).
### Результат
37 .md файлов в `/knowledge_base`

## Задание 3 - Индексация и Эмбеддинг
### Модель
- BGE-M3
- [Hugging Face](https://huggingface.co/BAAI/bge-m3), [Github](https://github.com/FlagOpen/FlagEmbedding)
- Size: 671MB
- Context: 512

### Установка зависимостей
#### Установки torch
GPU (рекомендуемое если есть подходящий GPU):
```shell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
или CPU-only fallback (no NVIDIA GPU, or limited VRAM/RAM):
```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
#### Остальные
```shell
pip install numpy faiss-cpu sentence-transformers langchain langchain-text-splitters
```
### Запуск
```shell
python build_index.py                    # build (if missing) + run 3 quality-check queries
python build_index.py --rebuild          # force full rebuild
python build_index.py --query "Who is Cypher?" --k 5
python build_index.py --no-check         # build only
```
### Notes:
Auto-detects CUDA and uses GPU if present (RTX 3060 here → ~40 chunks/s, ~105 s vs ~35 min on CPU). Falls back to CPU otherwise.
First run downloads the BGE-M3 weights (~2 GB) — a friendly message is printed.
Memory-safety: batched embedding + automatic OOM recovery addresses the "too many documents / limited memory" concern; chunk size can be lowered if needed.
### Результат
- Chunking: chunk_size=800, overlap=100 → 4194 chunks (avg 606 chars).
- Индексация 105.6s на cuda (NVIDIA GeForce RTX 3060 Laptop GPU, 6.0 GB VRAM)
- Загрузка модели 14.4s (embedding dim = 1024)
#### Проверка
```
======================================================================
Single query (k=5)
======================================================================

  QUERY: "Who is Cypher?"
    1. score=0.5557 | knowledge_base/Cypher.md | title="Cypher" | chunk_id=134 [chars 79055-79822]
       "Cypher Bounty Hunters aboard a spaceship called the Wayfarer. About the same time, the people of the planet Auros learned of an approaching Cypher fleet, and the population fled before the Cyphers arrived. Meanwhile, the Cyphers waited for ..."
    2. score=0.5515 | knowledge_base/Cypher.md | title="Cypher" | chunk_id=316
       "Cypher activated the Cypher Factor in Kate Yates who developed a Cypher personality and touched the dead casing, causing her Cypher life-force to grow a new Cypher in the casing from the casing's databanks. The Cypher hunted down Frank and ..."
    3. score=0.5445 | knowledge_base/Rose Turner.md | title="Rose Turner" | chunk_id=48 [chars 26090-26852]
       "Rose meets the last Cypher. (TV: Cypher ) Tracking a distress signal, the Professor and Rose arrived in 2012 Utah, in an underground alien musueum called the Vault owned by billionaire Henry van Statten. While the Professor, unbeknownst to ..."
    4. score=0.5416 | knowledge_base/Cypher.md | title="Cypher" | chunk_id=65
       "technology at an accelerated rate, soon surpassing their Earthly brothers by millions of years. According to the scientist Bryant Anderson, who was later able to confirm via dissection that, to his great distress, the Cypher creatures were ..."
    5. score=0.5414 | knowledge_base/Cypher.md | title="Cypher" | chunk_id=60 [chars 36700-37344]
       "the Cyphers, (AUDIO: The Four Professors ) several sources agreed that the Cypher Prime was the first-ever Cypher, (PROSE: The Evil of the Cyphers , War of the Cyphers and the one who had made the decision to exterminate the Cyphers' own cr ..."

======================================================================
```
#### Артефакты
| Файл         | Содержание                                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------------------|
| faiss.index  | Flat inner-product FAISS index, 4194 vectors × 1024 dims (~16.4 MB)                                                |
| metadata.pkl | Сопутствующее сопоставление метаданных row id → {source, title, chunk_id, text, char_start, char_end, …} (~2.7 MB) |

## Задание 4 - Реализация RAG-бота
```shell
pip install openai
```
```powershell
$env:OPENAI_API_KEY = "sk-..."
python rag_bot.py
```