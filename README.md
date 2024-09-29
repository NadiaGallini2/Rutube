🌟 Быстрый старт в Рутуб – Рекомендуем видео с умом! 🚀

https://github.com/user-attachments/assets/d66495fd-da57-4204-9082-7588deaccf99

## Запуск проекта

Чтобы запустить проект, выполните следующие шаги: сначала клонируйте репозиторий на свою машину с помощью
1. Сначала клонируйте репозиторий на свою машину с помощью команды:
   ```bash
   git clone https://github.com/NadiaGallini2/Rutube.git

и перейдите в каталог проекта. 

Затем убедитесь, что у вас установлен Python (рекомендуется версия 3.7 или выше), и установите необходимые зависимости, используя 
   ```bash
      pip install -r requirements.txt
   ```


После этого запустите приложение Streamlit командой 
   ```bash
   streamlit run app_rutube.py
   ```


Откройте браузер и перейдите по адресу [http://localhost:8501](http://localhost:8501), чтобы увидеть интерфейс.


# Интеграция трансформеров с мультимодальной функцией слияния на основе марковской цепи и ансамбля, обученного авторегрессией и автоэнкодированием
Также можно попробовать запустить код в файле [Model.ipynb](https://github.com/NadiaGallini2/Rutube/blob/main/Model.ipynb)

Этот проект реализует рекомендательную систему, использующую несколько методов для генерации рекомендаций видео. Основные компоненты включают:

- **GSASRec**: Модель на основе последовательностей для рекомендаций.
- **Bert4Rec**: Модель, использующая архитектуру BERT для рекомендаций.
- **Word2Vec**: Модель для извлечения похожих видео на основе пользовательских сессий.

## Установка

Перед запуском проекта убедитесь, что у вас установлен Python 3.6 или выше. Рекомендуется создать виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate  # Для Windows
```

# Загрузка моделей
```
loaded_gsasrec_model = load_model('gsasrec_model.h5')
loaded_bert4rec_model = load_model('bert4rec_model.h5')
loaded_transformer_xlnet_model = load_model('transformer_xlnet_model.h5')
loaded_word2vec_model = Word2Vec.load('word2vec_model.model')

ensemble_recommendations = improve_with_transformer_xlnet(
    bert4rec_recommendations, 
    gsasrec_recommendations, 
    loaded_transformer_xlnet_model, 
    interaction_matrix, 
    user_id, 
    top_n=top_n
)
