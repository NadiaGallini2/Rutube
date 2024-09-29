import streamlit as st
import pandas as pd
import numpy as np
from random import shuffle
from gensim.models import Word2Vec
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Загрузка данных из CSV и Parquet
@st.cache_data
def load_data():
    logs = pd.read_csv('cleaned_dataset.csv')  # Убедитесь, что файл 'cleaned_dataset.csv' находится в папке проекта
    video_stat = pd.read_parquet('video_stat.parquet')  # Убедитесь, что файл 'video_stat.parquet' находится в папке проекта
    return logs, video_stat

# Предобработка данных
def preprocess_data(logs, video_stat):
    logs['watchtime'] = logs['watchtime'].clip(lower=0)
    interaction_matrix = logs.pivot_table(index='user_id', columns='video_id', values='watchtime', fill_value=0)
    return interaction_matrix

# Модель GSASRec
class GSASRec(Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(GSASRec, self).__init__()
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, input_length=1)
        self.dense_layer = Dense(128, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, item_id = inputs
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = user_embedding * item_embedding
        x = self.dense_layer(x)
        return self.output_layer(x)

# Обучение GSASRec
@st.cache_resource
def train_gsasrec(interaction_matrix):
    num_users, num_items = interaction_matrix.shape
    model = GSASRec(num_users, num_items, embedding_size=50)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    user_ids, item_ids, labels = [], [], []
    for user_id, row in interaction_matrix.iterrows():
        for video_id, watchtime in row.iteritems():
            user_ids.append(user_id)
            item_ids.append(video_id)
            labels.append(1 if watchtime > 0 else 0)

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    labels = np.array(labels)
    model.fit([user_ids, item_ids], labels, epochs=10, batch_size=32, verbose=0)
    return model

# Модель Bert4Rec
class Bert4Rec(Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(Bert4Rec, self).__init__()
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, input_length=1)
        self.lstm_layer = LSTM(64, return_sequences=False)
        self.dense_layer = Dense(128, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, item_id = inputs
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = user_embedding * item_embedding
        x = self.lstm_layer(x)
        x = self.dense_layer(x)
        return self.output_layer(x)

# Обучение Bert4Rec
@st.cache_resource
def train_bert4rec(interaction_matrix):
    num_users, num_items = interaction_matrix.shape
    model = Bert4Rec(num_users, num_items, embedding_size=50)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    user_ids, item_ids, labels = [], [], []
    for user_id, row in interaction_matrix.iterrows():
        for video_id, watchtime in row.iteritems():
            user_ids.append(user_id)
            item_ids.append(video_id)
            labels.append(1 if watchtime > 0 else 0)

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    labels = np.array(labels)
    model.fit([user_ids, item_ids], labels, epochs=10, batch_size=32, verbose=0)
    return model

# Word2Vec для рекомендаций
@st.cache_resource
def train_word2vec_model(logs):
    logs['video_id'] = logs['video_id'].astype(str)
    user_sessions = logs.groupby('user_id')['video_id'].apply(list)
    model = Word2Vec(sentences=user_sessions, vector_size=50, window=5, min_count=1, sg=1)
    return model

# Рекомендации Word2Vec
def recommend_videos_word2vec(video_id, word2vec_model, top_n=10):
    try:
        recommendations = word2vec_model.wv.most_similar(str(video_id), topn=top_n)
        recommended_videos = [int(video[0]) for video in recommendations]
    except KeyError:
        recommended_videos = []
    return recommended_videos

# Рекомендации GSASRec
def recommend_videos_gsasrec(user_id, model, interaction_matrix, top_n=10):
    if user_id not in interaction_matrix.index:
        return []

    user_index = user_id  # Предполагается, что user_id соответствует индексу
    item_indices = np.arange(interaction_matrix.shape[1])
    predictions = model.predict([np.full_like(item_indices, user_index), item_indices], verbose=0)
    recommended_indices = predictions.flatten().argsort()[-top_n*2:][::-1]
    recommended_videos = interaction_matrix.columns[recommended_indices].tolist()

    watched_videos = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index.tolist()
    recommended_videos = [video for video in recommended_videos if video not in watched_videos]
    return recommended_videos[:top_n]

# Рекомендации Bert4Rec (похожи на GSASRec)
def recommend_videos_bert4rec(user_id, model, interaction_matrix, top_n=10):
    if user_id not in interaction_matrix.index:
        return []

    user_index = user_id  # Предполагается, что user_id соответствует индексу
    item_indices = np.arange(interaction_matrix.shape[1])
    predictions = model.predict([np.full_like(item_indices, user_index), item_indices], verbose=0)
    recommended_indices = predictions.flatten().argsort()[-top_n*2:][::-1]
    recommended_videos = interaction_matrix.columns[recommended_indices].tolist()

    watched_videos = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index.tolist()
    recommended_videos = [video for video in recommended_videos if video not in watched_videos]
    return recommended_videos[:top_n]

# Ансамбль Bert4Rec и GSASRec (упрощённый вариант)
def recommend_ensemble(user_id, gsasrec_model, bert4rec_model, interaction_matrix, video_stat, top_n=10):
    gsasrec_recommendations = recommend_videos_gsasrec(user_id, gsasrec_model, interaction_matrix, top_n=top_n)
    bert4rec_recommendations = recommend_videos_bert4rec(user_id, bert4rec_model, interaction_matrix, top_n=top_n)

    ensemble_recommendations = list(set(gsasrec_recommendations) & set(bert4rec_recommendations))

    if len(ensemble_recommendations) < top_n:
        combined_recommendations = list(set(gsasrec_recommendations + bert4rec_recommendations))
        shuffle(combined_recommendations)

        final_recommendations = []
        categories_count = {}

        for video in combined_recommendations:
            category_series = video_stat[video_stat['video_id'] == video]['category']
            if category_series.empty:
                continue
            category = category_series.values[0]
            if categories_count.get(category, 0) < 2:
                final_recommendations.append(video)
                categories_count[category] = categories_count.get(category, 0) + 1
            if len(final_recommendations) >= top_n:
                break
    else:
        final_recommendations = ensemble_recommendations[:top_n]

    return final_recommendations

# Ограничение количества видео из одной категории
def limit_videos_per_category(recommendations, video_stat, max_per_category=2):
    video_categories = video_stat.set_index('video_id')['category']
    category_count = {}
    filtered_recommendations = []

    for video_id in recommendations:
        category = video_categories.get(video_id)
        if category is None:
            continue

        if category_count.get(category, 0) < max_per_category:
            filtered_recommendations.append(video_id)
            category_count[category] = category_count.get(category, 0) + 1

    return filtered_recommendations

# Основная функция для рекомендаций
def recommend_for_user(user_id, interaction_matrix, bert4rec_model, gsasrec_model, w2v_model, video_stat, top_n=10):
    if user_id not in interaction_matrix.index:
        # Новый или неизвестный пользователь - показываем популярные видео
        popular_videos = video_stat.sort_values(by='v_total_comments', ascending=False)['video_id'].head(top_n)
        return popular_videos.tolist()

    # Получаем рекомендации от GSASRec и Bert4Rec
    gsasrec_recommendations = recommend_videos_gsasrec(user_id, gsasrec_model, interaction_matrix, top_n=top_n)
    bert4rec_recommendations = recommend_videos_bert4rec(user_id, bert4rec_model, interaction_matrix, top_n=top_n)

    # Создаём ансамбль рекомендаций
    ensemble_recommendations = recommend_ensemble(user_id, gsasrec_model, bert4rec_model, interaction_matrix, video_stat, top_n=top_n)

    # Если после ансамбля рекомендаций всё ещё меньше top_n, дополняем Word2Vec
    if len(ensemble_recommendations) < top_n:
        # Предполагаем, что последнее рекомендованное видео является релевантным для Word2Vec
        last_recommended_video = ensemble_recommendations[-1] if ensemble_recommendations else None
        if last_recommended_video:
            w2v_recommendations = recommend_videos_word2vec(last_recommended_video, w2v_model, top_n=top_n - len(ensemble_recommendations))
            # Убираем повторы из рекомендаций Word2Vec
            w2v_recommendations = [vid for vid in w2v_recommendations if vid not in ensemble_recommendations]
            ensemble_recommendations.extend(w2v_recommendations)

    # Финальная фильтрация и ограничение по категориям
    final_recommendations = limit_videos_per_category(ensemble_recommendations, video_stat, max_per_category=2)

    return final_recommendations[:top_n]

# Функция отображения рекомендаций с дополнительной информацией
def display_recommendations(recommendations, video_stat):
    if recommendations:
        recommended_videos = video_stat[video_stat['video_id'].isin(recommendations)]
        st.write("### Рекомендованные видео:")
        for idx, row in recommended_videos.iterrows():
            st.write(f"**Видео ID**: {row['video_id']}, **Категория**: {row['category']}, **Комментарии**: {row['v_total_comments']}")
    else:
        st.write("Нет доступных рекомендаций.")

# Streamlit интерфейс
# Streamlit интерфейс
def main():
    st.title("Видео-рекомендательная система")

    st.write("### Загрузка данных")
    logs, video_stat = load_data()
    st.success("Данные успешно загружены!")

    st.write("### Предобработка данных")
    interaction_matrix = preprocess_data(logs, video_stat)
    st.success("Предобработка данных завершена!")

    st.write("### Обучение моделей GSASRec и Bert4Rec")
    with st.spinner("Обучение модели GSASRec..."):
        gsasrec_model = train_gsasrec(interaction_matrix)
    st.success("Модель GSASRec обучена!")

    with st.spinner("Обучение модели Bert4Rec..."):
        bert4rec_model = train_bert4rec(interaction_matrix)
    st.success("Модель Bert4Rec обучена!")

    st.write("### Обучение модели Word2Vec")
    with st.spinner("Обучение модели Word2Vec..."):
        word2vec_model = train_word2vec_model(logs)
    st.success("Модель Word2Vec обучена!")

    st.write("### Получение рекомендаций")
    user_id_input = st.text_input("Введите ID пользователя для получения рекомендаций:", "")

    if st.button("Получить рекомендации"):
        if user_id_input:
            try:
                user_id = int(user_id_input)  # Преобразуем ID пользователя в целое число
                recommendations = recommend_for_user(user_id, interaction_matrix, bert4rec_model, gsasrec_model, word2vec_model, video_stat, top_n=10)
                display_recommendations(recommendations, video_stat)
            except ValueError:
                st.error("Пожалуйста, введите корректный ID пользователя.")
        else:
            st.warning("Пожалуйста, введите ID пользователя для получения рекомендаций.")

if __name__ == "__main__":
    main()
