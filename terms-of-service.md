import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx, CompositeVideoClip
from tqdm import tqdm  # Библиотека для прогресс-бара
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import time  # Для задержки между загрузками

# Параметры YouTube API
CLIENT_SECRETS_FILE = "client_secrets.json"  # Файл с учетными данными
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def get_authenticated_service():
    """
    Аутентификация и создание сервиса YouTube API.
    """
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def upload_video(youtube, video_path, title, description, tags):
    """
    Загружает видео на YouTube.
    """
    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"  # Категория "Люди и блоги"
        },
        "status": {
            "privacyStatus": "private"  # Можно изменить на "public" или "unlisted"
        }
    }

    media = MediaFileUpload(video_path)
    response = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media
    ).execute()

    print(f"Видео загружено: {response['id']}")
    return response


def split_video_into_segments(video_path, segment_duration, output_folder):
    video = VideoFileClip(video_path)
    total_duration = video.duration
    saved_segments = []

    for i, start_time in enumerate(tqdm(range(0, int(total_duration), segment_duration), desc="Разделение видео", unit="segment")):
        end_time = start_time + segment_duration
        if end_time > total_duration:
            break

        segment = video.subclip(start_time, end_time)
        output_path = os.path.join(output_folder, f"segment_{i + 1}.mp4")
        segment.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
        print(f"Отрезок {i + 1} сохранен: {output_path}")
        saved_segments.append(output_path)

    return saved_segments


def process_vertical_format(video_clip):
    target_width = 1080
    target_height = 1920

    # Определяем исходные размеры видео
    original_width, original_height = video_clip.size

    # 1. Создаем размытую нижнюю дорожку
    def blur_frame(get_frame, t):
        frame = get_frame(t)  # Получаем кадр
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Преобразуем цветовое пространство
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 15)  # Увеличиваем размытие до 15%
        blurred_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)  # Возвращаем цветовое пространство
        return blurred_frame

    blur_clip = video_clip.fl(blur_frame).resize(height=target_height)  # Масштабируем по высоте (1920 пикселей)

    # Центрируем размытое видео по ширине
    blur_clip = blur_clip.crop(
        x_center=blur_clip.w // 2,
        y_center=blur_clip.h // 2,
        width=target_width,
        height=target_height
    )

    # 2. Создаем верхнюю дорожку
    top_clip = video_clip.resize(width=target_width)  # Масштабируем по ширине (1080 пикселей)

    # Увеличиваем верхнюю дорожку на 40%
    top_clip = top_clip.fx(vfx.resize, 1.4)

    # Центрируем верхнее видео по высоте
    top_clip = top_clip.crop(
        x_center=top_clip.w // 2,
        y_center=top_clip.h // 2,
        width=target_width,
        height=target_height
    )

    # Добавляем эффекты к верхней дорожке
    top_clip = top_clip.fx(vfx.lum_contrast, contrast=-0.1)  # Уменьшаем контрастность на 0.3
    top_clip = top_clip.fx(vfx.colorx, factor=0.7)  # Увеличиваем насыщенность на 2 пункта

    # Добавляем резкость через OpenCV
    def sharpen_frame(get_frame, t):
        frame = get_frame(t)  # Получаем кадр
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Преобразуем цветовое пространство

        # Применяем фильтр резкости (ядро свертки)
        kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]], dtype=np.float32)
        sharpened_frame = cv2.filter2D(frame, -1, kernel)

        sharpened_frame = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)  # Возвращаем цветовое пространство
        return sharpened_frame

    top_clip = top_clip.fl(sharpen_frame)

    # Размещаем верхнюю дорожку по центру
    top_clip = top_clip.set_position(("center", "center"))

    # 3. Комбинируем все элементы
    final_clip = CompositeVideoClip([blur_clip, top_clip])

    # 4. Растягиваем длительность до 58 секунд
    stretched_clip = final_clip.fx(vfx.speedx, final_duration=58)

    return stretched_clip


def add_music_with_volume(video_clip, music_path):
    """
    Накладывает музыку на видео с громкостью в половину от исходной.
    Если музыка короче видео, она зацикливается.
    """
    music = AudioFileClip(music_path)

    # Уменьшаем громкость музыки в половину
    music = music.volumex(0.1)  # Громкость = 0.5 (вместо 1.0)

    # Зацикливаем музыку, если её длительность меньше длительности видео
    if music.duration < video_clip.duration:
        music = music.fx(vfx.loop, duration=video_clip.duration)

    # Обрезаем музыку до длины видео (если она длиннее)
    music = music.subclip(0, video_clip.duration)

    # Смешиваем основной звук видео и музыку
    final_audio = CompositeAudioClip([video_clip.audio, music])

    # Накладываем аудио на видео
    final_clip = video_clip.set_audio(final_audio)

    return final_clip


def process_saved_segments(prom_folder, music_path, output_folder):
    segment_files = [f for f in os.listdir(prom_folder) if f.endswith(".mp4")]
    segment_files.sort()

    for i, segment_file in enumerate(tqdm(segment_files, desc="Обработка отрезков", unit="segment")):
        segment_path = os.path.join(prom_folder, segment_file)
        video_clip = VideoFileClip(segment_path)

        vertical_clip = process_vertical_format(video_clip)
        final_clip = add_music_with_volume(vertical_clip, music_path)

        output_path = os.path.join(output_folder, f"processed_{i + 1}.mp4")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
        print(f"Обработанное видео сохранено: {output_path}")


def upload_videos_to_youtube(output_folder):
    youtube = get_authenticated_service()
    video_files = [f for f in os.listdir(output_folder) if f.endswith(".mp4")]
    video_files.sort()

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(output_folder, video_file)
        title = f"Видео {i + 1}"
        description = "Это автоматически загруженное видео."
        tags = ["автоматическая загрузка", "вертикальное видео"]

        print(f"Загрузка видео {i + 1}: {video_path}")
        upload_video(youtube, video_path, title, description, tags)

        # Ждем 10 минут перед следующей загрузкой
        if i < len(video_files) - 1:
            print("Ждем 10 минут перед следующей загрузкой...")
            time.sleep(10 * 60)


if __name__ == "__main__":
    input_video = r"C:\py.project\PTV\Crop_vertikal\video\input.mp4"
    music_path = r"C:\py.project\PTV\Crop_vertikal\music\music.mp3"
    prom_folder = r"C:\py.project\PTV\Crop_vertikal\prom"
    output_folder = r"C:\py.project\PTV\Crop_vertikal\output"
    segment_duration = 55  # Разделение видео на отрезки по 55 секунд

    if not os.path.exists(input_video):
        print(f"Ошибка: файл {input_video} не найден!")
        exit(1)

    if not os.path.exists(music_path):
        print(f"Ошибка: файл {music_path} не найден!")
        exit(1)

    os.makedirs(prom_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    saved_segments = split_video_into_segments(input_video, segment_duration, prom_folder)
    process_saved_segments(prom_folder, music_path, output_folder)

    # Загрузка видео на YouTube
    upload_videos_to_youtube(output_folder)

    print("Обработка завершена!")