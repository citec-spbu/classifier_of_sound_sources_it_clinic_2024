"""
Класс, реализующий различные методы предобработки аудиосигналов
"""
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


class AudioPreprocessor:
    def __init__(self,
                 sr=None,
                 audio=None,
                 offset=0.0,
                 duration=None):
        """
        Конструктор класса
        
        Parameters:
        ----------
        sr: int
            sampling rate, default = None
        audio: np.ndarray 
            Временной ряд аудиосигнала, default = None
        offset: float
            Сдвиг (с какой секунды начинаем аудио), default = 0.0
        duration: float
            Сколько секунд прослушиваем аудио, default = None
        """
        self.sr = sr
        self.audio = audio
        self.offset = offset
        self.duration = duration

    def load_audio(self, file_path: str):
        """
        Загрузка аудиосигнала, если он не был передан в конструкторе
        
        Parameters:
        ----------
        file_path: str
            Путь к файлу
        """
        self.audio, self.sr = librosa.load(file_path, sr=self.sr, 
                                           offset=self.offset, duration=self.duration)
 
    def return_audio(self):
        """
        Вернуть аудио (например, для просмотра его состояния после преобразований)
        """
        return self.audio

    # обработка аудио
    def detect_noise_section(self, method='amplitude', threshold=0.02, noise_duration=0.5):
        """
        Определяет сегмент шума в аудио различными методами.

        Parameters:
        ----------
        method: str
            Метод обнаружения шума: 'amplitude', 'frequency', 'silence'.
            default = amplitude

        threshold: float
            Порог для обнаружения шума. Используется по-разному в зависимости от метода.
            default = 0.02

        noise_duration: float
            Длина фрагмента шума в секундах.
            default = 0.5

        Возвращает:
        ----------
        (start, end): кортеж
            Временной интервал в секундах, где обнаружен шум.
        """
        if method == 'amplitude':
            return self._detect_by_amplitude(threshold, noise_duration)
        elif method == 'frequency':
            return self._detect_by_frequency(threshold, noise_duration)
        elif method == 'silence':
            return self._detect_by_silence(threshold, noise_duration)
        else:
            raise ValueError("Неправильный метод. Используйте 'amplitude', 'frequency' или 'silence'.")

    def _detect_by_amplitude(self, threshold, noise_duration):
        """
        Обнаружение шума с помощью энергии сигнала (поиск областей низкой энергии).
        Метод, который может принести результаты, если шум имеет постоянную низкую амплитуду.
        """
        # Квадрат амплитуды аудио (энергия сигнала)
        energy = np.square(self.audio)
        low_energy_indices = np.where(energy < threshold)[0]
        segment_length = int(noise_duration * self.sr)
        
        for start in low_energy_indices:
            end = start + segment_length
            if end < len(self.audio):
                return (start / self.sr, end / self.sr)

        return (0, noise_duration)

    def _detect_by_frequency(self, threshold, noise_duration):
        """
        Обнаружение шума с помощью частот. 
        Используется преобразование Фурье для вычисления спектра 
        и поиска постоянных низкочастотных компонентов, характерных для шума.
        """
        # Вычисление спектра через STFT
        D = np.abs(librosa.stft(self.audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Средний спектр сигнала
        avg_spectrum = np.mean(D, axis=1)

        # Низкочастотные компоненты могут указывать на шум
        low_freq_indices = np.where(freqs < 300)  # Используем порог 300 Гц как пример
        avg_low_freq = np.mean(avg_spectrum[low_freq_indices])

        if avg_low_freq > threshold:
            start = 0
            end = int(noise_duration * self.sr)
            return (start / self.sr, end / self.sr)

        return (0, noise_duration)

    def _detect_by_silence(self, threshold, noise_duration):
        """
        Обнаружение шума
        """
        
        intervals = librosa.effects.split(self.audio, top_db=threshold)
        segment_length = int(noise_duration * self.sr)

        if len(intervals) > 0:
            for start, end in intervals:
                if end - start >= segment_length:
                    return (start / self.sr, (start + segment_length) / self.sr)

        return (0, noise_duration)

    def remove_noise(self, noise_start=None, noise_end=None):
        """
        Удаление шума

        Parameters:
        -----------
        noise_start: float
            Начальный момент времени в секундах для захвата примера шума,
            default = None
        
        noise_end: float
            Конечный момент времени в секундах для захвата примера шума,
            default = None
        """
        if noise_start and noise_end:
            noise_sample = self.audio[int(noise_start * self.sr):int(noise_end * self.sr)]
            self.audio = nr.reduce_noise(y=self.audio,
                                         sr=self.sr,
                                         y_noise=noise_sample)
        else:
            self.audio = nr.reduce_noise(y=self.audio,
                                         sr=self.sr)
        
    def preemphasis(self, coef=0.97):
        """
        Метод преэмфазиса. Увеличивает амплитуду высокочастотных компонент сигнала, уменьшая влияние низкочастотного шума 
        и улучшая общее качество сигнала для последующей обработки
        
        Parameters:
        -----------
        coef: float
            Коэффиент преэмфазиса. Обычно близок к 1.
        """
        self.audio = librosa.effects.preemphasis(self.audio, coef=coef)

    def normalize(self, method="peak", coef=0.1):
        """
        Нормализация сигнала
        
        Parameters:
        -----------
        method: str
            Метод масштабирования сигнала: "peak", "rms"
        coef: float
            Коэффициент для масштабирования при использовании "rms"
        """
        if method == "peak":
            self.audio = self.audio / np.max(np.abs(self.audio))
        elif method == "rms":
            self.audio = self.audio * (coef / np.sqrt(np.mean(self.audio**2)))
        else:
            raise ValueError("Неправильный метод. Используйте 'peak' или 'rms'")
    
    def equalize(self, lowcut, highcut, order=5):
        """
        Эквализация с помощью фильтра Бабичава-Баттерворта.

        Параметры:
        ----------
        lowcut: float
            Нижняя граница частоты для фильтрации.
        
        highcut: float
            Верхняя граница частоты для фильтрации.
        
        order: int
            Порядок фильтра, определяет "резкость" фильтра.
        """
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        self.audio = lfilter(b, a, self.audio)

    def trim(self):
        """
        Обрезает тишину в начале и конце аудиосигнала.
        """
        self.audio, _ = librosa.effects.trim(self.audio)

    def windowing(self, window_length=1024, hop_size=512):
        """
        Применяет оконную функцию Hamming к аудиосигналу для анализа.
        
        Параметры:
        ----------
        window_length: int
            Длина окна, default = 1024
        
        hop_size: int
            Шаг между окнами, default = 512
        """
        window_function = np.hamming(window_length)
        windows = librosa.util.frame(self.audio, frame_length=window_length, hop_length=hop_size)
        windowed_data = windows * window_function[:, np.newaxis]
        return windowed_data

    def preprocessing_pipeline(self, methods):
        for method in methods:
            if hasattr(self, method):
                getattr(self, method)()
    
    # features


    # визуализация
    def display_waveform(self):
        plt.figure()
        librosa.display.waveshow(self.audio, sr=self.sr)
        plt.title('Waveform for audio')
        plt.show()

    def display_spectrogram(self):
        X = librosa.stft(self.audio)
        X_db = librosa.amplitude_to_db(abs(X))
        plt.figure()
        librosa.display.specshow(X_db, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format="%+2.0f dB")
        plt.title('Spectrogram')
        plt.show()