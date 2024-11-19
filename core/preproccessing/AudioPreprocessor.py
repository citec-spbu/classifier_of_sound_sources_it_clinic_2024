"""
Класс, реализующий различные методы предобработки аудиосигналов
"""
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import soundfile as sf


class AudioPreprocessor:
    def __init__(self,
                 audio=None,
                 sr=None,
                 offset=0.0,
                 duration=None):
        """
        Конструктор класса
        
        Parameters:
        ----------
        audio: np.ndarray 
            Временной ряд аудиосигнала, default = None
        sr: int
            sampling rate, default = None
        offset: float
            Сдвиг (с какой секунды начинаем аудио), default = 0.0
        duration: float
            Сколько секунд прослушиваем аудио, default = None
        """
        self.sr = sr
        self.audio = audio
        self.offset = offset
        self.duration = duration
        self.mfcc = None

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
        Вернуть аудио (временной ряд и sampling rate)
        """
        return (self.audio, self.sr)
    
    def save_audio(self, output_path):
        """
        Сохраняет аудиосигнал в файл.

        Parameters
        ----------
        output_path: str
            Путь для сохранения файла.
        """
        sf.write(output_path, self.audio, self.sr)

    # обработка аудио
    def detect_noise_section(self, method='amplitude', threshold=0.02, noise_duration=0.5):
        """
        Определяет сегмент шума в аудио различными методами.

        Parameters
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

        Return
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

    def remove_noise(self, noise_start=None, noise_end=None, inplace=False):
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
        inplace: bool
            Преобразовываем ли текущий объект self.audio, 
            default = False
        """
        if noise_start and noise_end:
            noise_sample = self.audio[int(noise_start * self.sr):int(noise_end * self.sr)]
            ts = nr.reduce_noise(y=self.audio,
                                 sr=self.sr,
                                 y_noise=noise_sample)
        else:
            ts = nr.reduce_noise(y=self.audio,
                                 sr=self.sr)
        if inplace:
            self.audio = ts
        
        return AudioPreprocessor(audio=ts, sr=self.sr)
        
    def preemphasis(self, coef=0.97, inplace=False):
        """
        Метод преэмфазиса. Увеличивает амплитуду высокочастотных компонент сигнала, уменьшая влияние низкочастотного шума 
        и улучшая общее качество сигнала для последующей обработки
        
        Parameters:
        -----------
        coef: float
            Коэффиент преэмфазиса. Обычно близок к 1.
        inplace: bool
            Преобразовываем ли текущий объект self.audio, default = False
        """
        ts = librosa.effects.preemphasis(y=self.audio, coef=coef)
        
        if inplace:
            self.audio = ts
        
        return AudioPreprocessor(audio=ts, sr=self.sr)

    def normalize(self, method="peak", coef=0.1, inplace=False):
        """
        Нормализация сигнала
        
        Parameters:
        -----------
        method: str
            Метод масштабирования сигнала: "peak", "rms"
        coef: float
            Коэффициент для масштабирования при использовании "rms"
        inplace: bool
            Преобразовываем ли текущий объект self.audio, default = False
        """
        if method == "peak":
            ts = self.audio / np.max(np.abs(self.audio))
        elif method == "rms":
            ts = self.audio * (coef / np.sqrt(np.mean(self.audio**2)))
        else:
            raise ValueError("Неправильный метод. Используйте 'peak' или 'rms'")

        if inplace:
            self.audio = ts         

        return AudioPreprocessor(audio=ts, sr=self.sr)
    
    def _auto_select_equalize_params(self): 
        """
        Автоматический подбор параметров для эквализации 
        на основе преобразования фурье и анализа частот
        """
        # STFT для анализа частотный компонент 
        S = np.abs(librosa.stft(self.audio))
        
        # Средний спектральный уровень
        avg_spectrum = np.mean(S, axis=1) 
        frequencies = librosa.fft_frequencies(sr=self.sr) 
        
        # Частотные полосы с высоким содержанием энергии 
        threshold = np.median(avg_spectrum) 
        
        # Диапазоны частот для фильтрации 
        high_energy_indices = np.where(avg_spectrum > threshold)[0] 
        lowcut = frequencies[high_energy_indices[0]] 
        highcut = frequencies[high_energy_indices[-1]] 

        # Выполнение проверок
        if lowcut <= 0:
            lowcut = frequencies[1]
        if highcut >= 0.5 * self.sr:
            highcut = 0.5 * self.sr - frequencies[1]

        return lowcut, highcut
    
    def equalize(self, lowcut=None, highcut=None, order=5, inplace=False):
        """
        Эквализация с помощью фильтра Бабичава-Баттерворта.

        Параметры:
        ----------
        lowcut: float
            Нижняя граница частоты для фильтрации, default = None
        highcut: float
            Верхняя граница частоты для фильтрации, default = None
        order: int
            Порядок фильтра, определяет "резкость" фильтра.
        inplace: bool
            Преобразовываем ли текущий объект self.audio, default = False
        """
        if (lowcut is None) or (highcut is None): 
            lowcut, highcut = self._auto_select_equalize_params()
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        ts = lfilter(b, a, self.audio)

        if inplace:
            self.audio = ts
        
        return AudioPreprocessor(audio=ts, sr=self.sr)

    def trim(self, inplace=False):
        """
        Обрезает тишину в начале и конце аудиосигнала.

        Parameters
        ----------
        inplace: bool
            Преобразовываем ли текущий объект self.audio, 
            default = False
        """
        ts, _ = librosa.effects.trim(self.audio)
        
        if inplace:
            self.audio = ts
        
        return AudioPreprocessor(audio=ts, sr=self.sr)

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
    
    # features
    def mfcc(self, n_mfcc=20):
        """
        Вычисление Mel-frequency cepstral coefficients.

        Parameters
        ----------
        n_mfcc: int
            Количество коэффициентов. Можно варьировать для "скармливания" модели
        """
        return librosa.feature.mfcc(y=self.audio,
                                    sr=self.sr,
                                    n_mfcc=n_mfcc)
    
    def zcr(self):
        """
        Вычисление числа пересечений нуля. 
        Для анализа насыщенных сигналов и шумов.
        """
        return librosa.feature.zero_crossing_rate(y=self.audio)
    
    def chroma(self):
        """
        Вычисление распределении энергии в 12-полутоновой шкале. 
        Для музыкального анализа.
        """
        return librosa.feature.chroma_stft(y=self.audio,
                                           sr=self.sr)

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

    # пайплайны обработки
    def preprocessing_custom_pipeline(self, methods):
        for method in methods:
            if hasattr(self, method):
                getattr(self, method)()
    
    def preprocessing_auto_pipeline(self, environment: str = None):
        """
        Пайплайн предобраотки данных в зависимости от характеристик среды
        
        Parameters
        ----------
        environment: str
            Одна из заданных сред: 'city', 'mall', 'forest', 'highway'
            Если не задано, то выбирается автоматически на основе характиристик аудиосигнала
        """
        if environment is None:
            environment = self._analyze_environment()
        
        if environment == 'forest':
            self._forest_preprocessing_pipeline()
        elif environment == 'mall':
            self._mall_preprocessing_pipeline()
        elif environment == 'city':
            self._city_preprocessing_pipeline()
        elif environment == 'highway':
            self._highway_preprocessing_pipeline()
        else:
            raise ValueError()



    def _analyze_environment(self):
        """ 
        Анализ аудио для определения характеристик среды
        """ 
        # Пример анализа: анализ уровня шума и ZCR 
        avg_energy = np.mean(np.square(self.audio)) 
        zcr_value = np.mean(self.zcr()) 
        
        # Большая энергия и ZCR -> город 
        # Средняя энергия и низкий ZCR -> лес
        # Средняя энергия и средний ZCR -> торговый центр 
        

    def _forest_preprocessing_pipeline(self):
        """
        Пайплайн обработки данных в случае, когда среда похожа на лес
        """
        pass
    
    def _highway_preprocessing_pipeline(self):
        """
        Пайплайн обработки данных в случае, когда среда похожа на трассу
        """
        pass

    def _mall_preprocessing_pipeline(self):
        """
        Пайплайн обработки данных в случае, когда среда похожа на торговый центр
        """
        pass

    def _city_preprocessing_pipeline(self):
        """
        Пайплайн обработки данных в случае, когда среда похожа на город
        """
        pass


print(f"Predicted class: {predicted_class}")