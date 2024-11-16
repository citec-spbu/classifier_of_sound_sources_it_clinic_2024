"""
Класс, реализующий различные методы предобработки аудиосигналов, 
а также комбинацию различных методов
"""
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt


class AudioPreprocessor:
    def __init__(self,
                 sr=None,
                 audio=None,
                 offset=0.0,
                 duration=None):
        """
        Конструктор класса
        
        Parameters
        ----------
        sr: int
            sampling rate, default = None
        audio: np.ndarray 
            временной ряд аудиосигнала, default = None
        offset: float
            сдвиг (с какой секунды начинаем аудио), default = 0.0
        duration: float
            сколько секунд прослушиваем аудио, default = None
        """
        self.sr = sr
        self.audio = audio
        self.offset = offset
        self.duration = duration

    def load_audio(self, file_path: str):
        """
        Загрузка аудиосигнала, если он не был передан в конструкторе
        
        Parameters
        ----------
        file_path: str
            путь к файлу
        """
        self.audio, self.sr = librosa.load(file_path, sr=self.sr, 
                                           offset=self.offset, duration=self.duration)
    def return_audio(self):
        """
        Вернуть аудио (например, для просмотра его состояния после преобразований)
        """
        return self.audio

    def detect_noise(self, ):
        """
        Детекция шума в аудиосигнале
        """
        pass

    def remove_noise(self):
        """
        Удаление шума
        """
        self.audio = nr.reduce_noise(y=self.audio,
                                     sr=self.sr)
        self.audio = librosa.effects.preemphasis(self.audio)

    def normalize(self):
        self.audio = self.audio / np.max(np.abs(self.audio))

    def equalize(self):
        # Применяем простую эквализацию
        pass  # Здесь можете добавить вашу реализацию эквализации

    def display_waveform(self):
        plt.figure()
        librosa.display.waveshow(self.audio, sr=self.sr)
        plt.title('Waveform')
        plt.show()

    def display_spectrogram(self):
        X = librosa.stft(self.audio)
        X_db = librosa.amplitude_to_db(abs(X))
        plt.figure()
        librosa.display.specshow(X_db, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format="%+2.0f dB")
        plt.title('Spectrogram')
        plt.show()

    def preprocessing_pipeline(self, methods):
        for method in methods:
            if hasattr(self, method):
                getattr(self, method)()


# Пример использования
audio_file_path = 'path_to_audio_file.wav'
processor = AudioPreprocessor()
processor.load_audio(audio_file_path)
processor.process_pipeline(['remove_noise', 'normalize'])
processor.display_waveform()
processor.display_spectrogram()