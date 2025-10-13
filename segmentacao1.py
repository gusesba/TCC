import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
from scipy.signal import butter, filtfilt, find_peaks

class PCGProcessor:
    def __init__(self, filepath, lowcut=20, highcut=800, fs=None):
        self.filepath = filepath
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.signal = None
        self.filtered = None
        self.wavelet_recon = None
        self.envelope = None
        self.peaks = None
        self.labels = []  # lista com marcação S1 / S2
    
    # 1. Carregar áudio
    def load_audio(self):
        self.signal, self.fs = librosa.load(self.filepath, sr=None)
        print(f"[INFO] Arquivo carregado: {self.filepath}, Fs = {self.fs} Hz, Duração = {len(self.signal)/self.fs:.2f}s")
        return self.signal, self.fs
    
    # 2. Filtro passa-banda
    def bandpass_filter(self):
        nyq = 0.5 * self.fs
        b, a = butter(4, [self.lowcut/nyq, self.highcut/nyq], btype='band')
        self.filtered = filtfilt(b, a, self.signal)
        print("[INFO] Filtro passa-banda aplicado")
        return self.filtered
    
        # 3. Wavelet decomposition + reconstrução de bandas relevantes
    def wavelet_reconstruction(self, wavelet='db6', level=5):
        coeffs = pywt.wavedec(self.filtered, wavelet, level=level)

        # Reconstrução d4
        coeffs_d4 = [c.copy() if i == 0 else None for i, c in enumerate(coeffs)]
        coeffs_d4[2] = coeffs[2]  # mantém apenas d4 + cA
        d4 = pywt.waverec(coeffs_d4, wavelet)

        # Reconstrução d5
        coeffs_d5 = [c.copy() if i == 0 else None for i, c in enumerate(coeffs)]
        coeffs_d5[1] = coeffs[1]  # mantém apenas d5 + cA
        d5 = pywt.waverec(coeffs_d5, wavelet)

        # Ajusta os comprimentos
        min_len = min(len(d4), len(d5), len(self.filtered))
        self.wavelet_recon = (d4[:min_len] + d5[:min_len]) / 2.0

        print("[INFO] Reconstrução Wavelet concluída (bandas médias d4+d5)")
        return self.wavelet_recon


    
    # 4. Envelope via Shannon Energy
    def compute_shannon_envelope(self, win_size=100):
        # Shannon energy = -x^2 * log(x^2)
        x = self.wavelet_recon
        x = x / np.max(np.abs(x))
        energy = x**2
        energy[energy <= 0] = 1e-12  # evitar log(0)
        shannon = -energy * np.log(energy)
        
        # média móvel
        window = np.ones(win_size) / win_size
        self.envelope = np.convolve(shannon, window, mode="same")
        self.envelope = self.envelope / np.max(self.envelope)
        print("[INFO] Envelope Shannon calculado")
        return self.envelope
    
    # 5. Detecção de picos → candidatos a S1 e S2
    def detect_peaks(self, distance_ms=200):
        distance = int(self.fs * distance_ms / 1000)
        self.peaks, _ = find_peaks(self.envelope, height=0.3, distance=distance)
        
        # Classificação simples S1 / S2 baseada no intervalo
        self.labels = []
        for i, p in enumerate(self.peaks):
            if i == 0:
                self.labels.append("S1")
            else:
                interval = (self.peaks[i] - self.peaks[i-1]) / self.fs
                # regra: sístole (S1–S2) é mais curta que diástole (S2–S1)
                if interval < 0.35:  # limiar aproximado
                    self.labels.append("S2")
                else:
                    self.labels.append("S1")
        print(f"[INFO] {len(self.peaks)} picos detectados e rotulados como S1/S2")
        return self.peaks, self.labels
    
    # 6. Plotar resultados
    def plot_results(self):
        time = np.arange(len(self.signal)) / self.fs
        fig, axs = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
        
        axs[0].plot(time, self.signal, color='gray')
        axs[0].set_title("Sinal Original")
        
        axs[1].plot(time, self.filtered, color='blue')
        axs[1].set_title("Após Filtro Passa-Banda (20–800 Hz)")
        
        axs[2].plot(time, self.wavelet_recon, color='green')
        axs[2].set_title("Reconstrução Wavelet (bandas médias)")
        
        axs[3].plot(time, self.signal - self.filtered, color='purple')
        axs[3].set_title("Diferença: Original – Filtrado")
        
        axs[4].plot(time, self.signal - self.wavelet_recon, color='orange')
        axs[4].set_title("Diferença: Original – Reconstrução Wavelet")
        
        axs[5].plot(time, self.envelope, color='red')
        for i, p in enumerate(self.peaks):
            axs[5].scatter(time[p], self.envelope[p], color='magenta')
            axs[5].text(time[p], self.envelope[p]+0.05, self.labels[i], fontsize=8, color='black')
        axs[5].set_title("Envelope Shannon + Detecção de S1/S2")
        
        for ax in axs:
            ax.grid(True)
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    arquivo = "segment_7.wav"  
    pcg = PCGProcessor(arquivo)
    pcg.load_audio()
    pcg.bandpass_filter()
    pcg.wavelet_reconstruction()
    pcg.compute_shannon_envelope()
    pcg.detect_peaks()
    pcg.plot_results()
