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
        self.labels = []  # 'S1' ou 'S2'
    
    # 1. Carregar áudio
    def load_audio(self):
        self.signal, self.fs = librosa.load(self.filepath, sr=None)
        print(f"[INFO] Arquivo carregado: {self.filepath}, Fs = {self.fs} Hz, Duração = {len(self.signal)/self.fs:.2f}s")
        return self.signal, self.fs
    
    # 2. Filtro passa-banda
    def bandpass_filter(self, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [self.lowcut/nyq, self.highcut/nyq], btype='band')
        self.filtered = filtfilt(b, a, self.signal)
        print("[INFO] Filtro passa-banda aplicado")
        return self.filtered

    # helper: redimensiona sinal por interp para target_len
    @staticmethod
    def _resize_signal(sig, target_len):
        if len(sig) == target_len:
            return sig
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, sig)

    # 3. Wavelet decomposition + reconstrução de bandas relevantes (d3,d4,d5,a4 testáveis)
    def wavelet_reconstruction(self, wavelet='db6', level=5, bands_to_use=['d4','d5']):
        coeffs = pywt.wavedec(self.filtered, wavelet, level=level)
        # coeffs[0] = cA (aproximação), coeffs[1] = d5, coeffs[2] = d4, coeffs[3] = d3, ...
        # mapping de nomes para índices (considerando level=5)
        name_to_idx = {'a5':0, 'd5':1, 'd4':2, 'd3':3, 'd2':4, 'd1':5}
        target_len = len(self.filtered)
        parts = []
        for band in bands_to_use:
            idx = name_to_idx.get(band, None)
            if idx is None or idx >= len(coeffs):
                continue
            # Mantém cA (coeffs[0]) e o detalhe desejado; zera os outros
            coeffs_band = [None] * len(coeffs)
            coeffs_band[0] = coeffs[0].copy()  # aproximação necessária para reconstrução estável
            coeffs_band[idx] = coeffs[idx]
            rec = pywt.waverec(coeffs_band, wavelet)
            rec = self._resize_signal(rec, target_len)
            parts.append(rec)
        if not parts:
            raise RuntimeError("Nenhuma banda reconstruída: verifique level/wavelet/bands_to_use")
        # média das partes selecionadas
        self.wavelet_recon = np.mean(np.vstack(parts), axis=0)
        print(f"[INFO] Reconstrução Wavelet concluída (bandas: {bands_to_use})")
        return self.wavelet_recon

    # 4. Envelope via Shannon Energy (mais sensível a picos fracos/fortes)
    def compute_shannon_envelope(self, win_size_ms=40):
        x = self.wavelet_recon
        if x is None:
            raise RuntimeError("Execute wavelet_reconstruction() antes")
        # normaliza localmente
        x = x / (np.max(np.abs(x)) + 1e-12)
        energy = x**2
        energy[energy <= 0] = 1e-12
        shannon = -energy * np.log(energy)
        # média móvel (janela em samples)
        win_size = max(1, int(self.fs * win_size_ms / 1000))
        window = np.ones(win_size) / win_size
        env = np.convolve(shannon, window, mode='same')
        env = env / (np.max(env) + 1e-12)
        self.envelope = env
        print("[INFO] Envelope Shannon calculado")
        return self.envelope

    # pequeno K-means 1D para separar intervalos curtos/longos (implementação simples)
    @staticmethod
    def _kmeans_1d(values, k=2, max_iter=100):
        vals = np.array(values).reshape(-1, 1)
        # iniciais: quantis
        cents = np.percentile(vals, np.linspace(0, 100, k+2)[1:-1])
        if len(cents) < k:
            # fallback
            cents = np.linspace(vals.min(), vals.max(), k)
        cents = cents.reshape(k, 1)
        for _ in range(max_iter):
            # assign
            dists = np.abs(vals - cents.reshape(1, -1))
            labels = np.argmin(dists, axis=1)
            new_cents = np.array([vals[labels == i].mean() if np.any(labels==i) else cents[i] for i in range(k)]).reshape(k,1)
            if np.allclose(new_cents, cents):
                break
            cents = new_cents
        return labels, cents.flatten()

    # 5. Detecção robusta de picos + rotulação S1/S2 usando clustering dos intervalos
    def detect_peaks_and_label(self,
                               prominence=0.12,
                               min_distance_ms=100,
                               adaptive_try=True):
        if self.envelope is None:
            raise RuntimeError("Execute compute_shannon_envelope() antes")
        # tentativa de detectar picos; se poucos picos, relaxa parâmetros
        distance = max(1, int(self.fs * min_distance_ms / 1000))
        peaks, props = find_peaks(self.envelope, prominence=prominence, distance=distance)
        # se poucos picos, tentar diminuir prominence (parâmetro adaptativo)
        if adaptive_try and len(peaks) < 4:
            for prom in [prominence*0.8, prominence*0.6, prominence*0.4, prominence*0.2]:
                peaks, props = find_peaks(self.envelope, prominence=prom, distance=int(distance*0.8))
                if len(peaks) >= 4:
                    print(f"[INFO] find_peaks: ajustado prominence para {prom:.3f}")
                    break

        self.peaks = peaks
        if len(peaks) < 2:
            print("[WARN] Poucos picos detectados:", len(peaks))
            self.labels = ['S1' for _ in peaks]
            return peaks, self.labels

        times = peaks / self.fs
        deltas = np.diff(times)  # Δt entre picos consecutivos (s)
        # cluster dos deltas (curto vs longo)
        labels_delta, centers = self._kmeans_1d(deltas, k=2)
        # identifica qual cluster é "curto"
        short_cluster = np.argmin(centers)
        # cria contagem de votos para cada pico
        votes = [dict(S1=0, S2=0) for _ in peaks]

        # regra de pares: se delta_i curto => (p_i, p_{i+1}) = (S1, S2)
        # se delta_i longo  => (p_i, p_{i+1}) = (S2, S1)
        for i, ld in enumerate(labels_delta):
            if ld == short_cluster:
                votes[i]['S1'] += 1
                votes[i+1]['S2'] += 1
            else:
                votes[i]['S2'] += 1
                votes[i+1]['S1'] += 1

        # decide rótulos por maioria de votos
        final_labels = []
        for v in votes:
            if v['S1'] >= v['S2']:
                final_labels.append('S1')
            else:
                final_labels.append('S2')

        # correção simples para forçar alternância quando possível:
        # se houver duas S1 ou duas S2 consecutivas, tenta inverter o segundo se amplitude justificar
        for i in range(1, len(final_labels)):
            if final_labels[i] == final_labels[i-1]:
                # compara amplitudes no envelope: tipicamente S1 mais forte; se o segundo for bem mais fraco, inverter
                a_prev = self.envelope[peaks[i-1]]
                a_cur = self.envelope[peaks[i]]
                if a_prev > 0 and a_cur > 0 and (abs(a_prev - a_cur) / (a_prev + 1e-9)) > 0.35:
                    # inverter o mais fraco
                    if a_cur < a_prev:
                        final_labels[i] = 'S2' if final_labels[i]=='S1' else 'S1'
                    else:
                        final_labels[i-1] = 'S2' if final_labels[i-1]=='S1' else 'S1'
                else:
                    # se não for claro, tenta alternar para manter padrão S1,S2,S1,...
                    final_labels[i] = 'S2' if final_labels[i-1]=='S1' else 'S1'

        self.labels = final_labels

        # estatísticas úteis
        short_intervals = deltas[labels_delta == short_cluster] if len(deltas)>0 else np.array([])
        long_intervals = deltas[labels_delta != short_cluster] if len(deltas)>0 else np.array([])
        if short_intervals.size>0 and long_intervals.size>0:
            ciclo_medio = np.median(short_intervals) + np.median(long_intervals)
            hr_est = 60.0 / ciclo_medio
            print(f"[INFO] picos: {len(peaks)}, "
                f"Δ curto mediano = {np.median(short_intervals):.3f}s, "
                f"Δ longo mediano = {np.median(long_intervals):.3f}s, "
                f"HR≈{hr_est:.1f} bpm")
        else:
            print(f"[INFO] picos: {len(peaks)} (não foi possível calcular estatísticas de intervalos)")


        return self.peaks, self.labels

    # 6. Plotar resultados (atualizado)
    def plot_results(self):
        time = np.arange(len(self.signal)) / self.fs
        fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
        
        axs[0].plot(time, self.signal, color='gray')
        axs[0].set_title("Sinal Original")
        
        axs[1].plot(time, self.filtered, color='blue')
        axs[1].set_title("Após Filtro Passa-Banda (20–800 Hz)")
        
        axs[2].plot(time, self.wavelet_recon, color='green')
        axs[2].set_title("Reconstrução Wavelet (bandas selecionadas)")
        
        axs[3].plot(time, self.signal - self.filtered, color='purple')
        axs[3].set_title("Diferença: Original – Filtrado")
        
        axs[4].plot(time, self.signal - self.wavelet_recon, color='orange')
        axs[4].set_title("Diferença: Original – Reconstrução Wavelet")
        
        axs[5].plot(time, self.envelope, color='red')
        if self.peaks is not None and len(self.peaks)>0:
            axs[5].scatter(self.peaks / self.fs, self.envelope[self.peaks], color='magenta')
            for i, p in enumerate(self.peaks):
                axs[5].text(p/self.fs, self.envelope[p] + 0.04, self.labels[i], fontsize=8, color='black', ha='center')
        axs[5].set_title("Envelope Shannon + Detecção de S1/S2")
        
        for ax in axs:
            ax.grid(True)
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        plt.show()

arquivo = "segment_3.wav"
pcg = PCGProcessor(arquivo)
pcg.load_audio()
pcg.bandpass_filter()
pcg.wavelet_reconstruction(wavelet='db6', level=5, bands_to_use=['d4','d5'])
pcg.compute_shannon_envelope(win_size_ms=40)
pcg.detect_peaks_and_label(prominence=0.12, min_distance_ms=100, adaptive_try=True)
pcg.plot_results()
