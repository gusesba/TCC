from enum import Enum
from typing import List
import numpy as np
import os
import soundfile as sf
from pathlib import Path
import os

class SegmentMode(Enum):
    FIVE_SECONDS = 0
    TEN_SECONDS = 1

class Segmenter:
    def __init__(self, mode: SegmentMode, sample_rate: int) -> None:
        self.mode = mode
        self.sample_rate = sample_rate

    def _get_segment_size(self) -> int:
        if self.mode == SegmentMode.FIVE_SECONDS:
            return 5 * self.sample_rate
        elif self.mode == SegmentMode.TEN_SECONDS:
            return 10 * self.sample_rate
        else:
            raise ValueError("Modo de segmentação inválido.")
        
    def segment(self, signal: np.ndarray) -> List[np.ndarray]:
        segment_samples = self._get_segment_size()
        return [
            signal[i:i + segment_samples]
            for i in range(0, len(signal), segment_samples)
            if len(signal[i:i + segment_samples]) == segment_samples
        ]
    
    def segment_wav_file(self, input_path: str, output_dir: str, output_name: str) -> None:
        signal, sr = sf.read(input_path)

        if sr != self.sample_rate:
            raise ValueError(f"Sample rate do arquivo ({sr}) difere do esperado ({self.sample_rate})")

        if self.mode == SegmentMode.FIVE_SECONDS:
            segments = self.segment(signal)

            # Cria diretório para armazenar segmentos
            target_folder = os.path.join(output_dir, output_name)
            os.makedirs(target_folder, exist_ok=True)

            for idx, segment in enumerate(segments, start=1):
                output_file = os.path.join(target_folder, f"{idx}.wav")
                sf.write(output_file, segment, samplerate=self.sample_rate)

        elif self.mode == SegmentMode.TEN_SECONDS:
            # Apenas salva o arquivo com novo nome
            output_file = os.path.join(output_dir, f"{output_name}.wav")
            os.makedirs(output_dir, exist_ok=True)
            sf.write(output_file, signal, samplerate=self.sample_rate)

        else:
            raise ValueError("Modo de segmentação inválido.")


class BatchSegmenter:
    def __init__(self, input_dir: Path, output_dir_10s: Path, output_dir_5s: Path, sample_rate: int = 2000):
        self.input_dir = input_dir
        self.output_dir_10s = output_dir_10s
        self.output_dir_5s = output_dir_5s
        self.segments = [f"segment_{i}.wav" for i in range(3, 10)]
        self.segmenter_10s = Segmenter(mode=SegmentMode.TEN_SECONDS, sample_rate=sample_rate)
        self.segmenter_5s = Segmenter(mode=SegmentMode.FIVE_SECONDS, sample_rate=sample_rate)

    def process_all_segments(self):
        for participant_number in range(1, 79):  # Participantes de 1 a 78
            folder_name = str(participant_number)
            participant_dir = self.input_dir / folder_name

            for segment_file in self.segments:
                input_file = participant_dir / segment_file

                if not input_file.exists():
                    print(f"[!] Arquivo não encontrado: {input_file}")
                    continue

                segment_name = segment_file.replace(".wav", "")

                # ------------------- Segmento 10s ------------------- #
                output_path_10s = self.output_dir_10s / folder_name
                output_file_10s = output_path_10s / f"{segment_name}.wav"
                if output_file_10s.exists():
                    print(f"[i] Segmento 10s já existe: {output_file_10s}")
                else:
                    self.segmenter_10s.segment_wav_file(
                        input_path=str(input_file),
                        output_dir=str(output_path_10s),
                        output_name=segment_name
                    )

                # ------------------- Segmento 5s ------------------- #
                output_path_5s = self.output_dir_5s / folder_name / segment_name
                file_1 = output_path_5s / "1.wav"
                file_2 = output_path_5s / "2.wav"

                if file_1.exists() and file_2.exists():
                    print(f"[i] Segmentos 5s já existem: {file_1}, {file_2}")
                else:
                    self.segmenter_5s.segment_wav_file(
                        input_path=str(input_file),
                        output_dir=str(output_path_5s),
                        output_name=""
                    )