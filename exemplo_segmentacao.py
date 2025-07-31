from src.preprocessing.segmenter import BatchSegmenter
from pathlib import Path

INPUT_DIR = Path("./data_raw/signal")
OUTPUT_DIR_5S = Path("./data_processed/segments/5s")
OUTPUT_DIR_10S = Path("./data_processed/segments/10s")
SAMPLE_RATE = 2000

batch_segmenter = BatchSegmenter(
    input_dir=INPUT_DIR,
    output_dir_10s=OUTPUT_DIR_10S,
    output_dir_5s=OUTPUT_DIR_5S,
    sample_rate=SAMPLE_RATE
)

batch_segmenter.process_all_segments()