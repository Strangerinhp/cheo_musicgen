import os
current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(current_dir, "ffmpeg_tool", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_path
import json
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import pipeline
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- Global Model Holder for Multiprocessing ---
# This prevents the model from being pickled/reloaded constantly
instrument_classifier = None

def init_worker_model():
    """Initialize the classifier in the worker process."""
    global instrument_classifier
    if instrument_classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        try:
            # Using MIT's Audio Spectrogram Transformer finetuned on AudioSet
            instrument_classifier = pipeline(
                "audio-classification", 
                model="mit/ast-finetuned-audioset-10-10-0.4593",
                device=device
            )
        except Exception as e:
            print(f"Warning: Could not load instrument classifier: {e}")
            instrument_classifier = None

@dataclass
class AudioSegment:
    """Represents a processed audio segment with metadata."""
    segment_id: str
    original_file: str
    lan_dieu: str
    segment_index: int
    start_time: float
    end_time: float
    duration: float
    tempo: float
    energy: float
    key: str
    mode: str
    instruments: List[str]  # Added instruments field
    prompt: str
    output_path: str

@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing."""
    input_dir: str = "cheo"
    output_dir: str = "cheo_processed"
    segment_duration: float = 30.0
    segment_overlap: float = 5.0
    target_sr: int = 32000
    normalize_audio: bool = True
    min_segment_duration: float = 10.0
    audio_extensions: Tuple[str, ...] = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    num_workers: int = 1  # Reduced default workers due to GPU memory usage
    include_artist_in_prompt: bool = True
    
    prompt_template: str = "Làn điệu: {lan_dieu}, {vocals}, {instruments}, tempo: {tempo} BPM, energy: {energy}, key: {key} {mode}"

    # 1. Instrument Map (Keep existing)
    western_to_cheo_map = {
        "guitar": "đàn nguyệt",
        "banjo": "đàn nguyệt", 
        "lute": "đàn nguyệt",
        "violin": "đàn nhị",
        "fiddle": "đàn nhị",
        "flute": "sáo trúc",
        "wind": "tiêu",
        "drum": "trống chèo",
        "percussion": "bộ gõ",
        "cymbal": "chũm chọe"
    }

    # 2. NEW: Vocal Map (Your rules + additions)
    western_vocal_map = {
        "female": "Giọng nữ",
        "woman": "Giọng nữ",
        "male": "Giọng nam",
        "man": "Giọng nam",
        "opera": "Hát cao/Vang",
        "soprano": "Hát cao",
        "chant": "Hát nói/Vỉa",
        "religious": "Hát văn/Vỉa",
        "choir": "Tiếng đế/Tốp ca",
        "group": "Tiếng đế",
        "yodel": "Nảy hạt",
        "vibrato": "Nảy hạt",
        "crying": "Bi ai/Sầu",
        "wailing": "Than khóc",
        "speech": "Nói lối",
        "narration": "Nói lối",
        "laughter": "Hề chèo/Cười",
        "laugh": "Hề chèo/Cười"
    }

class CheoPreprocessor:
    """Preprocessor for Chèo music dataset."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.segments: List[AudioSegment] = []
        
        # Key name mapping
        self.key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.mode_names = ['minor', 'major']
        
    def extract_artist_from_filename(self, filename: str) -> Optional[str]:
        """Extract artist name from filename if present."""
        try:
            if ' - ' in filename:
                artist_part = filename.rsplit(' - ', 1)[-1]
                artist = os.path.splitext(artist_part)[0].strip()
                for suffix in ['(Chèo)', '(chèo)', '(NH Chèo QĐ)']:
                    artist = artist.replace(suffix, '').strip()
                if artist and len(artist) > 1:
                    return artist
        except:
            pass
        return None
    
    def detect_tags(self, audio_array: np.ndarray, sr: int) -> Tuple[List[str], List[str]]:
        """
        Detect tags directly from memory (no file reading needed).
        """
        global instrument_classifier
        if instrument_classifier is None:
            return [], []

        detected_instruments = set()
        detected_vocals = set()
        
        try:
            # FIX: Pass the dictionary directly to pipeline
            # This bypasses ffmpeg file reading entirely
            input_data = {"array": audio_array, "sampling_rate": sr}
            predictions = instrument_classifier(input_data, top_k=8)
            
            for pred in predictions:
                label = pred['label'].lower()
                
                # 1. Check Instruments
                for western_key, vn_value in self.config.western_to_cheo_map.items():
                    if western_key in label:
                        formatted_tag = f"{western_key}-like ({vn_value})"
                        detected_instruments.add(formatted_tag)

                # 2. Check Vocals
                for western_key, vn_value in self.config.western_vocal_map.items():
                    if western_key in label:
                        formatted_tag = f"{western_key} style ({vn_value})"
                        detected_vocals.add(formatted_tag)
                        
        except Exception as e:
            # Printing "e" here will now show useful info if something else breaks
            print(f"Tag detection error: {e}")
            
        return sorted(list(detected_instruments)), sorted(list(detected_vocals))

    def analyze_audio_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract musical features from audio."""
        features = {}
        
        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = round(float(tempo), 1)
        except:
            features['tempo'] = 0.0
        
        # Energy (RMS)
        try:
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = float(np.mean(rms))
            if mean_rms < 0.02:
                features['energy'] = 'soft'
            elif mean_rms < 0.05:
                features['energy'] = 'moderate'
            else:
                features['energy'] = 'strong'
        except:
            features['energy'] = 'moderate'
        
        # Key and mode estimation
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_avg = np.mean(chroma, axis=1)
            key_idx = int(np.argmax(chroma_avg))
            features['key'] = self.key_names[key_idx]
            
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            major_corr = np.corrcoef(chroma_avg, np.roll(major_profile, key_idx))[0, 1]
            minor_corr = np.corrcoef(chroma_avg, np.roll(minor_profile, key_idx))[0, 1]
            
            features['mode'] = 'major' if major_corr > minor_corr else 'minor'
        except:
            features['key'] = 'unknown'
            features['mode'] = ''
        
        return features
    
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to peak at -1dB."""
        peak = np.max(np.abs(y))
        if peak > 0:
            target_peak = 10 ** (-1 / 20)  # -1dB
            y = y * (target_peak / peak)
        return y
    
    def create_prompt(self, lan_dieu: str, features: Dict, instruments: List[str], vocals: List[str], artist: Optional[str] = None) -> str:
        
        # Join the "Bridged" tags
        # Result: "instruments: guitar-like (đàn nguyệt), violin-like (đàn nhị)"
        inst_str = "Instruments: " + ", ".join(instruments) if instruments else "Instruments: traditional ensemble"
        
        # Result: "Vocals: female style (Giọng nữ), chant style (Hát nói)"
        vocal_str = "Vocals: " + ", ".join(vocals) if vocals else "Vocals: traditional singing"

        prompt = self.config.prompt_template.format(
            lan_dieu=lan_dieu,
            vocals=vocal_str,
            instruments=inst_str,
            tempo=int(features['tempo']),
            energy=features['energy'],
            key=features['key'],
            mode=features['mode']
        )
        
        if artist and self.config.include_artist_in_prompt:
            prompt += f", Performer: {artist}"
        
        return prompt
    
    def process_single_file(self, audio_path: Path, lan_dieu: str) -> List[AudioSegment]:
        """Process a single audio file into segments."""
        segments = []
        
        # Initialize the model if running in a worker
        init_worker_model()

        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.config.target_sr, mono=True)
            
            if len(y) == 0:
                return segments
            
            # Extract artist from filename
            artist = self.extract_artist_from_filename(audio_path.name)
            
            # Analyze full track features (fallback)
            full_features = self.analyze_audio_features(y, sr)
            
            # Segment parameters
            segment_samples = int(self.config.segment_duration * sr)
            overlap_samples = int(self.config.segment_overlap * sr)
            hop_samples = segment_samples - overlap_samples
            
            # Normalize
            if self.config.normalize_audio:
                y = self.normalize_audio(y)
            
            # Generate segments
            segment_idx = 0
            start_sample = 0
            
            while start_sample < len(y):
                end_sample = min(start_sample + segment_samples, len(y))
                segment_audio = y[start_sample:end_sample]
                
                segment_duration = len(segment_audio) / sr
                if segment_duration < self.config.min_segment_duration:
                    break
                
                # Analyze segment features
                if segment_duration >= 10:
                    segment_features = self.analyze_audio_features(segment_audio, sr)
                else:
                    segment_features = full_features
                
                # --- Create temporary file for AI Tag Detection ---
                file_hash = hashlib.md5(str(audio_path).encode()).hexdigest()[:8]
                segment_id = f"{file_hash}_{segment_idx:03d}"
                
                safe_lan_dieu = lan_dieu.replace(' ', '_')
                output_filename = f"{safe_lan_dieu}_{segment_id}.wav"
                output_path = Path(self.config.output_dir) / "audio" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save first to allow detector to read it
                sf.write(str(output_path), segment_audio, sr)
                
                instruments, vocals = self.detect_tags(segment_audio, sr)
                
                # Create prompt with both instruments and vocals
                prompt = self.create_prompt(lan_dieu, segment_features, instruments, vocals, artist)
                
                # Create segment metadata
                segment = AudioSegment(
                    segment_id=segment_id,
                    original_file=str(audio_path),
                    lan_dieu=lan_dieu,
                    segment_index=segment_idx,
                    start_time=start_sample / sr,
                    end_time=end_sample / sr,
                    duration=segment_duration,
                    tempo=segment_features['tempo'],
                    energy=segment_features['energy'],
                    key=segment_features['key'],
                    mode=segment_features['mode'],
                    instruments=instruments, # Store list
                    prompt=prompt,
                    output_path=str(output_path)
                )
                
                segments.append(segment)
                segment_idx += 1
                start_sample += hop_samples
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
        
        return segments
    
    def discover_audio_files(self) -> Dict[str, List[Path]]:
        """Discover all audio files organized by làn điệu."""
        audio_files = {}
        input_path = Path(self.config.input_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory '{input_path}' not found.")
            return {}

        for lan_dieu_dir in input_path.iterdir():
            if lan_dieu_dir.is_dir():
                lan_dieu = lan_dieu_dir.name
                files = []
                for ext in self.config.audio_extensions:
                    files.extend(lan_dieu_dir.glob(f"*{ext}"))
                if files:
                    audio_files[lan_dieu] = sorted(files)
        
        return audio_files
    
    def process_all(self) -> None:
        """Process all audio files."""
        print("Discovering audio files...")
        audio_files = self.discover_audio_files()
        
        total_files = sum(len(files) for files in audio_files.values())
        print(f"Found {len(audio_files)} làn điệu with {total_files} total files")
        
        output_path = Path(self.config.output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        
        output_path.mkdir(parents=True)
        (output_path / "audio").mkdir()
        
        all_segments = []
        
        # Use multiprocessing with fewer workers to save VRAM
        # We pass the class method, which internally handles the global model init
        
        # Flatten the list for the executor
        tasks = []
        for lan_dieu, files in audio_files.items():
            for f in files:
                tasks.append((f, lan_dieu))

        print(f"Starting processing with {self.config.num_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # We wrap the process_single_file call
            futures = [executor.submit(self.process_single_file, f, ld) for f, ld in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    segments = future.result()
                    all_segments.extend(segments)
                except Exception as e:
                    print(f"Task failed: {e}")
        
        self.segments = all_segments
        self.save_metadata()
        
        print(f"\nProcessing complete! Total segments: {len(all_segments)}")
    
    def save_metadata(self) -> None:
        """Save metadata."""
        output_path = Path(self.config.output_dir)
        
        # Save full metadata
        metadata = [asdict(seg) for seg in self.segments]
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save training JSONL
        with open(output_path / "train.jsonl", 'w', encoding='utf-8') as f:
            for seg in self.segments:
                entry = {
                    "audio_path": seg.output_path,
                    "prompt": seg.prompt,
                    "duration": seg.duration,
                    "lan_dieu": seg.lan_dieu
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Save stats
        stats = self.compute_statistics()
        with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        stats = {
            'total_segments': len(self.segments),
            'total_duration_hours': sum(s.duration for s in self.segments) / 3600 if self.segments else 0,
            'lan_dieu_distribution': {},
            'instrument_counts': {}
        }
        
        for seg in self.segments:
            stats['lan_dieu_distribution'][seg.lan_dieu] = \
                stats['lan_dieu_distribution'].get(seg.lan_dieu, 0) + 1
            
            for inst in seg.instruments:
                stats['instrument_counts'][inst] = \
                    stats['instrument_counts'].get(inst, 0) + 1
                    
        return stats

def create_train_val_split(metadata_path: str, output_dir: str, val_ratio: float = 0.1, seed: int = 42) -> None:
    """Split the dataset into training and validation sets."""
    np.random.seed(seed)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    by_lan_dieu = {}
    for item in data:
        ld = item['lan_dieu']
        if ld not in by_lan_dieu:
            by_lan_dieu[ld] = []
        by_lan_dieu[ld].append(item)
    
    train_data = []
    val_data = []
    
    for _, items in by_lan_dieu.items():
        np.random.shuffle(items)
        val_size = max(1, int(len(items) * val_ratio))
        val_data.extend(items[:val_size])
        train_data.extend(items[val_size:])
    
    output_path = Path(output_dir)
    with open(output_path / "train_split.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(output_path / "val_split.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We use fewer workers because the instrument detector uses GPU memory
    config = PreprocessConfig(
        input_dir=current_dir,
        output_dir=os.path.join(current_dir, "cheo_processed"),
        num_workers=1 # Set to 1 if you run out of VRAM, 2 or 4 if you have a strong GPU
    )
    
    # Ensure 'start_method' is spawn for compatibility with CUDA in multiprocessing
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    preprocessor = CheoPreprocessor(config)
    preprocessor.process_all()
    
    create_train_val_split(
        metadata_path=os.path.join(config.output_dir, "train.jsonl"),
        output_dir=config.output_dir
    )