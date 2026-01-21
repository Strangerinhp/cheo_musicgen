"""
MusicGen Fine-tuning Script for Vietnamese Chèo Music
======================================================
This script fine-tunes the MusicGen model on preprocessed Chèo music data.

Based on the audiocraft library by Meta AI.
Supports:
- Multiple MusicGen model sizes (small, medium, large)
- Gradient checkpointing for memory efficiency
- Wandb logging for experiment tracking
- Checkpoint saving and resuming
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchaudio
from tqdm import tqdm
import numpy as np

# Hugging Face Hub for model sharing
try:
    from huggingface_hub import HfApi, login, create_repo, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Model upload will be unavailable.")

# Install audiocraft: pip install audiocraft
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

try:
    from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA fine-tuning will be unavailable.")


@dataclass
class TrainingConfig:
    """Configuration for MusicGen fine-tuning."""
    # Data
    data_dir: str = "./cheo_processed"
    train_file: str = "train_split.jsonl"
    val_file: str = "val_split.jsonl"
    
    # Model
    model_name: str = "facebook/musicgen-small"  # small, medium, or large
    
    # Training
    epochs: int = 50
    batch_size: int = 2  # Small due to memory constraints
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Strategy
    strategy: str = "full"  # "full" or "lora"
    
    # LoRA Config
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    
    
    # Audio
    sample_rate: int = 32000
    duration: float = 30.0
    
    # Checkpointing
    output_dir: str = "./cheo_musicgen_finetuned"
    save_every_n_epochs: int = 5
    eval_every_n_steps: int = 100
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "cheo-musicgen"
    
    # Reproducibility
    seed: int = 42
    keep_last_n_checkpoints: int = 3

class CheoDataset(Dataset):
    """
    Dataset for loading Chèo audio segments with Runtime Fixes:
    1. Natural Language Prompt Processing
    2. Silence Detection & Skipping
    3. Peak Normalization
    """
    
    def __init__(
        self,
        data_file: str,
        sample_rate: int = 32000,
        duration: float = 30.0,
        augment: bool = False
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment
        
        self.data_root = os.path.dirname(data_file)

        # Load data entries
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self) -> int:
        return len(self.data)

    def process_prompt(self, raw_prompt_str: str) -> str:
        """Converts 'Key: Value' format to Natural Language."""
        # (Same Logic as before - re-included for completeness)
        import re
        info = {}
        patterns = {
            'style': r'Làn điệu:\s*([^,]+)',
            'vocals': r'Vocals:\s*([^,]+)',
            'instruments': r'Instruments:\s*([^:]+?)(?=(?:, \w+:|$))',
            'tempo': r'tempo:\s*([^,]+)',
            'energy': r'energy:\s*([^,]+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, raw_prompt_str, re.IGNORECASE)
            info[key] = match.group(1).strip() if match else ""

        new_prompt = "A traditional Vietnamese Cheo song"
        if info['style']: new_prompt += f" in the {info['style']} style."
        else: new_prompt += "."

        features = []
        if info['vocals']: features.append(info['vocals'])
        if info['instruments']: features.append(info['instruments'])
        if features: new_prompt += " Features " + ", ".join(features) + " throughout."

        vibe_tags = ["High quality", "clear beat"]
        if info['tempo']: vibe_tags.append(info['tempo'])
        if info['energy']: vibe_tags.append(f"{info['energy']} energy")
        new_prompt += " " + ", ".join(vibe_tags) + "."

        return new_prompt

    def load_audio(self, path: str) -> Optional[torch.Tensor]:
        """
        Load audio with robust Normalization and Silence Check.
        Returns None if file is broken or silent.
        """
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"❌ Corrupt file {path}: {e}")
            return None
        
        # 1. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 2. SILENCE CHECK (The Critical Fix)
        # If RMS is too low, consider it garbage
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms < 0.003:  # Threshold for "effectively silent"
            return None

        # 3. Peak Normalization (The Volume Fix)
        # Ensure audio hits -1dB to 1dB range so EnCodec can hear it
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak  # Normalize to 1.0
            waveform = waveform * 0.95  # Scale to 0.95 (-0.5 dB)
        
        # 4. Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 5. Pad/Cut
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
        
        return waveform.squeeze(0)

    def augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.augment: return waveform
        # Gentle Gain
        if random.random() < 0.3:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Robust getitem that NEVER returns broken data.
        If idx is bad, it recursively tries a random new index.
        """
        max_retries = 10
        current_idx = idx
        
        for _ in range(max_retries):
            entry = self.data[current_idx]
            
            # Path Logic
            raw_path = entry['audio_path']
            filename = raw_path.replace('\\', '/').split('/')[-1]
            corrected_path = os.path.join(self.data_root, 'audio', filename)
            
            if not os.path.exists(corrected_path):
                # Fallback check
                fallback = os.path.join(self.data_root, filename)
                if os.path.exists(fallback):
                    corrected_path = fallback
                else:
                    # File missing -> Try random other file
                    current_idx = random.randint(0, len(self.data) - 1)
                    continue

            # Load & Check Quality
            waveform = self.load_audio(corrected_path)
            
            if waveform is None:
                # File was silent/corrupt -> Try random other file
                # print(f"⚠️ Skipping silent/bad file: {filename}")
                current_idx = random.randint(0, len(self.data) - 1)
                continue
            
            # If we got here, data is GOOD
            waveform = self.augment_audio(waveform)
            original_prompt = entry['prompt']
            natural_prompt = self.process_prompt(original_prompt)
            
            return {
                'audio': waveform,
                'prompt': natural_prompt,
                'lan_dieu': entry.get('lan_dieu', '')
            }
        
        # If we failed 10 times (extremely unlikely), crash gracefully
        raise RuntimeError("Could not find a valid audio file after 10 attempts. Check dataset path!")

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    audios = torch.stack([item['audio'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    lan_dieus = [item['lan_dieu'] for item in batch]
    
    return {
        'audio': audios,
        'prompts': prompts,
        'lan_dieus': lan_dieus
    }


class MusicGenTrainer:
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        self.set_seed(config.seed)
        
        # Initialize model
        print(f"Loading MusicGen model: {config.model_name}")
        self.model = MusicGen.get_pretrained(config.model_name)
        
        # Get the language model from MusicGen
        self.lm = self.model.lm
        self.lm.to(self.device)

        # Force model to float32 to ensure gradients are FP32 for GradScaler
        self.lm.float()
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Apply LoRA if requested
        if config.strategy == "lora":
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed. Cannot use strategy='lora'")
            
            print("Applying LoRA to model...")
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=config.lora_target_modules
            )
            self.model.lm = get_peft_model(self.model.lm, peft_config)
            self.model.lm.print_trainable_parameters()
            self.lm = self.model.lm  # Update reference
        
        trainable_params = [p for p in self.lm.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            import wandb
            wandb.init(project=config.wandb_project, config=vars(config))
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def apply_delay_pattern(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Applies the delay pattern but pads with -100 instead of 0
        so the loss function knows to ignore these spots.
        """
        B, K, T = codes.shape
        
        # Initialize with -100 (The "Ignore Me" flag)
        delayed_codes = torch.full_like(codes, -100)
        
        for k in range(K):
            shift = k
            if shift == 0:
                delayed_codes[:, k, :] = codes[:, k, :]
            else:
                # Shift right, leaving -100 at the start
                delayed_codes[:, k, shift:] = codes[:, k, :-shift]
                
        return delayed_codes

    def compute_loss(self, audio_codes: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Compute training loss with proper masking.
        """
        # 1. Prepare Text Conditions
        attributes, _ = self.model._prepare_tokens_and_attributes(prompts, None)
        
        # 2. Apply Delay Pattern (Pad with -100)
        # audio_codes shape: [B, K, T]
        delayed_codes = self.apply_delay_pattern(audio_codes)
        
        # 3. Create Inputs and Targets
        # We need to feed the model "clean" codes where possible, 
        # but for the targets, we strictly use the delayed/masked version.
        
        # NOTE: When passing inputs to the model, we must replace -100 with 0 
        # (or a valid token) temporarily because the Embedding layer will crash 
        # if it sees -100. The specific value doesn't matter because we 
        # mask the LOSS at these positions.
        
        input_codes = delayed_codes[:, :, :-1].clone()
        target_codes = delayed_codes[:, :, 1:].clone()
        
        # Replace -100 in INPUTS with 0 so the model doesn't crash
        input_codes[input_codes == -100] = 0
        
        # 4. Forward Pass
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            logits = self.lm.forward(
                input_codes,
                conditions=attributes
            )
            
            # Logits: [B, K, T-1, vocab_size]
            # Targets: [B, K, T-1] (Still contains -100s)
            
            loss = 0.0
            for k in range(logits.shape[1]): # Iterate over codebooks
                k_logits = logits[:, k].reshape(-1, logits.shape[-1])
                k_targets = target_codes[:, k].reshape(-1)
                
                # CRITICAL: ignore_index=-100 prevents learning from the padding
                loss += F.cross_entropy(k_logits, k_targets, ignore_index=-100)
            
            # Average loss over codebooks
            loss = loss / logits.shape[1]
        
        return loss
    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.lm, 'transformer'):
            for layer in self.lm.transformer.layers:
                layer.gradient_checkpointing = True
    
    def get_lr_scheduler(self, num_training_steps: int):
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_steps = getattr(self.config, 'warmup_steps', 100)
        
        if warmup_steps == 0:
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=num_training_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        
        scheduler1 = LinearLR(
            self.optimizer, 
            start_factor=0.01, # Start at 1% of max_lr
            end_factor=1.0,    # End at 100% of max_lr
            total_iters=warmup_steps
        )

        scheduler2 = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1 
        )

        return SequentialLR(
            self.optimizer, 
            schedulers=[scheduler1, scheduler2], 
            milestones=[warmup_steps]
        )
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio using MusicGen's compression model."""
        audio = audio.unsqueeze(1)  # Add channel dim: [B, 1, T]
        audio = audio.to(self.device)
        
        with torch.no_grad():
            codes, _ = self.model.compression_model.encode(audio)
        
        return codes  # [B, K, T] where K is codebook count
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler,
        epoch: int
    ) -> float:
        """Train for one epoch."""
        self.lm.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Encode audio to codes
            audio_codes = self.encode_audio(batch['audio'])
            
            # Compute loss
            loss = self.compute_loss(audio_codes, batch['prompts'])
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.lm.parameters(),
                    self.config.max_grad_norm
                )
                
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.config.use_wandb and self.global_step % 10 == 0:
                import wandb
                wandb.log({
                    'train_loss': loss.item() * self.config.gradient_accumulation_steps,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.lm.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            audio_codes = self.encode_audio(batch['audio'])
            loss = self.compute_loss(audio_codes, batch['prompts'])
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        import shutil
        import glob
        
        # config settings
        output_dir = Path(self.config.output_dir)
        keep_n = getattr(self.config, 'keep_last_n_checkpoints', 2) # Default to keeping 2
        
        # --- STEP 1: PRUNE FIRST (The Fix) ---
        # Find all existing checkpoint files
        all_ckpts = sorted(
            glob.glob(str(output_dir / "checkpoint_epoch_*.pt")),
            key=os.path.getmtime
        )
        
        # If we are already at the limit (e.g., have 2, and about to make a 3rd), 
        # delete the oldest one NOW to make space.
        while len(all_ckpts) >= keep_n:
            oldest_ckpt = all_ckpts.pop(0) # Remove from list and delete file
            try:
                os.remove(oldest_ckpt)
                print(f"🧹 Pruned old checkpoint to free space: {os.path.basename(oldest_ckpt)}")
            except OSError as e:
                print(f"   -> Warning: Could not delete {oldest_ckpt}: {e}")


        filename = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = output_dir / filename
        
        print(f"Saving checkpoint to {checkpoint_path}...")
        
        state_dict_to_save = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.lm.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(), # Include optimizer for full finetune
            'val_loss': val_loss,
            'config': vars(self.config)
        }
        
        try:
            torch.save(state_dict_to_save, checkpoint_path)
        except OSError as e:
            print(f"❌ CRITICAL ERROR: Disk full? Failed to save checkpoint: {e}")
            # Optional: try to delete one more and retry?
            return

        best_path = output_dir / "best_model.pt"
        
        if val_loss < self.best_val_loss:
            print(f"🏆 New Best Model! (Loss: {val_loss:.4f} < {self.best_val_loss:.4f})")
            self.best_val_loss = val_loss
            try:
                # remove old best first to ensure space
                if os.path.exists(best_path):
                    os.remove(best_path)
                shutil.copy2(checkpoint_path, best_path)
                print("   -> Copied to best_model.pt")
            except Exception as e:
                print(f"   -> Warning: Could not update best_model.pt: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Robust Loader: Handles both Full (Heavy) and Light (Model-Only) checkpoints.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Load the file
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. Always load the model weights
        self.lm.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully.")
        
        # 2. Conditionally load the optimizer
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored (Resuming training exactly as left off).")
        else:
            print("CHECKPOINT IS 'LIGHT' (No Optimizer found).")
            print("   -> Starting with a FRESH optimizer.")
            print("   -> Learning Rate will reset to 0 and warmup again.")
            
        # 3. Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Return the next epoch
        return checkpoint['epoch'] + 1
    
    def train(self, resume_from: Optional[str] = None) -> None:
        """Main training loop."""
        # Load datasets
        train_dataset = CheoDataset(
            str(Path(self.config.data_dir) / self.config.train_file),
            sample_rate=self.config.sample_rate,
            duration=self.config.duration,
            augment=True
        )
        
        val_dataset = CheoDataset(
            str(Path(self.config.data_dir) / self.config.val_file),
            sample_rate=self.config.sample_rate,
            duration=self.config.duration,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Calculate total training steps
        num_training_steps = (
            len(train_loader) * self.config.epochs // 
            self.config.gradient_accumulation_steps
        )
        
        # Setup scheduler
        scheduler = self.get_lr_scheduler(num_training_steps)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming from epoch {start_epoch}")
        
        # Training loop
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader, scheduler, epoch + 1)
            print(f"Train loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss = self.evaluate(val_loader)
            print(f"Val loss: {val_loss:.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': train_loss,
                    'val_loss': val_loss
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1, val_loss)
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    @torch.no_grad()
    def generate_sample(
        self,
        prompt: str,
        duration: float = 30.0,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """Generate a music sample using the fine-tuned model."""
        self.lm.eval()
        self.model.set_generation_params(duration=duration)
        # Generate
        output = self.model.generate(
            descriptions=[prompt],
            progress=True
        )
        
        # Save if path specified
        if output_path:
            audio_write(
                output_path,
                output[0].cpu(),
                self.model.sample_rate,
                strategy="peak"
            )
            print(f"Saved generated audio to {output_path}")
        
        return output
    
    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload fine-tuned MusicGen for Chèo music"
    ) -> str:
        """
        Push the fine-tuned model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (e.g., 'username/cheo-musicgen')
            token: HF token (uses HF_TOKEN env var if not provided)
            private: Whether to make the repo private
            commit_message: Commit message for the upload
            
        Returns:
            URL of the uploaded model
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
        
        # Get token from environment if not provided
        token = token or os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id, private=private, exist_ok=True)
        except Exception as e:
            print(f"Note: {e}")
        
        # Prepare upload directory
        upload_dir = Path(self.config.output_dir) / "hub_upload"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model card
        readme_path = upload_dir / "README.md"
        
        # Adjust base model info for LoRA
        base_model_info = self.config.model_name
        
        if self.config.strategy == "lora":
            # For LoRA, we need to push the adapters
            print("Pushing LoRA adapters...")
            self.lm.save_pretrained(str(upload_dir))
            
            # Additional tags for LoRA
            extra_tags = "\n- lora\n- peft"
        else:
             # Save the model state dict
            model_path = upload_dir / "model.pt"
            torch.save(self.lm.state_dict(), model_path)
            print(f"Saved model to {model_path}")
            extra_tags = ""

        model_card = ''
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        # Upload folder
        print(f"Uploading to https://huggingface.co/{repo_id}...")
        url = api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            commit_message=commit_message
        )
        
        print(f"Model uploaded successfully!")
        print(f"   URL: https://huggingface.co/{repo_id}")
        
        return f"https://huggingface.co/{repo_id}"


def load_model_from_hub(
    repo_id: str,
    base_model: str = "facebook/musicgen-small",
    device: Optional[str] = None
) -> MusicGen:
    """
    Load a fine-tuned model from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (e.g., 'username/cheo-musicgen')
        base_model: Base MusicGen model name
        device: Device to load model on
        
    Returns:
        Loaded MusicGen model with fine-tuned weights
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    
    from huggingface_hub import hf_hub_download
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download model file
    print(f"Downloading model from {repo_id}...")
    
    # Check if it's a LoRA model (check for adapter_config.json)
    try:
        hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
        is_lora = True
    except:
        is_lora = False

    print(f"Loading base model: {base_model}")
    model = MusicGen.get_pretrained(base_model)

    if is_lora:
        print("Detected LoRA adapters. Loading PeftModel...")
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed but model uses LoRA.")
        
        hf_hub_download(repo_id=repo_id, filename="adapter_model.bin")
        # Load adapters
        # We need to use PeftModel.from_pretrained
        # But MusicGen structure is model.lm
        model.lm = PeftModel.from_pretrained(model.lm, repo_id)
    else:
        print("Loading full fine-tuned weights...")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
        state_dict = torch.load(model_path, map_location=device)
        model.lm.load_state_dict(state_dict)
    
    model.lm.to(device)
    model.lm.eval()
    
    print(f"✅ Model loaded successfully!")
    return model


def generate_samples_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    prompts: List[str],
    duration: float = 30.0,
    model_name: str = "facebook/musicgen-small",
    from_hub: bool = False
) -> None:
    """
    Generate samples from a trained checkpoint or HF Hub model.
    
    Args:
        checkpoint_path: Path to checkpoint file OR HF Hub repo ID if from_hub=True
        output_dir: Directory to save generated samples
        prompts: List of text prompts for generation
        duration: Duration of generated audio in seconds
        model_name: Base model name
        from_hub: If True, checkpoint_path is treated as HF Hub repo ID
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if from_hub:
        # Load from Hugging Face Hub
        model = load_model_from_hub(checkpoint_path, model_name, str(device))
    else:
        # Load from local checkpoint
        model = MusicGen.get_pretrained(model_name)
        
        # Check if it's a LoRA checkpoint directory or a pt file
        if Path(checkpoint_path).is_dir() or (Path(checkpoint_path).parent / "adapter_config.json").exists():
             if not PEFT_AVAILABLE:
                  raise RuntimeError("peft not installed but looks like LoRA checkpoint")
             
             print("Loading local LoRA adapters...")
             # If checkpoint_path is a file (like best_model.pt), we might just rely on state_dict if it was saved that way
             # BUT our save_checkpoint saves adapters to dir in save_pretrained.
             # If user passes the directory:
             if Path(checkpoint_path).is_dir():
                 model.lm = PeftModel.from_pretrained(model.lm, checkpoint_path)
             else:
                 # It's a file, try to load state dict (assuming it was saved as pt for full, or we have limitations)
                 # If we saved best_model.pt for LoRA, it contains state_dict of the wrapped model.
                 # We need to wrap it first?
                 # Actually, if we use PeftModel, load_state_dict works if keys match.
                 # But we need to know config.
                 # Simpler: if it's LoRA, prefer loading from directory using from_pretrained.
                 pass
        
        # Fallback/Standard loading
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Try loading. If LoRA keys are present, we might need to wrap first, but we don't know config here easily
        # unless we saved it.
        # For simplicity in this script, we assume if it's a .pt file it's a full model OR a LoRA state dict 
        # that can be loaded if we knew it was LoRA.
        # But wait, `save_checkpoint` saves everything in .pt.
        # If strategy was LoRA, `state_dict` has "base_model.model..." keys.
        
        try:
             model.lm.load_state_dict(state_dict)
        except RuntimeError as e:
             if "Unexpected key(s) in state_dict: \"base_model.model" in str(e):
                 print("Detected LoRA weights in checkpoint. Wrapping model...")
                 # We need config to wrap. 
                 # If it's in the checkpoint:
                 if 'config' in torch.load(checkpoint_path, map_location='cpu'):
                     saved_config = torch.load(checkpoint_path, map_location='cpu')['config']
                     if saved_config.get('strategy') == 'lora':
                         from peft import LoraConfig, get_peft_model
                         peft_config = LoraConfig(
                            r=saved_config.get('lora_r', 8),
                            lora_alpha=saved_config.get('lora_alpha', 32),
                            lora_dropout=saved_config.get('lora_dropout', 0.1),
                            bias="none",
                            target_modules=saved_config.get('lora_target_modules', ["lin_q", "lin_v"])
                        )
                         model.lm = get_peft_model(model.lm, peft_config)
                         model.lm.load_state_dict(state_dict)
                     else:
                         raise e
                 else:
                     raise e
             else:
                 raise e

        model.lm.to(device)
        model.lm.eval()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.set_generation_params(duration=duration)
    
    # Generate samples
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        print(f"\nGenerating for prompt: {prompt}")
        
        output = model.generate(
            descriptions=[prompt],
            progress=True
        )
        
        output_path = str(Path(output_dir) / f"sample_{i:03d}")
        audio_write(
            output_path,
            output[0].cpu(),
            model.sample_rate,
            strategy="peak"
        )
        print(f"Saved to {output_path}.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MusicGen Fine-tuning for Vietnamese Chèo Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python train.py train --data-dir ./cheo_processed --output-dir /output/cheo_model

  # Train with wandb logging
  python train.py train --data-dir ./cheo_processed --use-wandb --wandb-project cheo-musicgen

  # Resume training from checkpoint
  python train.py train --data-dir ./cheo_processed --resume ./cheo_musicgen_finetuned/best_model.pt --epochs 40 --batch-size 8

  # Generate samples from trained model
  python train.py generate --checkpoint /workspace/cheo/cheo_full_finetune/checkpoint_epoch_10.pt --prompt "Làn điệu: Chinh phụ, tempo: 163 BPM, performer: Bích Được" --base-model facebook/musicgen-medium --duration 30.0 --output-dir ./generated_samples

  # Generate from Hugging Face Hub
  python train.py generate --from-hub username/cheo-musicgen --prompt "Làn điệu: Dao lieu"

  # Push model to Hugging Face Hub
  python train.py push --checkpoint /cheo_lora/best_model.pt --repo-id hoang885002/cheo-musicgen
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # =========================================================================
    # Train subcommand
    # =========================================================================
    train_parser = subparsers.add_parser("train", help="Train/fine-tune MusicGen model")
    
    # Data arguments
    train_parser.add_argument("--data-dir", type=str, required=True,
                              help="Directory containing preprocessed data")
    train_parser.add_argument("--train-file", type=str, default="train_split.jsonl",
                              help="Training data file name (default: train_split.jsonl)")
    train_parser.add_argument("--val-file", type=str, default="val_split.jsonl",
                              help="Validation data file name (default: val_split.jsonl)")
    
    # Model arguments
    train_parser.add_argument("--model-name", type=str, default="facebook/musicgen-small",
                              choices=["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"],
                              help="Base MusicGen model (default: facebook/musicgen-small)")
    
    # Training arguments
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Number of training epochs (default: 50)")
    train_parser.add_argument("--batch-size", type=int, default=2,
                              help="Batch size (default: 2)")
    train_parser.add_argument("--grad-accum", type=int, default=8,
                              help="Gradient accumulation steps (default: 8)")
    train_parser.add_argument("--learning-rate", "--lr", type=float, default=1e-5,
                              help="Learning rate (default: 1e-5)")
    train_parser.add_argument("--weight-decay", type=float, default=0.01,
                              help="Weight decay (default: 0.01)")
    train_parser.add_argument("--warmup-steps", type=int, default=500,
                              help="Warmup steps (default: 500)")
                              
    # LoRA arguments
    train_parser.add_argument("--strategy", type=str, default="full", choices=["full", "lora"],
                              help="Fine-tuning strategy: 'full' or 'lora'")
    train_parser.add_argument("--lora-r", type=int, default=8, help="LoRA r dimension")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Audio arguments
    train_parser.add_argument("--sample-rate", type=int, default=32000,
                              help="Audio sample rate (default: 32000)")
    train_parser.add_argument("--duration", type=float, default=30.0,
                              help="Audio duration in seconds (default: 30.0)")
    
    # Output arguments
    train_parser.add_argument("--output-dir", type=str, default="./cheo_musicgen_finetuned",
                              help="Output directory for checkpoints")
    train_parser.add_argument("--save-every", type=int, default=5,
                              help="Save checkpoint every N epochs (default: 5)")
    
    # Optimization arguments
    train_parser.add_argument("--no-gradient-checkpointing", action="store_true",
                              help="Disable gradient checkpointing")
    train_parser.add_argument("--no-mixed-precision", action="store_true",
                              help="Disable mixed precision training")
    
    # Logging arguments
    train_parser.add_argument("--use-wandb", action="store_true",
                              help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb-project", type=str, default="cheo-musicgen",
                              help="W&B project name (default: cheo-musicgen)")
    
    # Resume training
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to checkpoint to resume from")
    
    # Seed
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed (default: 42)")
    
    # =========================================================================
    # Generate subcommand
    # =========================================================================
    gen_parser = subparsers.add_parser("generate", help="Generate music samples")
    
    gen_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to model checkpoint OR HF Hub repo ID if --from-hub is set")
    gen_parser.add_argument("--from-hub", action="store_true",
                            help="Load model from Hugging Face Hub")
    gen_parser.add_argument("--base-model", type=str, default="facebook/musicgen-small",
                            help="Base model name (default: facebook/musicgen-small)")
    gen_parser.add_argument("--prompt", type=str, action="append", required=True,
                            help="Text prompt(s) for generation (can specify multiple)")
    gen_parser.add_argument("--duration", type=float, default=30.0,
                            help="Duration of generated audio (default: 30.0)")
    gen_parser.add_argument("--output-dir", type=str, default="./generated_samples",
                            help="Output directory for generated samples")
    
    # =========================================================================
    # Push subcommand
    # =========================================================================
    push_parser = subparsers.add_parser("push", help="Push model to Hugging Face Hub")
    
    push_parser.add_argument("--output-dir", type=str, required=True,
                             help="Directory containing trained model (with best_model.pt)")
    push_parser.add_argument("--repo-id", type=str, required=True,
                             help="Hugging Face repository ID (e.g., username/cheo-musicgen)")
    push_parser.add_argument("--private", action="store_true",
                             help="Make the repository private")
    push_parser.add_argument("--token", type=str, default=None,
                             help="HF token (or set HF_TOKEN env var)")
    push_parser.add_argument("--base-model", type=str, default="facebook/musicgen-small",
                             help="Base model that was fine-tuned")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # =========================================================================
    # Execute commands
    # =========================================================================
    
    if args.command == "train":
        # Build config from arguments
        config = TrainingConfig(
            data_dir=args.data_dir,
            train_file=args.train_file,
            val_file=args.val_file,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            sample_rate=args.sample_rate,
            duration=args.duration,
            output_dir=args.output_dir,
            save_every_n_epochs=args.save_every,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
            use_mixed_precision=not args.no_mixed_precision,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            strategy=args.strategy,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
            keep_last_n_checkpoints=1
        )
        
        print("=" * 60)
        print("MusicGen Fine-tuning for Vietnamese Chèo Music")
        print("=" * 60)
        print(f"Data directory: {config.data_dir}")
        print(f"Model: {config.model_name}")
        print(f"Strategy: {config.strategy}")
        print(f"Epochs: {config.epochs}")
        print(f"Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Output: {config.output_dir}")
        print("=" * 60)
        
        # Initialize and train
        trainer = MusicGenTrainer(config)
        trainer.train(resume_from=args.resume)
        
    elif args.command == "generate":
        print("=" * 60)
        print("Generating Chèo Music Samples")
        print("=" * 60)
        
        generate_samples_from_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            prompts=args.prompt,
            duration=args.duration,
            model_name=args.base_model,
            from_hub=args.from_hub
        )
        
    elif args.command == "push":
        print("=" * 60)
        print("Pushing Model to Hugging Face Hub")
        print("=" * 60)
        
        # 1. Initialize Base Model
        # We start with the empty standard model structure
        config = TrainingConfig(
            output_dir=args.output_dir,
            model_name=args.base_model
        )
        trainer = MusicGenTrainer(config)
        
        # 2. Load the Checkpoint File
        best_model_path = Path(args.output_dir) / "best_model.pt"
        if not best_model_path.exists():
            print(f"Error: {best_model_path} not found!")
            sys.exit(1)

        print(f"Loading checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        
        # Extract the parts we need
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        saved_config = checkpoint.get("config", {})

        # 3. Apply LoRA Adapters (The Critical Step)
        # We check if this is a LoRA model, just like your generate script does
        if saved_config.get('strategy') == 'lora' or any("lora" in k for k in state_dict.keys()):
            print("Detected LoRA checkpoint. Merging weights...")
            
            from peft import LoraConfig, get_peft_model
            
            # Reconstruct the config using the saved parameters
            # If parameters are missing, defaults to r=8 (standard for musicgen-small)
            peft_config = LoraConfig(
                r=saved_config.get('lora_r', 8),
                lora_alpha=saved_config.get('lora_alpha', 32),
                lora_dropout=saved_config.get('lora_dropout', 0.1),
                bias="none",
                target_modules=saved_config.get('lora_target_modules', ["lin_q", "lin_v"])
            )
            
            # Wrap the base model with PEFT. 
            # Now trainer.lm expects keys like "base_model.model..." and "lora_A"
            trainer.lm = get_peft_model(trainer.lm, peft_config)
            
            # Load the weights
            # strict=False is safer here because PEFT sometimes renames keys slightly
            trainer.lm.load_state_dict(state_dict, strict=False)
            
            # 4. MERGE AND UNLOAD
            # This converts the LoRA model back into a standard MusicGen model
            # It calculates: Weight_Final = Weight_Base + (Lora_B * Lora_A)
            trainer.lm = trainer.lm.merge_and_unload()
            
            print("Successfully merged LoRA adapters into base model.")
            
        else:
            # If it wasn't LoRA, just load normally
            trainer.lm.load_state_dict(state_dict)

        # 5. Push to Hub
        print(f"Pushing to repo: {args.repo_id}")
        trainer.push_to_hub(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private
        )
