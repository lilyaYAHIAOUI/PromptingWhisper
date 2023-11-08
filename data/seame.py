from pathlib import Path

import numpy as np


import torch
import whisper
import torchaudio
import torchaudio.transforms as at


import opencc


##======== from eval.py and utils.py of https://github.com/HLTCHKUST/ASCEND ========##
import re
import editdistance
import inflect # convert numbers to words
#####
# Common Functions
#####
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]




def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True) # normalization is not required, but since spectrogram is extracted, whether or not normalizing doesn't make a difference
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


class SEAMEDataset(torch.utils.data.Dataset):
    def __init__(self, args, sample_rate, dataset_dir):
        super().__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.tokenizer =  whisper.tokenizer.get_tokenizer(True, language="zh", task="transcribe")
        self.data = []
        self.audio_files = list(Path(dataset_dir).rglob('*.wav'))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        audio_path = self.audio_files[id]

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten(), length=self.args.audio_max_length)
        mel = whisper.log_mel_spectrogram(audio)
        return {
            "audio_path": audio_path,
            "input_mel": mel
        }
    def collate(self, batch):
        audio_paths, input_mels = [], []
        for f in batch:
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])

        input_mels = torch.stack(input_mels, dim=0)

        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths

        return collated_batch   


def get_dataloader(args):

    tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language="zh", task=args.task)
    dataset = SEAMEDataset(args, args.sample_rate,'/kaggle/input/seame-conversation-phase1/dev_man_audio')
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        drop_last=False, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return tokenizer, loader