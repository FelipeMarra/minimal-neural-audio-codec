import os

import torch
import torchaudio
from torch.utils.data import Dataset

class SNESDataset(Dataset):
    """
        Torch Audio dataset for the SNES-MVDB dataset

        Args:
            root: root path for the SNES-MVDB dataset
    """

    def __init__(self, root:str):
        self.root = root
        self.soundtracks_paths = self._get_soundtracks_paths()

    def __len__(self):
        return len(self.soundtracks_paths)

    def __getitem__(self, index):
        audio_path = self.soundtracks_paths[index]
        audio, sr = torchaudio.load(audio_path)
        audio = self._sample_one_sec(audio, sr)
        return {
            'audio': audio,
            'sr': sr,
            'path': audio_path
        }

    def _get_soundtracks_paths(self):
        soundtracks:list[str] = []

        for game_folder in os.listdir(self.root):
            game_folder_path = os.path.join(self.root, game_folder)
            soundtrack_folder_path = os.path.join(game_folder_path, 'soundtracks')

            if not os.path.exists(soundtrack_folder_path): # TODO: remove after complete downloading the dataset
                continue

            for soundtrack in os.listdir(soundtrack_folder_path):
                soundtrack_path = os.path.join(soundtrack_folder_path, soundtrack)
                soundtracks.append(soundtrack_path)

        return soundtracks

    def _sample_one_sec(self, audio:torch.Tensor, sr:int) -> torch.Tensor:
        _, time = audio.size()
        min_time = time - sr
        rand_starting_point = torch.randint(0, min_time, (1,)).tolist()[0]
        #print(rand_starting_point)
        return audio[:, rand_starting_point : rand_starting_point+sr]
