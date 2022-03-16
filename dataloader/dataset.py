import torch
import torchaudio
from datasubsampling import data_subsampling

spect_func = torchaudio.transforms.Spectrogram()


def transform_audio(audio_pack):
    """
    transform audio pack to spectrogram
    """
    waveform, transcript = audio_pack

    spectrogram = spect_func(waveform)
    # (channel, feature, timestep) -> (channel, timestep, feature)
    spectrogram = spectrogram.permute(0, 2, 1)
    spectrogram = data_subsampling(spectrogram)
    spectrogram = spectrogram.squeeze()

    return spectrogram, transcript


def load_librispeech_item(fileid: str, path: str, ext_audio: str, ext_txt: str):
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return transform_audio((waveform, transcript))


class LibriSpeechDataset(torch.utils.data.Dataset):

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, clean_path=None, other_path=None, data_type="train100"):
        """
        data_type \in ['train100', 'train460', 'train960', 'dev', 'test']
        """

        # clean_path = '../input/librispeech-clean/LibriSpeech/'
        # other_path = '../input/librispeech-500-hours/LibriSpeech/'

        # train 100
        self.list_url = [clean_path + "train-clean-100"]

        if data_type == "train_460":
            self.list_url += [clean_path + "train-clean-360"]
        elif data_type == "train960":
            self.list_url += [clean_path + "train-clean-360"]
        elif data_type == "dev":
            self.list_url = [clean_path + "dev-clean"]
        elif data_type == "test":
            self.list_url = [clean_path + "test-clean"]

        self._walker = []
        for path in self.list_url:
            walker = [
                (str(p.stem), path) for p in Path(path).glob("*/*/*" + self._ext_audio)
            ]
            self._walker.extend(walker)
        self._walker = sorted(self._walker)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n):
        fileid, path = self._walker[n]
        return load_librispeech_item(fileid, path, self._ext_audio, self._ext_txt)
