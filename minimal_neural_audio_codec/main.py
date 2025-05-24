import hydra
from hydra.core.config_store import ConfigStore
from config import NeuralAudioCodecConfig
from datasets import get_dataloader 
from datasets.datasets_config import SplitEnum
from nn_modules import NeuralAudioCodec
from train.trainer import Trainer

# Config store so that Hydra will load our config at minimal-neural-audio-encoder/conf/ as a NeuralAudioCodecConfig class
config_store = ConfigStore.instance()
config_store.store(name="neural_audio_codec_config", node=NeuralAudioCodecConfig)

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: NeuralAudioCodecConfig):

    trainer = Trainer(
        model = NeuralAudioCodec(verbose=cfg.model.verbose), 
        train_dataloader = get_dataloader(cfg.data, SplitEnum.TRAIN), 
        eval_dataloader = get_dataloader(cfg.data, SplitEnum.EVAL), 
        test_dataloader = get_dataloader(cfg.data, SplitEnum.TEST), 
        cfg=cfg.train
    )

    trainer.train()

if __name__ == "__main__":
    main()

# %%
#TODO Eval Loop

#TODO Tensorboard

#TODO Checkpoints

#TODO Generate examples

#TODO Test Loop

#TODO
# We further split the input into chunks of 1 seconds, with an overlap of 10 ms to avoid clicks, and normalize each
# chunk before feeding it to the model, applying the inverse operation on the output of the decoder, adding a
# negligible bandwidth overhead to transmit the scale.
# https://discuss.pytorch.org/t/how-to-normalize-audio-data-in-pytorch/187709

#TODO Loader that goes over 100% of the audios
# -> load every 1sec as an example
# -> chose fixed amout of time for every music? Like load 30s per music