import argparse
import yaml
import pytorch_lightning as pl
from .model.encoder import ConformerCTC
from .dataloader.datamodule import LibrispeechDataModule
from .dataloader.dataset import LibriSpeechDataset


def main():

    parser = argparse.ArgumentParser(description="Process Conformer")
    parser.add_argument("--config-path", type=str)

    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except Exception as e:
            print(f"Có lỗi xảy ra khi đọc file {args.config_path}")
            print(str(e))

    train_dataset = LibriSpeechDataset(
        clean_path=configs["clean_path"],
        other_path=configs["other_path"],
        data_type="train460",
    )
    val_dataset = LibriSpeechDataset(
        clean_path=configs["clean_path"],
        other_path=configs["other_path"],
        data_type="dev",
    )
    test_dataset = LibriSpeechDataset(
        clean_path=configs["clean_path"],
        other_path=configs["other_path"],
        data_type="test",
    )

    model = ConformerCTC(batch_size=configs["batch_size"])
    data_module = LibrispeechDataModule(train_dataset, val_dataset, test_dataset)

    logger = pl.loggers.TensorBoardLogger(
        "tb_logs_dir", name="confomer-ctc-tensorboard"
    )
    trainer = pl.Trainer(gpus=configs['num_gpus'], logger=logger, max_epochs=configs["max_epochs"])
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
