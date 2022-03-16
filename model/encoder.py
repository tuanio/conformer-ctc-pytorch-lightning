import torch
import pytorch_lightning as pl
import torch.nn.functionals as F
from decode import CTCBeamDecoder


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.specaug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
        )

    def forward(self, x):
        return self.specaug(x)


class ConformerCTC(pl.LightningModule):
    def __init__(
        self, lr: float, kenlm_path: str, batch_size: int, conformer_hyp: dict
    ):
        super(ConformerCTC, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.specaug = SpecAugment()
        self.encoder = Conformer(**conformer_hyp)
        self.decoder = CTCBeamDecoder(kenlm_path=kenlm_path)

    def forward(self, inputs, input_length):
        inputs = self.specaug(inputs)
        outputs, output_lengths = self.encoder(inputs, input_length)
        return outputs, output_lengths

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98),
            lr=self.lr,
            eps=1e-9,
            weight_decay=1e-6,
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        specs, trans = train_batch

        inputs, input_lengths = make_input(specs)
        targets, target_lengths = make_target(trans)

        outputs, output_lengths = self.forward(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2)

        loss = F.ctc_loss(
            outputs, targets, output_lengths, target_lengths, zero_infinity=True
        )

        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, train_batch, batch_idx):
        specs, trans = train_batch

        inputs, input_lengths = make_input(specs)
        targets, target_lengths = make_target(trans)

        outputs, output_lengths = self.forward(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2)

        loss = F.ctc_loss(
            outputs, targets, output_lengths, target_lengths, zero_infinity=True
        )

        outputs = outputs.permute(1, 0, 2)
        decoded = [self.decoder(out.unsqueeze(0)) for out in outputs]

        avg_wer = jiwer.wer(trans, decoded)

        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("avg_wer", avg_wer, on_epoch=True, batch_size=self.batch_size)
        return {"loss": loss, "avg_wer": avg_wer}
