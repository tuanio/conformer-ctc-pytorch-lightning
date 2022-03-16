import ctcdecode

labels = [
    "_",
    "<sos>",
    "<eos>",
    " ",
    "'",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


class CTCBeamDecoder:
    def __init__(self, beam_size=100, blank_id=labels.index("_"), kenlm_path=None):
        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels,
            alpha=0.522729216841,
            beta=0.96506699808,
            beam_width=beam_size,
            blank_id=labels.index("_"),
            model_path=kenlm_path,
            log_probs_input=True,
        )
        print("finished loading beam search")

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return "".join([vocab[x] for x in tokens[0:seq_len]])
