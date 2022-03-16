import torch


def make_input(specs):
    """
    specs: (batch, time step, feature)
    """
    batch, time_step, _ = specs.size()
    input_length = torch.full(size=(batch,), fill_value=time_step, dtype=torch.long)
    return specs, input_length


def make_target(transcript):
    encoded = encode_string(transcript)
    target_length = torch.LongTensor([i.size(0) for i in encoded])
    target = torch.nn.utils.rnn.pad_sequence(encoded)
    return target.permute(1, 0), target_length
