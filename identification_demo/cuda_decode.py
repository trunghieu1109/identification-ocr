"""Greedy VietOCR decoding without copying every prefix/logit back to the CPU.

Keep upstream width buckets, tokens, EOS stopping and confidence semantics.
Only the experimental namlhn CUDA worker opts into this implementation.
"""
from collections import defaultdict
from types import MethodType
import math

DECODER = "cuda_batched_v1"
MAX_MERGED_BATCH = 12


def translate_on_device(image, model, max_seq_length=128, sos_token=1, eos_token=2):
    import numpy as np
    import torch

    model.eval()
    batch_size = len(image)
    with torch.no_grad():
        memory = model.transformer.forward_encoder(model.cnn(image))
        # Upstream's <= limit generates up to max_seq_length + 1 tokens.
        tokens = torch.empty((max_seq_length+2, batch_size), dtype=torch.long, device=image.device)
        probabilities = torch.empty((max_seq_length+2, batch_size), dtype=image.dtype, device=image.device)
        tokens[0].fill_(sos_token)
        probabilities[0].fill_(1)
        finished = tokens[0] == eos_token
        length = 1
        for step in range(max_seq_length+1):
            if bool(finished.all().item()):
                break
            output, memory = model.transformer.forward_decoder(tokens[:length], memory)
            # Preserve upstream FP32 softmax; only the last position predicts
            # the next token. Keep that selection and the prefix on device.
            output = torch.softmax(output, dim=-1)[:, -1, :]
            values, indices = torch.topk(output, 5, dim=-1)
            tokens[length].copy_(indices[:, 0])
            probabilities[length].copy_(values[:, 0])
            finished |= indices[:, 0] == eos_token
            length += 1
        sentences = tokens[:length].T.cpu().numpy()
        # Upstream averages all non-special positive probabilities, including
        # tokens emitted after an earlier EOS in shorter batch members.
        scores = probabilities[:length].T.cpu().numpy().astype(np.float64)
        scores = scores * (sentences > 3)
        with np.errstate(invalid='ignore', divide='ignore'):
            scores = scores.sum(axis=-1)/(scores > 0).sum(axis=-1)
        return sentences, scores


def decode_memories(language_model, memories, max_seq_length=128, sos_token=1, eos_token=2):
    """Decode differently sized encoder memories together, masking added cells.

    Each original width group keeps its own EOS stop and confidence interval,
    so joining groups does not change which tokens contribute to its score.
    """
    import numpy as np
    import torch

    device = memories[0].device
    sizes = [item.shape[1] for item in memories]
    batch_size = sum(sizes)
    memory = memories[0].new_zeros((max(item.shape[0] for item in memories),batch_size,memories[0].shape[2]))
    padding = torch.ones((batch_size,memory.shape[0]),dtype=torch.bool,device=device)
    offset = 0
    groups = []
    for item, size in zip(memories,sizes):
        memory[:item.shape[0],offset:offset+size] = item
        padding[offset:offset+size,:item.shape[0]] = False
        groups.append((offset,offset+size))
        offset += size
    tokens = torch.empty((max_seq_length+2,batch_size),dtype=torch.long,device=device)
    probabilities = memory.new_empty((max_seq_length+2,batch_size))
    tokens[0].fill_(sos_token)
    probabilities[0].fill_(1)
    finished = tokens[0] == eos_token
    lengths = [None]*len(groups)
    length = 1
    for step in range(0 if sos_token == eos_token else max_seq_length+1):
        mask = language_model.gen_nopeek_mask(length).to(device)
        target = language_model.pos_enc(language_model.embed_tgt(tokens[:length])*math.sqrt(language_model.d_model))
        output = language_model.transformer.decoder(target,memory,tgt_mask=mask,memory_key_padding_mask=padding)
        output = language_model.fc(output.transpose(0,1))
        values, indices = torch.topk(torch.softmax(output,dim=-1)[:,-1,:],5,dim=-1)
        tokens[length].copy_(indices[:,0])
        probabilities[length].copy_(values[:,0])
        finished |= indices[:,0] == eos_token
        length += 1
        complete = finished.cpu().tolist()
        for index,(start,end) in enumerate(groups):
            if lengths[index] is None and all(complete[start:end]):
                lengths[index] = length
        if all(value is not None for value in lengths):
            break
    tokens = tokens[:length].T.cpu().numpy()
    probabilities = probabilities[:length].T.cpu().numpy().astype(np.float64)
    result = []
    for (start,end),stop in zip(groups,lengths):
        stop = length if stop is None else stop
        sentences = tokens[start:end,:stop]
        scores = probabilities[start:end,:stop]*(sentences > 3)
        with np.errstate(divide='ignore',invalid='ignore'):
            scores = scores.sum(axis=-1)/(scores > 0).sum(axis=-1)
        result.append((sentences,scores))
    return result


def predict_batch_on_device(self, images, return_prob=False):
    import torch
    from vietocr.tool.translate import process_input

    buckets, indexes = defaultdict(list), defaultdict(list)
    texts, probabilities = [None]*len(images), [None]*len(images)
    dataset = self.config['dataset']
    for index, image in enumerate(images):
        tensor = process_input(image, dataset['image_height'], dataset['image_min_width'], dataset['image_max_width'])
        buckets[tensor.shape[-1]].append(tensor)
        indexes[tensor.shape[-1]].append(index)
    if not buckets:
        return (texts, probabilities) if return_prob else texts
    self.model.eval()
    with torch.no_grad():
        if len(buckets) == 1:
            width,tensors = next(iter(buckets.items()))
            results = [translate_on_device(torch.cat(tensors,0).to(self.device),self.model)]
        else:
            # Each CNN/encoder still sees exactly the original width group.
            memories = [self.model.transformer.forward_encoder(self.model.cnn(torch.cat(tensors,0).to(self.device)))
                        for tensors in buckets.values()]
            results, chunk, size = [], [], 0
            for memory in memories:
                if chunk and size+memory.shape[1] > MAX_MERGED_BATCH:
                    results.extend(decode_memories(self.model.transformer,chunk))
                    chunk, size = [], 0
                chunk.append(memory)
                size += memory.shape[1]
            if chunk:
                results.extend(decode_memories(self.model.transformer,chunk))
    for width,(sentences,scores) in zip(buckets,results):
        decoded = self.vocab.batch_decode(sentences.tolist())
        for index, text, score in zip(indexes[width], decoded, scores.tolist()):
            texts[index], probabilities[index] = text, score
    return (texts, probabilities) if return_prob else texts


def install_cuda_decoder(predictor):
    if not str(predictor.device).startswith('cuda'):
        raise ValueError('The optimized decoder requires a CUDA predictor')
    if not hasattr(predictor, '_legacy_predict_batch'):
        predictor._legacy_predict_batch = predictor.predict_batch
    predictor.predict_batch = MethodType(predict_batch_on_device, predictor)
