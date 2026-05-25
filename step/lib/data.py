# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from datasets import load_from_disk
# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_from_disk('yourpath/LLM/data/wiki'+'/train')
    testdata  = load_from_disk('yourpath/LLM/data/wiki'+'/test')
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # trainenc = tokenizer(" ".join(traindata['page']), return_tensors='pt')
    # testenc = tokenizer("\n\n".join(testdata['page']), return_tensors='pt')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    
    #traindata = load_dataset(path="/home/wufan/llm/lm_eval_wanda/lm-evaluation-harness/wanda/allenai___c4" ,data_dir="allenai--c4-3eabed9451dbe93d/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96", split='train')
    #valdata = load_dataset(path="/home/wufan/llm/lm_eval_wanda/lm-evaluation-harness/wanda/allenai___c4" ,data_dir="allenai--c4-4cb1202b07c0cd0b/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96", split='validation')
    traindata = load_from_disk('yourpath/LLM/data/c4'+'/train')
    valdata = load_from_disk('yourpath/LLM/data/c4'+'/validation') 
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc.input_ids = valenc.input_ids[:, :(256 * seqlen)]
    valenc.attention_mask = valenc.attention_mask[:, :(256 * seqlen)]
    #valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_calib_train_data(name,tokenizer, nsamples=128, seqlen=2048, seed=0, batch_size=1):
    cache_file = (
        f"cache/{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    nsamples += 1 #############################
    import os
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_from_disk('yourpath/LLM/data/c4'+'/train')
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_from_disk('yourpath/LLM/data/wiki/'+'/train')
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_from_disk('yourpath/ptb'+'/train')
    testdata = load_from_disk('yourpath/ptb'+'/test')
   
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)