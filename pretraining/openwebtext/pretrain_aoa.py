import os
import sys
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, parent_dir_path)
import random
import logging
from time import time
from dataclasses import dataclass

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from electra_pytorch import Electra_aoa

from openwebtext import arg
from openwebtext.dataset import load_owt, new_tokenizer, wrap_example_builder

logger = logging.getLogger(__name__)

########################################################################################################
## args

@dataclass
class Args:
    data_dir: arg.Str = 'data/openwebtext_features'
    data_vocab_file: arg.Str = 'data/vocab.txt'
    data_n_tensors_per_file: arg.Int = 2048
    data_max_seq_length: arg.Int = 128

    gpu: arg.Int = 0
    gpu_enabled: arg.Bool = True
    gpu_deterministic: arg.Bool = False
    gpu_mixed_precision: arg.Bool = False
    distributed_port: arg.Int = 8888
    distributed_enabled: arg.Bool = False
    distributed_world_size: arg.Int = 4

    model_generator: arg.Str = 'pretraining/openwebtext/small_generator.json'
    model_discriminator: arg.Str = 'pretraining/openwebtext/small_discriminator.json'
    model_mask_prob: arg.Float = 0.15

    opt_lr: arg.Float = 5e-4
    opt_batch_size: arg.Int = 128 // (distributed_world_size if distributed_enabled else 1)
    opt_warmup_steps: arg.Int = 10_000
    opt_num_training_steps: arg.Int = 250_000 # all experiements
    # opt_num_training_steps: arg.Int = 350_000 # last experiment

    step_log: arg.Int = 10
    step_ckpt: arg.Int = 10_000


########################################################################################################
## train

def train(rank, args):

    #######################
    ## distributed
    
    print("#"*230)
    print(args.distributed_enabled)
    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    if args.gpu_enabled:
 
        device = torch.device('cuda:{}'.format(rank))
    else:
        device = torch.device('cpu')

    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0


    #######################
    ## preamble

    set_gpus(rank)
    set_seed(rank)
    set_cuda(deterministic=args.gpu_deterministic)

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=False)

    setup_logging(filename=f'{output_dir}/output.log', console=is_master)


    #######################
    ## dataset

    tokenizer = new_tokenizer(vocab_file=args.data_vocab_file)
    vocab_size = len(tokenizer.vocab)
    ds_train = wrap_example_builder(dataset=load_owt(owt_dir=args.data_dir, n_tensors_per_file=args.data_n_tensors_per_file), vocab=tokenizer.vocab, max_length=args.data_max_seq_length)

    pad_token_id = tokenizer.vocab['[PAD]']
    mask_token_id = tokenizer.vocab['[MASK]']
    cls_token_id = tokenizer.vocab['[CLS]']
    sep_token_id = tokenizer.vocab['[SEP]']

    assert pad_token_id == 0
    assert cls_token_id == 101
    assert sep_token_id == 102
    assert mask_token_id == 103

    # get ID to bin AOA
    import pandas as pd
    aoa = pd.read_csv('/home/osamanatouf/CS557_final_project/electra-pytorch/data/aoa_with_bins.csv').dropna(subset=["Word", "AoA_bin"])
    word2bin = {row['Word']: row['AoA_bin'] for _, row in aoa.iterrows()}
    DEFAULT_HARDEST_BIN = int(aoa["AoA_bin"].max())
    id2bin = np.full(vocab_size, fill_value=DEFAULT_HARDEST_BIN, dtype=np.int64)
    # map AoA bins (words) to token ids in the tokenizer vocab
    mapped = 0
    ###############################################
    # Need to keep an eye on this mapping step
    ##############
    for word, bin_idx in word2bin.items():
        # try exact match, then lowercase as a fallback
        if word.startswith('[') and word.endswith(']'):
            # skip special tokens
            continue
        word = word[2:] if word.startswith('##') else word
        if word in tokenizer.vocab:
            id2bin[tokenizer.vocab[word]] = int(bin_idx)
            mapped += 1
        elif word.lower() in tokenizer.vocab:
            id2bin[tokenizer.vocab[word.lower()]] = int(bin_idx)
            mapped += 1

    print(f"Mapped {mapped} tokens to AoA bins (vocab_size={vocab_size})")
    # print the nums of tokens per bin
    from collections import Counter
    bin_counts = Counter(id2bin)
    print("Token counts per AoA bin:")
    for bin_idx in sorted(bin_counts.keys()):
        print(f"  Bin {bin_idx}: {bin_counts[bin_idx]} tokens")
    
    print(f"Number of bins: {len(set(bin_counts.keys()))}")
          
    ###############################################                                



    def collate_batch(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        input_mask = torch.nn.utils.rnn.pad_sequence([example['input_mask'] for example in examples], batch_first=True, padding_value=pad_token_id)
        segment_ids = torch.nn.utils.rnn.pad_sequence([example['segment_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        return input_ids, input_mask, segment_ids

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    ds_train_loader = iter(cycle(DataLoader(ds_train, batch_size=args.opt_batch_size, collate_fn=collate_batch)))


    #######################
    ## model

    def to_distributed_model(model):
        return model if not args.distributed_enabled else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    def tie_weights(generator, discriminator):
        generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings

    class LogitsAdapter(torch.nn.Module):
        def __init__(self, adaptee):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]

    from transformers import AutoConfig, ElectraForMaskedLM, ElectraForPreTraining

    generator = ElectraForMaskedLM(AutoConfig.from_pretrained(args.model_generator))
    discriminator = ElectraForPreTraining(AutoConfig.from_pretrained(args.model_discriminator))

    tie_weights(generator, discriminator)
    # getting the tensor ready
    id2bin_t = torch.tensor(id2bin, device=device)
    
    emb_weight = discriminator.electra.embeddings.word_embeddings.weight

    ############################################################
    #### AoA Embedding Row Freezer Hook#########################
    ############################################################
    class EmbeddingRowHookFreezer:
        """
        Freezes gradients for embedding rows not in the allowed set.
        
        """
        def __init__(self, embedding_weight: torch.nn.Parameter, vocab_size: int, device: torch.device):
            self.weight = embedding_weight
            # boolean mask indicating which embedding rows are allowed to receive gradients
            self.allowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
            def _hook(grad):
                # ensure mask is on the same device as grad and has a numeric dtype that can be multiplied
                mask = self.allowed_mask.view(-1, 1).to(grad.device)
                mask = mask.to(dtype=grad.dtype)
                return grad * mask
            self.handle = self.weight.register_hook(_hook)
        def set_allowed_rows(self, allowed_mask: torch.Tensor):
            # normalize the incoming mask: bool dtype and same device as the embedding weight
            if allowed_mask.dtype != torch.bool:
                allowed_mask = allowed_mask.bool()
            self.allowed_mask = allowed_mask.to(device=self.weight.device)

        def close(self):
            self.handle.remove()
        


    freezer = EmbeddingRowHookFreezer(emb_weight, vocab_size=vocab_size, device=device)


    freeze_epochs = 3000
    ########################################################################################
    #    warmup_epochs = 5000 for exp 1
    ####################################################################
    warmup_epochs = 7000
    training_epochs = 15000
    # total_bins    = len(unfreeze_schedule)
    total_steps   = 7 * 25000
    assert args.opt_num_training_steps >= total_steps, \
        f"opt_num_training_steps ({args.opt_num_training_steps}) must be >= {total_steps}"
    
    # mask_selector will be passed to the model to ensure only current and previous bins are in use
    def aoa_mask_selector(id2bin_t,):
        state = {"bin_idx": 0, "steps_in_bin": 0}
        def selector(input_ids, attention_mask):
            token_bin = id2bin_t[input_ids]  # (batch_size, seq_len)
            bin_idx = state["bin_idx"]
            bin_steps = state["steps_in_bin"]
            if bin_steps < freeze_epochs:
                p0 = 0.30  # try 0.30; if acc_disc still too high, lower to 0.20
                gate = torch.rand_like(input_ids, dtype=torch.float, device=input_ids.device) < p0
                eligible = ((token_bin < bin_idx) | ((token_bin == bin_idx) & gate))
            elif bin_steps < warmup_epochs + freeze_epochs:
                progress = (bin_steps - freeze_epochs) / float(warmup_epochs + 1e-9)
                gate = torch.rand_like(input_ids, dtype=torch.float, device=input_ids.device) < progress
                # tokens from earlier bins are always eligible; current bin becomes eligible stochastically
                eligible = (token_bin < bin_idx) | ((token_bin == bin_idx) & gate)
            else:
                eligible = token_bin <= bin_idx


            min_elig = 0.05  # 5% of positions per batch
            if eligible.float().mean().item() < min_elig:
                cur = (token_bin == bin_idx) & attention_mask.bool()
                # top up current-bin eligibility with random 5% of remaining current-bin positions
                need = int(min_elig * eligible.numel())
                if need > 0:
                    r = torch.rand_like(eligible, dtype=torch.float)
                    eligible = eligible | ((r < min_elig) & cur)

            # apply attention mask
            mask = eligible & attention_mask.bool()
            return mask

        selector.state  = state
        return selector

    mask_selector= aoa_mask_selector(id2bin_t=id2bin_t)
    current_bin =-1




    model = to_distributed_model(Electra_aoa(
        LogitsAdapter(generator),
        LogitsAdapter(discriminator),
        num_tokens = vocab_size,
        mask_token_id = mask_token_id,
        pad_token_id = pad_token_id,
        mask_prob = args.model_mask_prob,
        mask_ignore_token_ids = [tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
        random_token_prob = 0.0,
        mask_selector=mask_selector).to(device))


    #######################
    ## optimizer
    # original base
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    # smooth scheduler for aoa trainig for all experiments
    def get_curriculum_bin_scheduler(
            optimizer,
            bin_size=25000,           # steps per curriculum bin
            warmup_steps=3000,        # warmup at start of each bin
            last_epoch=-1
        ):

        def lr_lambda(current_step):
            # which bin are we in? how far into that bin?
            bin_idx, step_in_bin = divmod(current_step, bin_size)

            # ✅ Warmup stage
            if step_in_bin < warmup_steps:
                return float(step_in_bin) / float(warmup_steps)

            # ✅ Linear decay stage (within CURRENT bin)
            decay_steps = bin_size - warmup_steps
            return max(0.0, 1.0 - float(step_in_bin - warmup_steps) / float(decay_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


    # original for the base
    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    # adjusting the weight decay for model params during AOA training
    def get_optim_groups(model, emb_weight):
        no_decay = ('bias', 'LayerNorm.weight')
        decay_params, nodecay_params, emb_params = [], [], []
        for n, p in model.named_parameters():
            if p is emb_weight:
                emb_params.append(p)
            elif any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {'params': decay_params,  'weight_decay': 0.1},
            {'params': nodecay_params,'weight_decay': 0.0},
            {'params': emb_params,    'weight_decay': 0.0},  # embeddings: NO DECAY
        ]


    # optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer = torch.optim.AdamW(get_optim_groups(model, emb_weight), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scheduler = get_curriculum_bin_scheduler(optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)
    
   
    # Determine number of bins from AoA map (max bin + 1)
    # total_bins = int(id2bin_t.max().item()) + 1
    total_bins = 7
    print(f"Total AoA bins: {total_bins}")
    total_steps_min = total_bins * 25000
    if args.opt_num_training_steps < total_steps_min and is_master:
        logger.warning(f"opt_num_training_steps ({args.opt_num_training_steps}) < total curriculum steps ({total_steps_min}). "
                       f"Training will stop mid-curriculum.")




    t, steps_s, eta_m = time(), 0., 0
    for step in range(args.opt_num_training_steps+1):
       

        bin_idx , bin_steps = divmod(step, 25000)
        bin_idx = min(bin_idx, total_bins - 1)
    
        if step > 6 * 25000:
            #starting all trian from the 6th bin since the last is what we did not match
            allowed_rows = torch.ones(vocab_size, dtype=torch.bool, device=device)
            stage = "ALL_TRAIN"
        elif bin_steps < freeze_epochs:
            #small fraction of the current bin with the previous bins to avoid stalling the training
            p0 = 0.15
            gate = (torch.rand(vocab_size, device=device) < p0)
            allowed_rows = (id2bin_t < bin_idx) | ((id2bin_t == bin_idx) & gate)
            stage = "FREEZE"            
        elif bin_steps < freeze_epochs + warmup_epochs:
            progress = (bin_steps - freeze_epochs) / float(warmup_epochs + 1e-9)
            progress = max(0.0, min(1.0, progress))

            gate = torch.rand(vocab_size, device=device) < progress
            allowed_rows = (id2bin_t < bin_idx) | ((id2bin_t == bin_idx) & gate)
            stage = "WARMUP"
        else:
            allowed_rows = id2bin_t <= bin_idx
            stage = "TRAINING"

        freezer.set_allowed_rows(allowed_rows)

        mask_selector.state["bin_idx"] = bin_idx
        mask_selector.state["steps_in_bin"] = bin_steps

        def unwrap(m):  # ensures we access the actual Electra_aoa instance (even if DDP wrapped later)
            return getattr(m, "module", m)

        core = unwrap(model)

        if stage == "FREEZE":
            core.mask_prob    = 0.20     # slightly more masking
            core.replace_prob = 0.90 #1.00     # always replace → easier disc detection
            core.temperature  = 1.20 #1.50     # generator samples worse → raises acc_disc
        elif stage == "WARMUP":
            core.mask_prob    = 0.18
            core.replace_prob = 0.90
            core.temperature  = 1.20
        elif stage == "TRAINING":
            core.mask_prob    = 0.15     # back to normal-ish
            core.replace_prob = 0.85
            core.temperature  = 1.00
        else:  # ALL_TRAIN
            core.mask_prob    = 0.15
            core.replace_prob = 0.85




        input_ids, input_mask, segment_ids = next(ds_train_loader)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        assert input_ids.shape[1] <= args.data_max_seq_length

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

        if step > 0 and step % 100 == 0:
            t2 = time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        if step % 200 == 0:
            logger.info(np.array2string(disc_labels[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(np.array2string(disc_pred[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(f"*******[Curriculum] Step {step}: Bin {bin_idx+1}/{total_bins} | Stage: {stage} ")

        if step % 500 == 0:
            frac_allowed = allowed_rows.float().mean().item()
            logger.info(
            f"[AoA] Stage={stage} Bin={bin_idx+1}/{total_bins} "
            f"allowed_vocab_frac={frac_allowed:.3f} "
            f"mask_prob={core.mask_prob:.2f} replace_prob={core.replace_prob:.2f} temp={core.temperature:.2f}"
            )
        
        if step > 0 and step % args.step_ckpt == 0 and is_master:
            discriminator.electra.save_pretrained(f'{args.output_dir}/ckpt/{step}')

########################################################################################################
## preamble

def set_gpus(gpu):

    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(filename, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


########################################################################################################
## main

def main():

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/ckpt', exist_ok=False)
    copy_source(__file__, output_dir)

    # args
    args = arg.parse_to(Args)
    args.output_dir = output_dir+ f"modified_scheduler_{args.opt_num_training_steps}_steps_all_train_after_6th_bin_with_small_precetange_allowed_during_from_current_bin_during_freeze_0.05"


    #FOR EXP4 MAKE SMALL PERCATAGE ALLOWED TO DURING FREEZE TO BE MASKED TO AVOID EMPTINESS WITH ALL_TRAIN AFTER 6TH BIN
    # args.output_dir = output_dir+ f"modified_scheduler_{args.opt_num_training_steps}_steps_all_train_after_6th_bin_with_small_precetange_allowed_during_from_current_bin_during_freeze"

    # for exp3
    # args.output_dir = output_dir+ f"modified_scheduler_{args.opt_num_training_steps}_steps_all_train_after_6th_bin"

    args.exp_id = exp_id

    # distributed
    if args.distributed_enabled:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        torch.multiprocessing.spawn(train, nprocs=args.distributed_world_size, args=(args,))
    else:
        train(rank=args.gpu, args=args)


if __name__ == '__main__':
    main()
