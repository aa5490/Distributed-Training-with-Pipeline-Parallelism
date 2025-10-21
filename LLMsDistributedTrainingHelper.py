"""
This module contains functions used by multiprocessing workers for pipeline parallelism experiments.
"""

import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B
from dataclasses import dataclass

# Model Architecture:
# the transformer modle is adapted from the tutorial with the following features:
# token embeddings layer
# configurable number of transformer decoder layers
# layer normalization
# output projection to vocabulary size
# ModuleDict for layers enables easy splitting across pipeline stages

@dataclass
class ModelArgs:
    dim: int = 768  # changed from 512 to be divisible by 4, 8, and 12 heads
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 10000


class Transformer(nn.Module):
    """transformer model for pipeline parallelism"""
    
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        # Using ModuleDict to enable easy layer deletion for pipeline splitting
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(
                model_args.dim, 
                model_args.n_heads,
                batch_first=True  # for proper batch handling
            )
        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)
    
    def forward(self, tokens: torch.Tensor):
        # handle none layers to enable pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        for layer in self.layers.values():
            h = layer(h, h)
        h = self.norm(h) if self.norm else h
        output = self.output(h).clone() if self.output else h
        return output

# function to split the model across the pipeline stages and is changed form the tutorial to handle varying number of stages and layers
# Each stage gets a subset of transformer layers where first stage keeps the embedding layer and last stage keeps the normalization and 
# output layers (this part same as the tutorial).
def manual_model_split(model: nn.Module, stage_index: int, num_stages: int, device: torch.device) -> PipelineStage:
    """
    split model across pipeline stages manually.
    
    args:
        model: transformer model to split
        stage_index: current pipeline stage (0 to num_stages-1)
        num_stages: total number of pipeline stages
        device: device that holds the stage
    """
    n_layers = len(model.layers)
    layers_per_s = n_layers // num_stages # number of layers per stage
    
    # layers in this stage
    start_l = stage_index * layers_per_s
    end_l = start_l + layers_per_s if stage_index < num_stages - 1 else n_layers
    
    # delete layers not in this stage
    layers_to_del = []
    for i in range(n_layers):
        if i < start_l or i >= end_l:
            layers_to_del.append(str(i))
    
    for i in layers_to_del:
        del model.layers[i]
    
    # stage 0 keeps embeddings, last stage keeps norm and output
    if stage_index != 0:
        model.tok_embeddings = None
    if stage_index != num_stages - 1:
        model.norm = None
        model.output = None
    
    stage = PipelineStage(model, stage_index, num_stages, device)
    return stage 

# this function was not in the tutorial but uses the same logic in the main execution to step through the stages. defined as 
# a separate function to avoid clutter in the worker execution that follows
def run_train_iterations(schedule, x: torch.Tensor, y: torch.Tensor, rank: int, world_size: int, num_iterations: int = 10) -> Dict[str, float]:
    """
    run training iterations and measure performance metrics.
    
    args:
        schedule: pipeline schedule to use
        x: input tensor
        y: target tensor
        rank: process rank (worker ID)
        world_size: total number of workers
        num_iterations: number of iterations to run
    """
    total_toks = x.shape[0] * x.shape[1] * num_iterations
    
    # warmup iterations (not timed)
    for _ in range(2):
        if rank == 0:
            schedule.step(x)
        else:
            losses = []
            schedule.step(target=y, losses=losses)
    
    # timed iterations
    start_t = time.time()
    
    for _ in range(num_iterations):
        if rank == 0:
            schedule.step(x)
        elif rank == world_size - 1:
            losses = []
            output = schedule.step(target=y, losses=losses)
        else:
            # intermediate workers
            schedule.step()
    
    end_t = time.time()
    elapsed_t = end_t - start_t
    
    # calculate throughput (tokens/second)
    tp = total_toks / elapsed_t
    
    return {
        'elapsed_time': elapsed_t,
        'throughput': tp,
        'tokens_processed': total_toks
    }


# this is the main worker process that spins up a worker in a particular CPU core and uses to run the previous training function
# this is totally different from the tutorial as we are simulating a distribited environment by rank, port, etc rather than 
#letting the hardware dictate the environment; however there are some similar elements.
# This is run the same number of times as the number of processes. 
def worker_process(rank, world_size, n_layers, n_heads, schedule_type, 
                   batch_size, seq_length, num_iterations, results_queue):
    """
    Individual worker process for distributed training.
    
    args:
        rank: Process rank (worker ID), range [0, world_size-1]
        world_size: total number of workers/processes
        n_layers: number of transformer layers in the full model
        n_heads: number of attention heads per layer
        schedule_type: pipeline schedule to use ('GPipe', '1F1B', or 'Interleaved1F1B')
        batch_size: training batch size
        seq_length: sequence length for input tokens
        num_iterations: number of training iterations to run
        results_queue: multiprocessing queue for returning metrics to main process
    """
    try:
        # setup distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # init process group
        dist.init_process_group(backend='gloo')
        
        device = torch.device('cpu')
        pp_group = dist.new_group()
        
        # calculate stages per worker for interleaved1f1b
        stages_per_w = 1 # default is 1 stage per worker
        if schedule_type == 'Interleaved1F1B' and n_layers % (world_size * 2) == 0:
            stages_per_w = 2
        
        num_stages = world_size * stages_per_w
        
        # create model args
        model_args = ModelArgs(n_layers=n_layers, n_heads=n_heads)
        
        # create dummy data
        x = torch.randint(0, model_args.vocab_size, (batch_size, seq_length), dtype=torch.long)
        y = torch.randint(0, model_args.vocab_size, (batch_size, seq_length), dtype=torch.long)
        x = x.to(device)
        y = y.to(device)
        
        # define loss function
        def tokenwise_loss_fn(outputs, targets):
            loss_fn = nn.CrossEntropyLoss()
            outputs = outputs.reshape(-1, model_args.vocab_size)
            targets = targets.reshape(-1)
            return loss_fn(outputs, targets)
        
        # create stages for this worker
        # for interleaved: you want to stagger the stages so that each worker has different sets of stages that are not sequential
        stages = []
        for i in range(stages_per_w):
            model = Transformer(model_args)
            stage_idx = rank + (world_size * i)
            stage = manual_model_split(model, stage_idx, num_stages, device)
            model.to(device) # move model to device
            stages.append(stage)
        
        # create schedule based on type
        num_mb = 4 # default is 4 microbatches
        if schedule_type == 'GPipe':
            schedule = ScheduleGPipe(stages[0], n_microbatches=num_mb, loss_fn=tokenwise_loss_fn) # gpipe schedule
        elif schedule_type == '1F1B':
            schedule = Schedule1F1B(stages[0], n_microbatches=num_mb, loss_fn=tokenwise_loss_fn) # 1f1b schedule
        elif schedule_type == 'Interleaved1F1B':
            schedule = ScheduleInterleaved1F1B(stages, n_microbatches=num_mb, loss_fn=tokenwise_loss_fn) # interleaved1f1b schedule
        
        # run training and collect metrics
        metrics = run_train_iterations(schedule, x, y, rank, world_size, num_iterations) # run training iterations
        
        # only last rank returns metrics
        if rank == world_size - 1:
            results_queue.put(metrics) # put metrics in queue
        
        dist.destroy_process_group() # destroy process group to clean up
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put({'error': str(e)})

