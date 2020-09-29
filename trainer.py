# For neural network
from typing import NewType
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

# TODO
# 1. Add plotting functions to track progress of accuracy and loss

# Add callbacks to enable
# 1. Early stopping
# 2. Learning rate scheduler
# 3. Save model 
# 4. Reload model
# 5. Change generator properties

class Trainer(object):
  """
  Base NN class implements training functions
  """
  def __init__(self, net):
    super().__init__()
    self.net = net 

  def print_line(self):
    print("-"*80)

  def print_message(self, message):
    self.print_line()
    print(message)
    self.print_line()

  def print_net_params(self):
    self.print_message(f"{self.net.name} network summary: ")
    params_table = PrettyTable(["Name", "#", "Trainable"])
    all_params = 0
    trainable_params = 0
    for p in self.net.named_parameters():
      num_params = p[1].numel()
      params_table.add_row([p[0], num_params, p[1].requires_grad])
      all_params += num_params
      if p[1].requires_grad:
        trainable_params += num_params
    print(params_table)
    self.print_message(f"Number of parameters = {all_params}, Trainable parameters = {trainable_params}")

  def forward_pass(self, data, compute_loss=True):
    x, t = data
    x = x.to(self.net.dev)
    t = t.to(self.net.dev)
    y = self.net.forward(x)
    if compute_loss:
      loss = self.net.lossfn(y, t)
      return loss, y, t
    return y, t

  def train_batch(self, train_data):
    self.net.optimizer.zero_grad()
    loss,_,_ = self.forward_pass(train_data, compute_loss=True)
    loss.backward()
    self.net.optimizer.step()
    return loss

  def train_epoch(self, nbatches, train_dataloader, val_dataloader):
    batch_progress_bar = tqdm(range(nbatches), desc="Batch status", leave=False)
    acc = 0
    for _ in batch_progress_bar:
      train_data = next(iter(train_dataloader))
      val_data = next(iter(val_dataloader))
      train_loss = self.train_batch(train_data)
      val_loss,output,target = self.forward_pass(val_data, compute_loss=True)
      batch_desc = f"Batch status V:{th.mean(val_loss):.{3}}, T:{th.mean(train_loss):.{3}}"
      batch_progress_bar.set_description(batch_desc )
      acc += self.net.accuracy(output, target)
    acc /= nbatches
    return acc

  def train(self, nepochs, nbatches, train_dataloader, val_dataloader):
    self.net.to(self.net.dev)
    self.print_net_params()
    self.print_message("Starting training")
    epoch_progress_bar = tqdm(range(nepochs), desc="Epoch status: Acc=?", leave=True)

    for _ in epoch_progress_bar:
      acc = self.train_epoch(nbatches, train_dataloader, val_dataloader)
      epoch_desc = f"Current accuracy: {acc:3.2f}% # Progress: "
      epoch_progress_bar.set_description(epoch_desc)