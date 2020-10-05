# For neural network
from typing import NewType
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

from .utils import PrintUtils as P
from .record import Record

# TODO
# 1. Add recording functions to track progress of accuracy and loss

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
    self.record_keys = ["batch",
                        "epoch",
                        "batch_train_loss",
                        "batch_val_loss",
                        "batch_val_acc",
                        "epoch_val_acc",
                        "epoch_val_loss"]

    # Initializing the record
    self.record_dict = {}
    for attr in self.record_keys:
      self.record_dict[attr] = setattr(self, attr, 'NA')
    self.logs = Record(self.record_keys)

    if not(hasattr(self.net, 'name')):
      self.net.name = 'NEURAL NETWORK'

  def plot(self, params=None):
    self.logs.plot(params)

  def update_record(self):
    """
    Update the record
    """
    for attr in self.record_dict:
       attr_val = getattr(self, attr)
       if th.is_tensor(attr_val):
        #  print("akjdshflaksdjhfalikdfha", attr_val.item())
         attr_val = attr_val.item()
       self.record_dict[attr] = [attr_val]
    self.logs.update(self.record_dict)

  def print_net_params(self):
    """
    Function that describes the trainable parameters of the network
    """
    P.print_message(f"{self.net.name} network summary: ")
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
    P.print_message(f"Number of parameters = {all_params}, Trainable parameters = {trainable_params}")

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
    self.batch_train_loss,_,_ = self.forward_pass(train_data, compute_loss=True)
    self.batch_train_loss.backward()
    self.net.optimizer.step()
    return self.batch_train_loss

  def train_epoch(self, nbatches, train_dataloader, val_dataloader):
    batch_progress_bar = tqdm(range(nbatches), desc="Epoch status", leave=False)
    acc, loss = 0, 0
    for idx in batch_progress_bar:
      train_data = next(iter(train_dataloader))
      val_data = next(iter(val_dataloader))
      # Training batch
      self.batch_train_loss = self.train_batch(train_data)
      # Check validation performance
      self.batch_val_loss,output,target = self.forward_pass(val_data, compute_loss=True)
      self.batch_val_acc = self.net.accuracy(output, target)
      self.batch = self.epoch*self.nbatches + idx
      acc += self.batch_val_acc
      loss += self.batch_val_loss
      self.update_record()
      # Update progress bar
      batch_desc = f"ValAcc:{th.mean(self.batch_val_acc):3.1f}%, \
                    ValLoss:{th.mean(self.batch_val_loss):.{2}}, \
                    TrainLoss:{th.mean(self.batch_train_loss):.{2}}"
      batch_progress_bar.set_description(batch_desc )

    acc /= nbatches
    loss /= nbatches
    return acc, loss

  def train(self, nepochs, nbatches, train_dataloader, val_dataloader):
    self.nepochs = nepochs
    self.nbatches = nbatches
    self.net.to(self.net.dev)
    self.print_net_params()
    P.print_message("Starting training")
    epoch_progress_bar = tqdm(range(nepochs), desc="Epoch status: Acc=?", leave=True)

    for self.epoch in epoch_progress_bar:
      self.epoch_val_acc, self.epoch_val_loss = self.train_epoch(nbatches, train_dataloader, val_dataloader)
      epoch_desc = f"Current accuracy: {self.epoch_val_acc:3.2f}% # Progress: "
      epoch_progress_bar.set_description(epoch_desc)
      self.update_record()
