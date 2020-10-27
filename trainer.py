# For neural network
from typing import NewType
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from prettytable import PrettyTable
from collections import OrderedDict

from .utils import PrintUtils as P
from .record import Record

# Enable
# 1. Early stopping
# 2. Save model 
# 3. Reload model
# 4. Change generator properties

class Trainer(object):
  """
  Base NN class implements training functions
  """
  def __init__(self, net):
    super().__init__()
    self.net = net
    self.record_dict = {"batch":0,
                        "epoch":0,
                        "batch_train_loss":0,
                        "batch_val_loss":0,
                        "batch_val_acc":0,
                        "epoch_val_acc":0,
                        "epoch_val_loss":0,
                        "lr":self.record_lr()}

    # # Initializing the record
    for attr in self.record_dict.keys():
      setattr(self, attr, self.record_dict[attr])
    self.logs = Record(self.record_dict.keys())

    if not(hasattr(self.net, 'name')):
      self.net.name = 'NEURAL NETWORK'

  def plot(self, params=None):
    self.logs.plot(params)

  def record_lr(self):
    l = 0
    for idx,p in enumerate(self.net.optimizer.param_groups):
      l += p['lr']
    self.lr = l/(idx+1)

  def update_record(self):
    """
    Update the record
    """
    self.record_lr()

    for attr in self.record_dict:
       attr_val = getattr(self, attr)
       if th.is_tensor(attr_val):
         attr_val = attr_val.item()
       self.record_dict[attr] = [attr_val]
    self.logs.update(self.record_dict)
    # self.logs.savenet(net=self.net, net_name="net")

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
    # Below is more efficient than self.net.optimizer.zero_grad()
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    for param in self.net.parameters():
      param.grad = None
    self.batch_train_loss,_,_ = self.forward_pass(train_data, compute_loss=True)
    self.batch_train_loss.backward()
    # Clipping gradients
    if hasattr(self.net, "clip_norm"):
      nn.utils.clip_grad_norm_(self.net.parameters(), 
                               self.net.clip_norm)
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

      with th.no_grad():
        self.net.eval()
        # Check validation performance with grads disabled
        self.batch_val_loss,output,target = self.forward_pass(val_data, compute_loss=True)
        self.net.train()

      self.batch_val_acc = self.net.accuracy(output, target)
      self.batch = self.epoch*self.nbatches + idx
      acc += self.batch_val_acc.detach()
      loss += self.batch_val_loss.detach()
      # Update progress bar
      batch_desc = f"ValAcc:{th.mean(self.batch_val_acc):3.1f}%, \
                    ValLoss:{th.mean(self.batch_val_loss):.{2}}, \
                    TrainLoss:{th.mean(self.batch_train_loss):.{2}}"
      batch_progress_bar.set_description(batch_desc )
      self.update_record()

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
      # Update the learning rate if val_loss is stagnant
      if hasattr(self.net, "scheduler"):
        self.net.scheduler.step(self.epoch_val_loss)
      epoch_desc = f"Current accuracy: {self.epoch_val_acc:3.2f}% # Progress: "
      epoch_progress_bar.set_description(epoch_desc)
    
      self.logs.savelogs()
      self.plot(self.record_dict.keys())
    P.print_message("Training complete!")

  def test(self, test_dataloader):
    self.net.to(self.net.dev)
    acc, loss, idx = 0,0,0
    # P.print_message(f"Testing started")
    nbatches = int(len(test_dataloader.dataset)/test_dataloader.batch_size)
    test_progress_bar = tqdm((range(nbatches)), desc="Testing: Acc=?", leave=True)

    self.net.eval() # put in eval mode
    with th.no_grad():
      # for idx,test_data in enumerate(test_dataloader):
      for _ in test_progress_bar:
        test_data = next(iter(test_dataloader))
        val_loss,output,target = self.forward_pass(test_data, compute_loss=True)
        val_acc = self.net.accuracy(output, target)
        acc += val_acc.detach()
        loss += val_loss.detach()
        idx+=1
        test_desc = f"Testing accuracy: {acc/idx:3.2f}% # Progress: "
        test_progress_bar.set_description(test_desc)

      acc /= idx
      loss /= idx
    self.net.train()