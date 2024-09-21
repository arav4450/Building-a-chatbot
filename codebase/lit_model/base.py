"""Basic LightningModules on which other modules can be built."""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR, OneCycleLR,ReduceLROnPlateau


OPTIMIZER = "AdamW"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


def create_masks(question, reply_input, reply_target):
    
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    
    question_mask = question!=0
    question_mask = question_mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_words)
     
    reply_input_mask = reply_input!=0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data) 
    reply_input_mask = reply_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target!=0              # (batch_size, max_words)
    
    return question_mask, reply_input_mask, reply_target_mask




class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))   # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)   # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)       # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)    # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config
        self.mapping = self.data_config["mapping"]
        

        # optimizer
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        # learning rate
        self.lr = self.args.get("lr", LR)
        # loss function
        #loss = self.args.get("loss", LOSS)
        #self.loss_fn = getattr(torch.nn.functional, loss)
        self.loss_fn = LossWithLS(self.model.vocab_size, 0.1)
        # learning rate scheduling
        #self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        # evaluation metric
        #self.train_acc = Accuracy(task = 'binary')
        #self.val_acc = Accuracy(task = 'binary')
        #self.test_acc = Accuracy(task = 'binary')

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser
    
    @classmethod
    def log_dirname(cls):
        return Path(__file__).resolve().parents[2] / "logs"

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,betas=(0.9, 0.98), eps=1e-9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,mode='min', patience= 0, factor= 0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}

       
    def forward(self, a,b,c,d):
        return self.model(a,b,c,d)

    def predict(self,w, x,y,z):
        logits = self.model(w,x,y,z)
        return logits

    def training_step(self, batch, batch_idx):
        
        loss = self._run_on_batch(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True,logger=True)
        
        outputs = {"loss": loss}

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        
        question, reply = batch

        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)


        logits = self(question, question_mask, reply_input, reply_input_mask)

        #loss = self.loss_fn(logits.reshape(-1,logits.size(2)), reply_target.reshape(-1))
        loss = self.loss_fn(logits, reply_target, reply_target_mask)

    
        return loss


    def validation_step(self, batch, batch_idx):
        
        loss = self._run_on_batch(batch)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,logger=True)
        
        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        predictions = torch.nn.functional.softmax(logits, dim=1)
        self.test_acc(torch.argmax(predictions , dim=1), y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        