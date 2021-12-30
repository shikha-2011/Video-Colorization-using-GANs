import argparse
import os
import os 
import glob
import time
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch import nn, optim
from torchvision import transforms 
from torchvision.utils import make_grid
import load_data
import network
import utils

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=99.):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = network.init_model(network.Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)

        self.net_D = network.init_model(network.PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)

        ## Define Loss criterias
        self.GANcriterion = network.GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        # Optimizers
        #####################################################
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        
        
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
    
    def train_model(self, model, data_dir, epochs, chkpt_dir, display_every=94):
      #test_dir = '/content/drive/MyDrive/PROJECT/test_data'
      
      paths = []
      loss_list = []
      for subdir, dirs, files in os.walk(data_dir):
        for file in files:
          #print(os.path.join(subdir, file))
          file_path = os.path.join(subdir, file)
          paths.append(file_path)
      train_paths = paths[:4000]
      val_paths = paths[4000:]
      train_dl = load_data.make_dataloaders(paths=train_paths, split='train')
      val_dl = load_data.make_dataloaders(paths=val_paths, split='val')
      print(val_dl)
      data = next(iter(train_dl))
      Ls, abs_ = data['L'], data['ab']
      print(Ls.shape, abs_.shape)
      print(len(train_dl), len(val_dl))

      try:
            ckpt = utils.load_checkpoint('%s/latest_.ckpt' % (chkpt_dir))
            self.start_epoch = ckpt['epoch']
            self.net_G.load_state_dict(ckpt['net_G'])
            self.net_D.load_state_dict(ckpt['net_D'])
            
            self.opt_G.load_state_dict(ckpt['opt_G'])
            self.opt_D.load_state_dict(ckpt['opt_D'])
      except:
            print('you are here now')
            print(' [*] No checkpoint!')
            self.start_epoch = 0
      data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
      for e in range(self.start_epoch, epochs):
        loss_meter_dict = utils.create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            utils.update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            # Override the latest checkpoint
            #######################################################
            '''
            self.net_G.load_state_dict(ckpt['net_G'])
            self.net_D.load_state_dict(ckpt['net_D'])
            
            self.opt_G.load_state_dict(ckpt['opt_G'])
            self.opt_D.load_state_dict(ckpt['opt_D'])
            '''
            utils.save_checkpoint({'epoch': e + 1, 
                             'net_G': self.net_G.state_dict(),
                             'net_D': self.net_D.state_dict(),
                             'opt_G': self.opt_G.state_dict(),
                             'opt_D': self.opt_D.state_dict()},
                             '%s/latest_.ckpt' % (chkpt_dir))


            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                utils.log_results(loss_meter_dict) # function to print out the losses
                loss_list.append(dict(loss_meter_dict))
                utils.visualize(model, data, save=False) # function displaying the model's outputs
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default=None, type=str, help='Path to Dataset')
  parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
  parser.add_argument('--chkpt_dir', default=None, type=str, help='path to Checkpoint directory')

  ARGS = parser.parse_args()
  print(ARGS)

  model = MainModel()  
  model.train_model(model, ARGS.data_dir, ARGS.epochs, ARGS.chkpt_dir)