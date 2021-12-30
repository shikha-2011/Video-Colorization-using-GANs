
import argparse
import os
import os 
import glob
import time
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms 
from torchvision.utils import make_grid
import load_data
import network
import utils
import main_train

def test_fn(model, path, test_dir, output_dir, save=False):
    
    test_paths = []
    for subdir, dirs, files in os.walk(test_dir):
      for file in files:
          #print(os.path.join(subdir, file))
          file_path = os.path.join(subdir, file)
          test_paths.append(file_path)
    
    test_dl = load_data.make_dataloaders(batch_size=1, paths=test_paths, split='test')
    #data = next(iter(test_dl))
    print("Your test Dataloader is created..")
    #model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    #checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(torch.load(path,map_location = 'cpu'), strict=False)
    #model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.net_G.eval()
    print("Model is Loaded..")
    count=0
    for data in test_dl:
        with torch.no_grad():
            model.setup_input(data)
            model.forward()
        #model.net_G.train()
        fake_color = model.fake_color.detach()
        #print(fake_color.shape)
        real_color = model.ab
        L = model.L
        fake_imgs = utils.lab_to_rgb(L, fake_color)
        real_imgs = utils.lab_to_rgb(L, real_color)
        #print(len(fake_imgs))
        for i in range(len(fake_imgs)):
            plt.imshow(fake_imgs[i])
            plt.show()
            plt.imsave(output_dir+"/%#05d_horse.jpg" %(count),fake_imgs[i])
            count+=1
    print(count)

    
    #utils.visualize(model, data, save=False)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default=None, type=str, help='Path to Model')
  parser.add_argument('--test_dir', default=1, type=str, help='Path to Images')
  parser.add_argument('--output_dir', default=1, type=str, help='Path to output directory')

  ARGS = parser.parse_args()
  #print(ARGS)

  model = main_train.MainModel()
  test_fn(model, ARGS.model_path, ARGS.test_dir, ARGS.output_dir)