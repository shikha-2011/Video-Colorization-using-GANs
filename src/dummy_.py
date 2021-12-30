import argparse
import utils



def func(epochs, data_dir):
  print('from function',epochs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  #parser.add_argument("--model", default='model_imagecolor', type=str, help="Model Name")
  #parser.add_argument("--img_path", default='demo_imgs/ILSVRC2012_val_00040251.JPEG', type=str, help="Test dir path")
  #parser.add_argument("--video_path", default=None, type=str, help="Test dir path")
  #parser.add_argument('--network', type=str, default='half_hyper_unet',help='chooses which model to use. unet, fcn',choices=["half_hyper_unet", "hyper_unet"])
  parser.add_argument("--data_dir", default=None, type=str, help='Path to Dataset')
  parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
  parser.add_argument('--chkpt_dir', default=None, type=str, help='path to Checkpoint directory')

  ARGS = parser.parse_args()
  print(ARGS)

  func(ARGS.epochs, ARGS.data_dir)