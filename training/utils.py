import torch
import os

import torchvision
import torchvision.transforms as T

from matplotlib import pyplot as plt
from enum import IntEnum

from pathlib import Path
from PIL import Image


# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()


USE_MPS = True


class BimapClasses(IntEnum):
    BONE = 1
    BACKGROUND = 0
    

def save_model_checkpoint(model, cp_name, working_dir):
    torch.save(model.state_dict(), os.path.join(str(working_dir), cp_name))
 
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and USE_MPS:
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )

# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    elif torch.backends.mps.is_available() and USE_MPS:
        return x.to(torch.device("mps"))
    else:
        return x.cpu()
    
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")
# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def

def print_data_in_grid(tensor_data: torch.Tensor):
    return t2img(torchvision.utils.make_grid(tensor_data.float(), nrow=8))
    
def plot_inputs_targets_predictions(inputs, targets, predictions, save_path, title, show_plot):
      # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(t2img(torchvision.utils.make_grid(inputs, nrow=20)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(targets.float(), nrow=20)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(predictions.float(), nrow=20)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if save_path is not None:
        
        plt.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0.4)
        
        
    if show_plot is False:
        close_figures()
    else:
        plt.show()
    
def create_gif_from_images(images_dir: Path, gif_path_name: Path):
    
    
    filenames = sorted(images_dir.glob("epoch_*.png"), 
                       key=lambda name: int(name.stem.split("_")[-1]) # sort by epoch number
                       )
    images = [Image.open(image) for image in filenames]
    
    images += [images[-1]] * 100

    vid_name = gif_path_name

    frame_one = images[0]
    frame_one.save(vid_name, format="GIF", append_images=images,
                save_all=True, duration=100, loop=0)