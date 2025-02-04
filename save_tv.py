from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
import argparse
import torch.nn.functional as F
import open_clip

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--model_finetuned', type=str, default="", help='finetuned model')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    parser.add_argument('--tv_edit_alpha', type=float, default=0.5, help='amount of edit to task vector layer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_pretrained = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16, safety_checker=None)
    pipe_finetuned = StableDiffusionPipeline.from_pretrained(args.model_finetuned, torch_dtype=torch.float16, safety_checker=None)
    pipe_pretrained.to("cpu")
    pipe_finetuned.to("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
   
    #edit process
    unet_pretrained = pipe_pretrained.unet
    unet_finetuned = pipe_finetuned.unet

    #save model unet
    torch.save(unet_pretrained, "./unet_pretrained.pt")
    torch.save(unet_finetuned, "./unet_finetuned.pt")

    task_vector_unet = TaskVector(pretrained_checkpoint="./unet_pretrained.pt", 
                            finetuned_checkpoint="./unet_finetuned.pt")
    
    
    task_vector_unet = -task_vector_unet

    unet_edited = task_vector_unet.apply_to("./unet_pretrained.pt", scaling_coef=args.tv_edit_alpha)
    
    pipe_pretrained.unet = unet_edited

    pipe_pretrained.save_pretrained(args.output_dir)
    
    os.remove("./unet_pretrained.pt")
    os.remove("./unet_finetuned.pt")
        
