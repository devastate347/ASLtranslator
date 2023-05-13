from pyexpat import model
import torch

PATH = '/home/ayaans/Documents/ASLtest/savedModel.txt' 
torch.save(model.state_dict(), asl_model.pth)