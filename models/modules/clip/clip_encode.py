import torch
from torch import nn
import clip

class ClipEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.freeze_clip, self.preprocess = clip.load('ViT-B/32', self.device)
        self.freeze_clip.requires_grad_(False)
        
    def forward(self, image_input, txt_input):
        #proc_image_input = self.preprocess(image_input).unsqueeze(0).to(device)
        proc_image_input = image_input
        proc_txt_input = torch.cat([clip.tokenize(txt_input)]).to(self.device)
        #with torch.no_grad():
        image_features = self.freeze_clip.encode_image(proc_image_input)
        text_features = self.freeze_clip.encode_text(proc_txt_input)

        #similarity = image_features @ text_features.T
        #return similarity
        return image_features, text_features
