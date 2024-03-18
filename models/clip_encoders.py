import clip
import torch
import torch.nn as nn


class TextEncoder():
    def __init__(self, labels, pretrain='ViT-B/32') -> None:
        self.clip_pretrained, _ = clip.load(pretrain, device='cuda', jit=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels
    
    def encode(self, labels):
        text = clip.tokenize(labels).to(self.device)
        text_features = self.clip_pretrained.encode_text(text) 

        return text_features  
    
    def get_similarity(self, feat):
        text_features = self.encode(self.labels)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = feat.to(self.device).half() @ text_features.t()
        return similarity


class ImageEncoder():
    def __init__(self, pretrain='ViT-B/32') -> None:
        self.clip_pretrained, self.transform = clip.load(pretrain, device='cuda', jit=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cos_sim = nn.CosineSimilarity()
    
    @torch.no_grad()
    def encode(self, images, transform=False):
        if transform:
            images = self.transform(images)
        image_features = self.clip_pretrained.encode_image(images)
        
        return image_features
    
    @torch.no_grad()  
    def get_similarity(self, feat1, feat2):
        return self.cos_sim(feat1, feat2)
    
    def get_similarity_from_image(self, image1, image2, transform=False):
        feat1 = self.encode(image1, transform)
        feat2 = self.encode(image2, transform)
        
        return self.get_similarity(feat1, feat2)
        