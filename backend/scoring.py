import torch
import lpips
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms

class Scorer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading scoring models on {self.device}...")
        
        # Load LPIPS
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_model.eval()
        
        # Load CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def compute_scores(self, original_img: Image.Image, generated_img: Image.Image, prompt: str):
        # --- CLIP Score ---
        # CLIP computes similarity between text prompt and generated image
        inputs = self.clip_processor(text=[prompt], images=generated_img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Logits per image is the cosine similarity * logit_scale
            clip_score = outputs.logits_per_image.item()

        # --- LPIPS Score ---
        # LPIPS computes perceptual distance between original and generated image
        # Images need to be normalized [-1, 1]
        img_a = self.transform(original_img).unsqueeze(0).to(self.device)
        img_b = self.transform(generated_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lpips_score = self.lpips_model(img_a, img_b).item()

        # Convert CLIP score to be roughly [0, 1] range to combine with LPIPS (which is [0, 1])
        scaled_clip = clip_score / 40.0
        
        # FinalScore = (0.7 * CLIPScore) - (0.3 * LPIPSScore)
        final_score = (0.7 * scaled_clip) - (0.3 * lpips_score)
        
        return {
            "clip_score": clip_score,
            "lpips_score": lpips_score,
            "final_score": final_score
        }

# Global instance
_scorer = None

def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer
