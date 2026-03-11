import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

class InstructionPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Diffusion model on {self.device}...")
        
        # Load InstructPix2Pix Model
        model_id = "timbrooks/instruct-pix2pix"
        
        # Use fp16 if CUDA is available
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, image: Image.Image, 
                 num_images: int = 8, num_inference_steps: int = 50, guidance_scale: float = 7.5, image_guidance_scale: float = 1.5):
        """
        Generate multiple images based on the instruction prompt and original image.
        """
        print(f"Generating {num_images} images for instruction: '{prompt}'...")
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                image=image,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator
            )
            
        return output.images

# Global instance
_pipeline = None

def get_pipeline() -> InstructionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InstructionPipeline()
    return _pipeline
