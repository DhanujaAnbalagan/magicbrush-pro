from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import decode_base64_image, encode_image_base64, preprocess_image_and_mask
from scoring import get_scorer
from model_pipeline import get_pipeline

app = FastAPI(title="MagicBrush Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    image: str       # base64
    prompt: str

@app.on_event("startup")
async def startup_event():
    print("Preloading models...")
    get_scorer()
    get_pipeline()
    print("Models loaded successfully.")

@app.post("/generate")
async def generate_images(request: GenerationRequest):
    try:
        print("Decoding inputs...")
        init_image = decode_base64_image(request.image)
        prompt = request.prompt
        
        if not prompt:
            raise ValueError("Prompt is required")

        print("Preprocessing image...")
        processed_image = init_image.resize((512, 512)).convert("RGB")

        print("Generating candidates...")
        pipeline = get_pipeline()
        generated_images = pipeline.generate(
            prompt=prompt,
            image=processed_image,
            num_images=8,
            num_inference_steps=50,
            guidance_scale=7.5,
            image_guidance_scale=1.5
        )

        print("Scoring candidates...")
        scorer = get_scorer()
        results = []
        for i, gen_img in enumerate(generated_images):
            scores = scorer.compute_scores(processed_image, gen_img, prompt)
            
            encoded_img = encode_image_base64(gen_img)
            
            results.append({
                "id": i,
                "image": encoded_img,
                "clip_score": round(scores["clip_score"], 4),
                "lpips_score": round(scores["lpips_score"], 4),
                "final_score": round(scores["final_score"], 4)
            })

        print("Done.")
        results.sort(key=lambda x: x["final_score"], reverse=True)

        return {"status": "success", "results": results}

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
