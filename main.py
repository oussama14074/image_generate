import torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt, output_path="output_image.png"):
    # Vérifie si un GPU est disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charge le modèle Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to(device)

    # Génère une image à partir du prompt
    image = pipe(prompt).images[0]

    # Sauvegarde l'image générée
    image.save(output_path)
    print(f"L'image a été générée et sauvegardée à {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    generate_image("a dog image")
