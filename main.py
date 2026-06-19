import os
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.1-flash-image"

def load_image(path: str | None):
    return Image.open(path) if path else None


def save_first_image(response, output_path: str):
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            Path(output_path).write_bytes(part.inline_data.data)
            return
    raise RuntimeError("No image returned from Gemini.")


def generate_with_control_images(
    render_path: str,
    output_path: str,
    garment_mask_path: str | None = None,
    avatar_mask_path: str | None = None,
    depth_path: str | None = None,
    face_reference_path: str | None = None,
    background_reference_path: str | None = None,
):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    contents = [
        "Main z-weave render:",
        Image.open(render_path),
    ]

    if garment_mask_path:
        contents += [
            "Garment mask. White area is the garment that must be preserved exactly.",
            Image.open(garment_mask_path),
        ]

    if avatar_mask_path:
        contents += [
            "Avatar mask. White area is the avatar/body region that may be replaced with a photorealistic human.",
            Image.open(avatar_mask_path),
        ]

    if depth_path:
        contents += [
            "Depth map from the same camera. Use it to preserve spatial layout, foreground/background separation, and pose depth.",
            Image.open(depth_path),
        ]


    if face_reference_path:
        contents += [
            "Face reference. Use this as identity/style reference for the generated human face.",
            Image.open(face_reference_path),
        ]

    if background_reference_path:
        contents += [
            "Background reference. Use this as background style reference.",
            Image.open(background_reference_path),
        ]

    contents.append(
        """
        Generate a photorealistic fashion image.

        Main requirements:
        - Replace the 3D avatar with a realistic human.
        - Keep the same body pose, camera angle, scale, and garment position.
        - Preserve the garment exactly using the garment mask.
        - Do not change garment silhouette, shape, seams, folds, pattern, print, logo, texture, or color.
        - Use the avatar mask only as the editable human/body area.
        - Use the depth map to keep correct spatial layout.
        - Improve lighting, skin, hair, background, and photographic realism.
        - Output only the final image.
        """
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        ),
    )

    save_first_image(response, output_path)


if __name__ == "__main__":
    generate_with_control_images(
        render_path="data/samples/sample1/sceneImage.png",
        output_path="data/results/result.png",
        garment_mask_path="data/samples/sample1/maskImage.png",
        avatar_mask_path="data/samples/sample1/backgroundMaskImage.png",
        depth_path="data/samples/sample1/depthImage.png",
        face_reference_path="data/face/female_1.jpg",
        background_reference_path="data/background/beach_0.jpg",
    )