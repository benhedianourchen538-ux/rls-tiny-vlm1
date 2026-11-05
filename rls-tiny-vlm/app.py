"""Web interface: drop an image, the model answers with its one-word description.

    pip install gradio
    python app.py --ckpt runs/b0/best.pt            # add --share in Colab to get a public link
The model only knows ShapeScenes (1-2 coloured shapes on a plain background): a photo of a cat
will give a meaningless answer. Use "Random test image" to see real inputs with their reference.
"""
import argparse
import random

import gradio as gr
import torch
from PIL import Image

from src.data import load_split
from src.predict import describe, load_model, predict
from src.train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    device = pick_device(a.device)
    model = load_model(a.ckpt, device)
    test = load_split(a.data_dir, "test")

    def answer(img):
        if img is None:
            return "", "Upload an image first."
        word = predict(model, img, device)
        return word, describe(word)

    def random_sample():
        i = random.randrange(len(test))
        img = Image.fromarray(test.images[i].permute(1, 2, 0).numpy())
        word = predict(model, img, device)
        ok = "correct" if word == test.words[i] else "WRONG"
        return img, word, describe(word), f"{test.words[i]}  ({ok})"

    with gr.Blocks(title="Tiny VLM") as demo:
        gr.Markdown("# Tiny VLM — spells what it sees\nShapeScenes images only (1–2 coloured shapes, 64×64).")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", height=320)
                with gr.Row():
                    run = gr.Button("Describe", variant="primary")
                    rnd = gr.Button("Random test image")
            with gr.Column():
                word = gr.Textbox(label="Generated word (letter by letter)")
                text = gr.Textbox(label="Read as")
                ref = gr.Textbox(label="Reference (test images only)")
        run.click(answer, image, [word, text])
        image.upload(answer, image, [word, text])
        rnd.click(random_sample, None, [image, word, text, ref])
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
