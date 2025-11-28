import gradio as gr
import torch
from PIL import Image
import gc
import numpy as np
import os

# It's good practice to handle potential import errors
try:
    from modules.segmentation_module import SemanticSegmenter, create_color_mask
    from modules.generation_module import run_inpainting_pipeline
except ImportError:
    print("="*80)
    print("ERROR: Could not import from 'modules'.")
    print("Please ensure 'segmentation_module.py' and 'generation_module.py' exist in the 'modules' folder.")
    print("="*80)
    exit()

# --- App State and Configuration ---
APP_STATE = { "original_image": None, "segmentation_map": None }
MODELS_PATH = os.environ.get('HF_HOME', './models')

# --- Main Workflow Functions ---
def process_segmentation(image, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("Please upload an image first.")
    
    APP_STATE.clear()
    progress(0, desc="Loading Segmentation Model...")
    segmenter = SemanticSegmenter(cache_dir=MODELS_PATH)
    
    progress(0.5, desc="Analyzing image and finding objects...")
    pred_seg_map = segmenter.get_segmentation_map(image)
    color_mask_for_display = create_color_mask(pred_seg_map)
    detected_objects = segmenter.get_detected_objects(pred_seg_map)
    
    APP_STATE["original_image"] = image
    APP_STATE["segmentation_map"] = pred_seg_map
    
    object_labels = sorted([f"{v} (ID: {k})" for k, v in detected_objects.items()])
    progress(1.0, desc="Analysis complete!")
    gr.Info("Analysis complete! Select objects from the list to edit.")

    del segmenter
    gc.collect()
    torch.cuda.empty_cache()
    
    return color_mask_for_display, gr.update(choices=object_labels, value=[], interactive=True)

def process_generation(selected_objects_str, prompt, progress=gr.Progress(track_tqdm=True)):
    if APP_STATE["original_image"] is None or not selected_objects_str:
        raise gr.Error("Please run segmentation and select at least one object first.")
    if not prompt:
        raise gr.Error("Please enter an editing prompt.")

    progress(0, desc="Creating custom masks...")
    pred_seg_map = APP_STATE["segmentation_map"]
    
    selected_ids = [int(s.split('(ID: ')[1][:-1]) for s in selected_objects_str]
    binary_mask_np = np.zeros_like(pred_seg_map, dtype=np.uint8)
    for class_id in selected_ids:
        binary_mask_np[pred_seg_map == class_id] = 255
    binary_mask = Image.fromarray(binary_mask_np)
    
    color_control_mask = create_color_mask(pred_seg_map)

    progress(0.2, desc="Starting generative inpainting process...")
    final_image = run_inpainting_pipeline(
        binary_mask=binary_mask,
        color_control_mask=color_control_mask,
        original_image=APP_STATE["original_image"],
        prompt=prompt,
        cache_dir=MODELS_PATH
    )
    
    progress(1.0, desc="Edit complete!")
    gr.Info("Edit complete!")
    return final_image

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Semantic-Guided Editing") as demo:
    gr.Markdown("# 🖼️ Semantic-Guided Generative Editing")
    gr.Markdown("An end-to-end pipeline for targeted image editing. Upload an image, find objects, then describe your edit!")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            run_segmentation_btn = gr.Button("1. Analyze & Find Objects", variant="primary")
        with gr.Column(scale=2):
            segmentation_color_mask = gr.Image(type="pil", label="Identified Objects")
            detected_objects_checkboxes = gr.CheckboxGroup(label="Select objects to edit", interactive=False)
            prompt_textbox = gr.Textbox(label="Enter your editing prompt", placeholder="e.g., a car made of wood, a person wearing a space suit")
            run_generation_btn = gr.Button("2. Generate Edit", variant="primary")
        with gr.Column(scale=2):
            final_output_image = gr.Image(type="pil", label="Final Edited Image")

    run_segmentation_btn.click(
        fn=process_segmentation, 
        inputs=[input_image], 
        outputs=[segmentation_color_mask, detected_objects_checkboxes]
    )
    
    run_generation_btn.click(
        fn=process_generation, 
        inputs=[detected_objects_checkboxes, prompt_textbox], 
        outputs=[final_output_image]
    )

if __name__ == "__main__":
    # --- CHANGE: Removed debug=True for a cleaner, more stable launch ---
    demo.launch(share=True)