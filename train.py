import os
import torch
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cpu"
print("Running Vision-Encoder + Text-Decoder training on CPU...")

# 1. Load Open-Source Vision-Encoder + Text-Decoder Model
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

# Freeze Vision Encoder so only Cross-Attention & Text Decoder train (fast on CPU)
for param in model.vision_model.parameters():
    param.requires_grad = False

# 2. Load Dataset & Save 3 Sample Images for Streamlit Demo Testing
dataset = load_dataset("AI-Lab-Makerere/beans", split="train[:60]")
test_ds = load_dataset("AI-Lab-Makerere/beans", split="test")

os.makedirs("sample_test_images", exist_ok=True)
saved_classes = set()
for idx, sample in enumerate(test_ds):
    label_id = sample["labels"]
    if label_id not in saved_classes:
        name = test_ds.features["labels"].names[label_id]
        sample["image"].convert("RGB").save(f"sample_test_images/{name}.jpg")
        saved_classes.add(label_id)
    if len(saved_classes) == 3:
        break
print("Saved 3 sample leaf images in 'sample_test_images/' for testing!")

# 3. Map Labels to Full Diagnostic Descriptions for the Text Decoder
diagnostic_map = {
    0: "diagnosis: angular leaf spot disease with dark brown polygonal lesions on the leaf.",
    1: "diagnosis: bean rust fungal infection characterized by reddish-brown circular pustules.",
    2: "diagnosis: healthy green plant leaf with no visible fungal spots or disease symptoms.",
}

# 4. Fine-Tuning Loop
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
model.train()

epochs = 2
for epoch in range(epochs):
    total_loss = 0.0
    for step, item in enumerate(dataset):
        image = item["image"].convert("RGB")
        target_text = diagnostic_map[item["labels"]]

        inputs = processor(images=image, text=target_text, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

# 5. Save Fine-Tuned Model Locally
save_dir = "./saved_leaf_model"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Training complete! Model saved to '{save_dir}'.")
