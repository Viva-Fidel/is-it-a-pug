import torch
import gradio as gr
import requests
from PIL import Image
from torchvision import transforms

model = torch.load('best_model.pth')
model = model.eval()

labels = ("not pug", "pug")

mean = [0.5707, 0.5531, 0.4893]
std = [0.2411, 0.2346, 0.2364]

image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])

def predict(inp):
  #inp = transforms.ToTensor()(inp).unsqueeze(0)
  inp = image_transforms(inp).float().unsqueeze(0)

  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(2)}
  return confidences

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             examples=["pug.jpg", "not_pug.jpg"]).launch()