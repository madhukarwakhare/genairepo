import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

#load the pretrained processor and model

processor=AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#load image

img_path="image.jpg"

image=Image.open(img_path).convert('RGB')

text="the image of"

inputs=processor(images=image,text=text,return_tensors="pt")

#print(inputs)

output=model.generate(**inputs,max_length=50)

caption=processor.decode(output[0],skip_special_tokens=True)
print(caption)