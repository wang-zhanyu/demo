import torch
# from models.SwinGPT import MultiModal
import streamlit as st
from transformers import ViTImageProcessor
from PIL import Image
import requests
import numpy as np
from transformers import SwinForImageClassification
import torch.nn as nn
import torchxrayvision as xrv
import torch, torchvision
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# ckpt_file = '/home/zhanyu_wang/code/IPMI2023/save/mimic/v2/checkpoints/last.ckpt' 
# ckpt_file = '/home/zhanyu_wang/code/IPMI2023/save/mimic/v2/checkpoints/epoch=3-step=8460-bleu=0.1309-cider=0.3844.ckpt'
# SwinBert = MultiModal.load_from_checkpoint(ckpt_file, strict=False)
# SwinBert.to(device)
# SwinBert.eval()
# SwinBert.freeze()

classification_model = SwinForImageClassification.from_pretrained('microsoft/swin-large-patch4-window7-224-in22k')
classification_model.to(device)
detection_model = xrv.models.DenseNet(weights="densenet121-res224-rsna")
detection_model.to(device)


@st.cache(show_spinner=False)
def Inference(image, num_beams=3):
    image = BytesIO(requests.get(image).content)
    with Image.open(image) as pil:
        array = np.array(pil, dtype=np.uint8)
        if array.shape[-1] != 3 or len(array.shape) != 3:
            array = np.array(pil.convert("RGB"), dtype=np.uint8)
    pixel_values = image_processor(array, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
  
    outputs = classification_model(pixel_values)
    logits = nn.Softmax()(outputs.logits)
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    predict_label = classification_model.config.id2label[predicted_class_idx]
    confident_socre = logits.max().item()

    detect = get_detection_result(array)
    
    # seq = SwinBert.model.generate(pixel_values,
    #                         num_beams=num_beams,
    #                         do_sample=True,
    #                         no_repeat_ngram_size=2,
    #                     #   early_stopping=True,
    #                     #   top_p = 0.92, 
    #                         num_beam_groups=1,
    #                         diversity_penalty=0,
    #                         length_penalty=1,
    #                         repetition_penalty=1,
    #                         min_length=60, 
    #                         max_length=100)
    # report = SwinBert.tokenizer.decode(seq[0], skip_special_tokens=True).replace('.', ' .')
    # if report[-1] != '.':
    #     report = report+'.'
    # report = report.replace('pa and lateral views of the chest provided .', "")
    report = "The chest X-ray reveals findings consistent with pneumonia. The lung tissue appears hazy and white, indicating fluid accumulation in the lung tissue."
    return report, predict_label, confident_socre, detect



def get_detection_result(image):  
    img = xrv.datasets.normalize(image, 255) # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img)
    img = img.to(device)
    outputs = detection_model(img[None,...]) # or model.features(img[None,...]) 
    # Print results
    return dict(zip(detection_model.pathologies, outputs[0].detach().cpu().numpy()))



# @st.cache
def generate_text(prompt, model, api_key, temperature=0.5, max_tokens=20):
    completions_endpoint = f"https://api.openai.com/v1/engines/{model}/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": 1,
        "stop": None,
        "temperature": temperature
    }

    response = requests.post(completions_endpoint, headers=headers, json=data)
    response.raise_for_status()
    message = response.json()["choices"][0]["text"]
    return message



# def create_pdf(image_path, diagnosis, patient_name="Zhangsan", patient_age=50, doctor_name="Lisi"):
#     # Create the PDF document
#     doc = SimpleDocTemplate("diagnosis_report.pdf", pagesize=pagesizes.letter)

#     # Define the style for the report
#     styles = getSampleStyleSheet()
#     styleN = styles['Normal']
#     styleH = styles['Heading1']

#     # Create a list to hold the elements of the report
#     elements = []

#     # Add the title to the report
#     title = Paragraph("X-Ray Diagnosis Report", styleH)
#     elements.append(title)

#     # Add the image and patient information to the report
#     img = Image(image_path, 2 * inch, 2 * inch)
#     patient_info = Paragraph("Patient Name: " + patient_name + "<br />Patient Age: " + patient_age, styleN)
#     table = Table([[img, patient_info]])
#     table.setStyle(TableStyle([
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#     ]))
#     elements.append(table)

#     # Add the doctor information to the report
#     doctor_info = Paragraph("Doctor: " + doctor_name, styleN)
#     elements.append(doctor_info)

#     # Add the diagnosis to the report
#     diagnosis_title = Paragraph("Diagnosis:", styleN)
#     diagnosis_text = Paragraph(diagnosis, styleN)
#     elements.append(diagnosis_title)
#     elements.append(diagnosis_text)

#     # Build the report
#     doc.build(elements)



if __name__ == '__main__':
    xray_image = '/home/zhanyu_wang/data/mimic_cxr/images/p10/p10329986/s53179156/30fb027e-ee828baa-5ce014b6-056d6cb9-3c280d29.jpg'
    # report = Inference(xray_image)
    out = get_detection_result(xray_image)
    print(out)
    
    # Example usage
    # create_pdf(xray_image, "John Doe", "35", "Dr. Jane Smith", "Possible Fracture in the Left Arm")
