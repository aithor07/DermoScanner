from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F # Para calcular porcentajes

app = FastAPI()

# --- 1. CONFIGURACIÓN DEL MODELO ---
print("Inicializando Modelo...")

# A. Definimos la misma arquitectura que usamos para entrenar
# No cargamos weights='DEFAULT' porque vamos a usar los nuestros
model = models.mobilenet_v2(weights=None) 

# B. Reemplazamos la capa final (igual que en train.py)
# MobileNetV2 original tiene 1280 entradas en la ultima capa
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)

# C. Cargamos los pesos entrenados
# map_location='cpu' asegura que funcione aunque se haya entrenado en GPU
try:
    model.load_state_dict(torch.load("modelo_melanoma.pth", map_location=torch.device('cpu')))
    print("Pesos 'modelo_melanoma.pth' cargados correctamente.")
except FileNotFoundError:
    print("ERROR: No encuentro 'modelo_melanoma.pth'. Asegúrate de que esté en la misma carpeta.")

# D. Ponemos el modelo en modo evaluación para que no siga aprendiendo
model.eval()

# Definimos las transformaciones (las mismas que en el entrenamiento)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Definimos las clases (en orden alfabético de las carpetas)
CLASS_NAMES = ['Benigno', 'Maligno'] 

@app.get("/")
def home():
    return {"mensaje": "API Detector de Melanoma v2.0 (Modelo Propio)"}

@app.post("/predict")
async def predecir_melanoma(file: UploadFile = File(...)):
    
    # 1. LEER Y PROCESAR IMAGEN
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocesamos y añadimos la dimensión del batch (1, 3, 224, 224)
    input_tensor = preprocess(image).unsqueeze(0)
    
    # 2. INFERENCIA
    with torch.no_grad():
        output = model(input_tensor)
        
        # Convertimos la salida en probabilidades (0% a 100%)
        probabilities = F.softmax(output, dim=1)
        
        # Obtenemos la clase ganadora y su probabilidad
        top_prob, top_catid = torch.max(probabilities, 1)
        
        confidence = top_prob.item() * 100
        predicted_class = CLASS_NAMES[top_catid.item()]

    # 3. LÓGICA PRINCIPAL
    # Si la confianza es muy baja, avisamos
    mensaje_extra = ""
    if confidence < 70:
        mensaje_extra = "Confianza baja. La imagen podría no ser clara o no ser un lunar."

    return {
        "archivo": file.filename,
        "diagnostico_ia": predicted_class,
        "probabilidad_maligno": f"{probabilities[0][1].item() * 100:.2f}%",
        "probabilidad_benigno": f"{probabilities[0][0].item() * 100:.2f}%",
        "confianza_prediccion": f"{confidence:.2f}%",
        "aviso": mensaje_extra
    }