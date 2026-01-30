import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import torch_directml

def train_model():
    # 1. MOTOR
    try:
        device = torch_directml.device()
        print(f"Usando AMD DirectML")
    except:
        device = torch.device("cpu")

    # 2. TRANSFORMACIONES
    #Para que las imagenes con las que vamos a entrenar el modelo sean lo mas parecidas a las que podria sacar un usuario con su movil.
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'data/train'
    # Verificación de seguridad
    if not os.path.exists(data_dir):
        print(f"ERROR: No se encuentra {data_dir}")
        return

    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=64, shuffle=True, num_workers=0) #Intentando poner mas hilos puede acelerar el proceso de entrenamiento pero puede dar fallos, por lo que de momento num_workers = 0.

    # 3. MODELO
    print("Cargando MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Descongelado parcial
    for param in model.features.parameters():
        param.requires_grad = False 
    
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    model = model.to(device)

    # --- 4. CONFIGURACIÓN DEL CASTIGO (WEIGHTED LOSS) ---
    # Aquí definimos el peso: Benigno=1, Maligno=8
    weights = torch.tensor([1.0, 8.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=weights) # <--- ÚNICA definición válida
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # SCHEDULER
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 35 
    best_acc = 0.0

    print(f"Fine-tuning con castigo ponderado (Weighted Loss) ({epochs} Épocas)...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Usará la pérdida con pesos
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        print(f"Epoch {epoch+1}/{epochs} [LR: {current_lr:.6f}] -> Acc: {epoch_acc:.2%}")

        # Guardamos siempre que mejore la precisión
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "modelo_melanoma.pth")
            print(f"¡Récord! Modelo guardado.")

    print(f"Mejor Precisión Final: {best_acc:.2%}")

if __name__ == '__main__':
    train_model()