import base64
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle de détection (YOLOv8 nano pour être rapide)
try:
    detector = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Erreur YOLO: {e}")
    detector = None

# Charger le modèle de classification (EfficientNet)
# Dans Docker, le modèle sera monté dans /models/
model_path = "/models/cv_model_efficientNet.keras"
try:
    classifier = tf.keras.models.load_model(model_path)
    print("Modèle EfficientNet chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle EfficientNet : {e}")
    classifier = None

# Les classes sélectionnées dans le notebook
CLASSES_CHOISIES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle'
]

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Recevoir l'image encodée en base64 depuis Flutter
            data = await websocket.receive_text()
            
            # Décoder l'image
            img_data = base64.b64decode(data.split(",")[1] if "," in data else data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid image format"})
                continue
            
            detections = []
            
            # 1. Détection d'objets avec YOLO
            if detector:
                results = detector(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Ignorer les boîtes trop petites
                        if x2 - x1 < 10 or y2 - y1 < 10:
                            continue
                            
                        # Découper l'objet détecté
                        cropped_obj = frame[y1:y2, x1:x2]
                        
                        if classifier is not None:
                            # 2. Préprocessing pour EfficientNet (32x32)
                            resized_obj = cv2.resize(cropped_obj, (32, 32))
                            # OpenCV utilise BGR, Tensorflow attend RGB
                            resized_obj = cv2.cvtColor(resized_obj, cv2.COLOR_BGR2RGB)
                            img_array = np.expand_dims(resized_obj, axis=0).astype('float32')
                            
                            # 3. Prédiction
                            preds = classifier.predict(img_array, verbose=0)
                            pred_idx = np.argmax(preds[0])
                            confidence = preds[0][pred_idx]
                            
                            # Logique de couleur
                            if confidence > 0.8:
                                color = "green"
                                label = f"{CLASSES_CHOISIES[pred_idx]} ({confidence*100:.1f}%)"
                            else:
                                color = "red"
                                label = "Inconnu"
                        else:
                            color = "red"
                            label = "Modèle non chargé"
                            
                        detections.append({
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "color": color,
                            "label": label
                        })
                        
            # Renvoyer les détections au client Flutter
            await websocket.send_json({"detections": detections})
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
