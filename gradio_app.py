# -*- coding: utf-8 -*-
"""
gradio_app.py -- Detecteur CIFAR-100 : ResNet18 (PyTorch) + EfficientNet (TensorFlow/Keras)
============================================================================================

Lancement :
    python gradio_app.py            # acces local  http://localhost:7860
    python gradio_app.py --share    # lien public 72h (pour telephone)

Dependances :
    pip install gradio torch torchvision pillow tensorflow numpy
"""

import argparse
import sys
import os
import socket
import numpy as np
from PIL import Image, ImageDraw

# Fix encodage Windows (CP1252 ne supporte pas les caracteres speciaux)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Detection de l'environnement
IN_COLAB = 'google.colab' in sys.modules


# =============================================================================
# ETAPE 1 -- Definir les classes
# =============================================================================

# 100 classes CIFAR-100 (ordre officiel)
ALL_100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

# Les 20 classes sur lesquelles EfficientNet a ete entraine
# (indices 0->19 dans CIFAR-100)
EFFICIENTNET_CLASSES = ALL_100_CLASSES[:20]

print("[OK] Classes EfficientNet (%d) : %s" % (len(EFFICIENTNET_CLASSES), EFFICIENTNET_CLASSES))
print("[OK] Classes ResNet18 (100 classes CIFAR-100)")


# =============================================================================
# ETAPE 2 -- Charger ResNet18 (PyTorch)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n[INFO] Device PyTorch : %s" % DEVICE)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_ch  = 64
        self.conv1  = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make(64,  2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, num_classes)

    def _make(self, ch, n, stride):
        layers = [ResidualBlock(self.in_ch, ch, stride)]
        self.in_ch = ch
        for _ in range(1, n):
            layers.append(ResidualBlock(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def load_resnet(path):
    """Charge le modele ResNet18 depuis un fichier .pth"""
    model = ResNet18(num_classes=100).to(DEVICE)
    if not os.path.exists(path):
        print("[WARN] ResNet18 non trouve : %s" % path)
        print("       Mode DEMO actif (predictions aleatoires)")
        return None
    ckpt  = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[OK] ResNet18 charge depuis %s" % path)
    return model


# Chemin vers le checkpoint ResNet -- modifie si necessaire
RESNET_PATH = r'C:\Users\ZIDAN\PycharmProjects\Computer_Vision_project\cifar100-project\checkpoints\best.pth'
resnet_model = load_resnet(RESNET_PATH)

# Pipeline de preprocessing pour ResNet18 (normalisation CIFAR-100)
resnet_preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


# =============================================================================
# ETAPE 3 -- Charger EfficientNet (TensorFlow / Keras)
# =============================================================================
efficientnet_model = None

try:
    import tensorflow as tf

    # Chemin vers le fichier .keras
    # Le modele est dans notebooks/cv_model_efficientNet.keras
    EFFICIENTNET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'notebooks', 'cv_model_efficientNet.keras'
    )

    if os.path.exists(EFFICIENTNET_PATH):
        efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
        print("[OK] EfficientNet charge depuis %s" % EFFICIENTNET_PATH)
        print("     Input shape  : %s" % str(efficientnet_model.input_shape))
        print("     Output shape : %s" % str(efficientnet_model.output_shape))
    else:
        print("[WARN] EfficientNet non trouve : %s" % EFFICIENTNET_PATH)
        print("       Installe tensorflow : pip install tensorflow")

except ImportError:
    print("[WARN] TensorFlow non installe -> pip install tensorflow")
    print("       EfficientNet indisponible, mode DEMO actif")
except Exception as e:
    print("[ERROR] Chargement EfficientNet : %s" % str(e))


# =============================================================================
# ETAPE 4 -- Fonctions de prediction
# =============================================================================

def predict_resnet(pil_img, top_k=5):
    """
    ResNet18 (PyTorch) -- 100 classes CIFAR-100.
    Retourne une liste de (classe, score) triee par score decroissant.
    """
    img_rgb = pil_img.convert('RGB')
    if resnet_model is None:
        # Mode demo : scores aleatoires
        probs = np.random.dirichlet(np.ones(100))
    else:
        x = resnet_preprocess(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(resnet_model(x), dim=1)[0].cpu().numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    return [(ALL_100_CLASSES[i], float(probs[i])) for i in top_idx]


def predict_efficientnet(pil_img, top_k=5, confidence_threshold=0.8):
    """
    EfficientNet (Keras/TF) -- 20 classes CIFAR-100.
    Images 32x32, valeurs [0, 255] (pas de normalisation, identique a l'entrainement).
    Si la confiance max < seuil -> retourne 'Inconnu'.
    """
    img_rgb   = pil_img.convert('RGB').resize((32, 32))
    img_array = np.array(img_rgb, dtype='float32')   # [0, 255]
    img_input = np.expand_dims(img_array, axis=0)    # (1, 32, 32, 3)

    if efficientnet_model is None:
        # Mode demo : scores aleatoires sur 20 classes
        probs = np.random.dirichlet(np.ones(20))
    else:
        preds = efficientnet_model.predict(img_input, verbose=0)
        probs = preds[0]   # shape (20,)

    top_k    = min(top_k, len(probs))
    top_idx  = probs.argsort()[::-1][:top_k]
    results  = [(EFFICIENTNET_CLASSES[i], float(probs[i])) for i in top_idx]

    # Si la meilleure prediction est sous le seuil -> "Inconnu"
    best_score = results[0][1]
    if best_score < confidence_threshold:
        results[0] = ('Inconnu', best_score)

    return results


# =============================================================================
# ETAPE 5 -- Rendu visuel (overlay sur l'image)
# =============================================================================

def conf_color(score):
    """Vert si confiant, jaune si moyen, rouge si faible."""
    if score > 0.6:
        return (34, 197, 94)   # vert
    elif score > 0.3:
        return (234, 179, 8)   # jaune
    else:
        return (239, 68, 68)   # rouge


def draw_overlay(pil_img, results, model_name):
    """Dessine un panneau de resultats en bas de l'image."""
    img  = pil_img.copy().convert('RGB').resize((480, 360))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    pad  = 10
    oh   = min(len(results) * 38 + 50, 250)

    # Fond sombre en bas
    draw.rectangle([0, H - oh, W, H], fill=(10, 10, 20))
    draw.text((pad, H - oh + 6), "Model: %s" % model_name, fill=(148, 163, 184))

    for i, (label, score) in enumerate(results):
        y     = H - oh + 30 + i * 38
        color = conf_color(score)
        bw    = int(score * (W - 2 * pad - 120))
        draw.rectangle([pad + 110, y + 6, pad + 110 + bw, y + 28], fill=color)
        draw.rectangle([pad + 110, y + 6, W - pad,        y + 28],
                       outline=(80, 80, 80), width=1)
        draw.text((pad,    y + 6), '%d. %s' % (i + 1, label[:18]), fill=(255, 255, 255))
        draw.text((W - 58, y + 6), '%5.1f%%' % (score * 100),      fill=color)

    return img


# =============================================================================
# ETAPE 6 -- Callbacks Gradio
# =============================================================================

def predict_frame(frame, model_choice, confidence_threshold):
    """
    Callback principal : webcam ou upload.
    model_choice : 'EfficientNet (20 classes)' ou 'ResNet18 (100 classes)'
    """
    if frame is None:
        return None, 'En attente de l\'image...'

    pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame

    try:
        use_efficientnet = "EfficientNet" in model_choice

        if use_efficientnet:
            results    = predict_efficientnet(pil, top_k=5,
                                              confidence_threshold=confidence_threshold)
            model_name = "EfficientNet -- 20 classes"
        else:
            results    = predict_resnet(pil, top_k=5)
            model_name = "ResNet18 -- 100 classes"

    except Exception as e:
        return pil, 'Erreur : %s' % str(e)

    top_label, top_score = results[0]

    if top_score > 0.6:
        icon = '[CONFIANT]'
    elif top_score > 0.3:
        icon = '[MOYEN]'
    else:
        icon = '[FAIBLE]'

    status_text = (
        '%s  %s   --   %.1f%%\n\n' % (icon, top_label.upper(), top_score * 100)
        + '\n'.join(
            '  %d. %-20s %.1f%%' % (i + 1, label, score * 100)
            for i, (label, score) in enumerate(results)
        )
    )

    return draw_overlay(pil, results, model_name), status_text


def predict_upload(image, model_choice, confidence_threshold):
    """Callback pour l'onglet Upload."""
    return predict_frame(image, model_choice, confidence_threshold)


# =============================================================================
# ETAPE 7 -- Interface Gradio
# =============================================================================
import gradio as gr

CSS = """
body, .gradio-container {
    background: #070b14 !important;
    font-family: 'Segoe UI', monospace !important;
}
h1 {
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8rem;
    letter-spacing: 4px;
    padding: 16px 0 4px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 0.85rem;
    margin-bottom: 20px;
}
.gradio-textbox textarea {
    background: #0f172a !important;
    color: #94a3b8 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.9rem !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
footer { display: none !important; }
"""

resnet_status = "OK - Charge" if resnet_model else "DEMO (checkpoint manquant)"
effnet_status = "OK - Charge" if efficientnet_model else "Non disponible (pip install tensorflow)"

with gr.Blocks(title='CIFAR-100 Detector | ResNet18 + EfficientNet') as demo:

    # En-tete
    gr.HTML('<h1>CIFAR-100 DETECTOR</h1>')
    gr.HTML('''
        <div class="subtitle">
            ResNet18 (PyTorch) : {rs} &nbsp;|&nbsp;
            EfficientNet (Keras) : {es} &nbsp;|&nbsp;
            Webcam · Upload · Temps reel
        </div>
    '''.format(rs=resnet_status, es=effnet_status))

    # Controles
    with gr.Row():
        model_choice = gr.Radio(
            choices=["EfficientNet (20 classes)", "ResNet18 (100 classes)"],
            value="EfficientNet (20 classes)",
            label="Choisir le modele",
            info=(
                "EfficientNet -> 20 premieres classes CIFAR-100 "
                "(apple, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, bowl...)\n"
                "ResNet18     -> toutes les 100 classes CIFAR-100"
            )
        )
        confidence_slider = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.8, step=0.05,
            label="Seuil de confiance (EfficientNet uniquement)",
            info="En dessous de ce seuil, EfficientNet affiche 'Inconnu'"
        )

    # Onglets
    with gr.Tabs():

        # Onglet 1 : Webcam
        with gr.Tab("Webcam (temps reel)"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    cam = gr.Image(
                        sources=['webcam'],
                        streaming=True,
                        height=380,
                        show_label=False,
                        webcam_options=gr.WebcamOptions(mirror=False),
                    )
                with gr.Column(scale=2):
                    cam_out_img  = gr.Image(height=240, show_label=False)
                    cam_out_text = gr.Textbox(
                        lines=8,
                        interactive=False,
                        show_label=False,
                        placeholder='Pointe ta camera vers un objet...',
                    )

            cam.stream(
                fn=predict_frame,
                inputs=[cam, model_choice, confidence_slider],
                outputs=[cam_out_img, cam_out_text],
                stream_every=0.4,
            )

        # Onglet 2 : Upload
        with gr.Tab("Upload d'image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    upload_input = gr.Image(
                        sources=['upload'],
                        height=380,
                        show_label=False,
                        label="Depose une image ici",
                        type="numpy",
                    )
                    analyze_btn = gr.Button("Analyser", variant="primary", size="lg")
                with gr.Column(scale=2):
                    upload_out_img  = gr.Image(height=240, show_label=False)
                    upload_out_text = gr.Textbox(
                        lines=8,
                        interactive=False,
                        show_label=False,
                        placeholder="Upload une image puis clique sur Analyser...",
                    )

            analyze_btn.click(
                fn=predict_upload,
                inputs=[upload_input, model_choice, confidence_slider],
                outputs=[upload_out_img, upload_out_text],
            )

    # Legende
    gr.HTML('''
        <p style="text-align:center; color:#334155; font-size:0.75rem; margin-top:8px;">
            Sur telephone : lance avec --share et ouvre le lien &nbsp;|&nbsp;
            [CONFIANT] &gt;60% &nbsp; [MOYEN] 30-60% &nbsp; [FAIBLE] &lt;30%
        </p>
        <p style="text-align:center; color:#1e293b; font-size:0.7rem;">
            EfficientNet classes : apple · aquarium_fish · baby · bear · beaver ·
            bed · bee · beetle · bicycle · bottle · bowl · boy · bridge · bus ·
            butterfly · camel · can · castle · caterpillar · cattle
        </p>
    ''')


# =============================================================================
# ETAPE 8 -- Lancement
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-100 Gradio App (ResNet18 + EfficientNet)')
    parser.add_argument('--share', action='store_true',
                        help='Genere un lien public Gradio (72h) -- utile pour telephone')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port local (defaut : 7860)')
    args = parser.parse_args()

    print('\n' + '=' * 60)
    if args.share or IN_COLAB:
        print('>> Lancement avec lien public...')
        print('>> Le lien ci-dessous fonctionne sur telephone (72h)')
        print('=' * 60 + '\n')
        demo.launch(share=True, server_port=args.port, show_error=True, css=CSS)
    else:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = '127.0.0.1'
        print('>> Lancement local    -> http://localhost:%d' % args.port)
        print('>> Telephone (Wi-Fi) -> http://%s:%d' % (ip, args.port))
        print('   Lien public 72h   -> python gradio_app.py --share')
        print('=' * 60 + '\n')
        demo.launch(
            server_name='0.0.0.0',
            server_port=args.port,
            share=False,
            show_error=True,
            css=CSS,
        )
