"""
app_gradio.py — CIFAR-100 Détection temps réel
Compatible Gradio 4.x et 5.x / 6.x

Lancement local :
    pip install gradio torch torchvision pillow
    python app_gradio.py

Lancement avec lien public (téléphone) :
    python app_gradio.py --share
"""

import argparse
import sys
import os
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

# ─── Détection environnement ──────────────────────────────────────────────────
IN_COLAB = 'google.colab' in sys.modules
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Environnement : {"Colab" if IN_COLAB else "Local"}  |  Device : {DEVICE}')

# ─── 100 classes CIFAR-100 ────────────────────────────────────────────────────
CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle',
    'bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel',
    'can','castle','caterpillar','cattle','chair','chimpanzee','clock',
    'cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster',
    'house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion',
    'lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree',
    'pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine',
    'possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal',
    'shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank',
    'telephone','television','tiger','tractor','train','trout','tulip',
    'turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
]

# ─── Architecture ResNet18 ────────────────────────────────────────────────────
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
        for _ in range(1, n): layers.append(ResidualBlock(ch, ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return self.fc(torch.flatten(self.pool(x), 1))

# ─── Chargement du modèle ─────────────────────────────────────────────────────
def load_model(path: str) -> nn.Module:
    model = ResNet18(num_classes=100).to(DEVICE)
    if not os.path.exists(path):
        print(f'⚠️  Modèle non trouvé : {path}')
        print('   L\'app tourne en mode DEMO (prédictions aléatoires)')
        return None
    ckpt  = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'✅ Modèle chargé depuis {path}')
    return model

# ── Modifie ce chemin ─────────────────────────────────────────────────────────
MODEL_PATH = r'C:\Users\ZIDAN\PycharmProjects\Computer_Vision_project\cifar100-project\checkpoints\best.pth'
model = load_model(MODEL_PATH)

# ─── Pipeline d'inférence ─────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def predict(pil_img: Image.Image, top_k: int = 5):
    img_rgb = pil_img.convert('RGB')
    if model is None:
        # Mode demo — scores aléatoires
        probs = np.random.dirichlet(np.ones(100))
    else:
        x = preprocess(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top_idx = probs.argsort()[::-1][:top_k]
    return [(CLASSES[i], float(probs[i])) for i in top_idx]

# ─── Overlay sur l'image ──────────────────────────────────────────────────────
def conf_color(s):
    return (34, 197, 94) if s > 0.6 else ((234, 179, 8) if s > 0.3 else (239, 68, 68))

def draw_overlay(pil_img: Image.Image, results: list) -> Image.Image:
    img  = pil_img.copy().convert('RGB').resize((480, 360))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    pad  = 10
    oh   = min(len(results) * 38 + 20, 220)

    # Fond sombre en bas
    draw.rectangle([0, H - oh, W, H], fill=(0, 0, 0))

    for i, (label, score) in enumerate(results):
        y     = H - oh + 10 + i * 38
        color = conf_color(score)
        bw    = int(score * (W - 2 * pad - 120))
        draw.rectangle([pad + 110, y + 6, pad + 110 + bw, y + 28], fill=color)
        draw.rectangle([pad + 110, y + 6, W - pad,        y + 28], outline=(80, 80, 80), width=1)
        draw.text((pad,    y + 6), f'{i+1}. {label[:18]}', fill=(255, 255, 255))
        draw.text((W - 58, y + 6), f'{score*100:5.1f}%',   fill=color)

    return img

# ─── Callback Gradio ──────────────────────────────────────────────────────────
def predict_frame(frame):
    if frame is None:
        return None, '⏳ En attente de la caméra...'
    pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
    try:
        results = predict(pil, top_k=5)
    except Exception as e:
        return pil, f'❌ Erreur : {e}'
    top  = results[0]
    icon = '🟢' if top[1] > 0.6 else ('🟡' if top[1] > 0.3 else '🔴')
    txt  = (f'{icon} {top[0].upper()}  —  {top[1]*100:.1f}%\n\n'
            + '\n'.join(f'  {i+1}. {l:<20} {s*100:.1f}%' for i, (l, s) in enumerate(results)))
    return draw_overlay(pil, results), txt

# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0a0a0f !important; font-family: 'Courier New', monospace !important; }
h1 {
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.6rem; letter-spacing: 3px; padding: 12px 0;
}
.subtitle { text-align: center; color: #475569; font-size: 0.82rem; margin-bottom: 16px; }
.gradio-textbox textarea {
    background: #0f172a !important; color: #94a3b8 !important;
    font-family: 'Courier New', monospace !important; font-size: 0.9rem !important;
    border: 1px solid #1e293b !important; border-radius: 8px !important;
}
footer { display: none !important; }
"""

# ─── Interface ────────────────────────────────────────────────────────────────
with gr.Blocks(title='CIFAR-100 Detector', css=CSS) as demo:

    gr.HTML('<h1>◈ CIFAR-100 DETECTOR</h1>')
    gr.HTML('<div class="subtitle">ResNet18 · 100 classes · temps réel</div>')

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            # ✅ FIX : webcam_options remplace mirror_webcam (Gradio 4+)
            cam = gr.Image(
                sources=['webcam'],
                streaming=True,
                height=380,
                show_label=False,
                webcam_options=gr.WebcamOptions(mirror=False),
            )
        with gr.Column(scale=2):
            out_img  = gr.Image(height=240, show_label=False)
            out_text = gr.Textbox(
                lines=8,
                interactive=False,
                show_label=False,
                placeholder='Pointe ta caméra vers un objet...',
            )

    gr.HTML(
        '<p style="text-align:center;color:#334155;font-size:0.75rem;">'
        '📱 Sur téléphone : ouvre le lien et autorise la caméra  ·  '
        '🟢 &gt;60%  🟡 30-60%  🔴 &lt;30%</p>'
    )

    # ✅ FIX : stream_every remplace time_limit (Gradio 4+)
    cam.stream(
        fn=predict_frame,
        inputs=[cam],
        outputs=[out_img, out_text],
        stream_every=0.4,
    )

# ─── Lancement ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help='Génère un lien public Gradio (accessible sur téléphone)')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    print('\n' + '=' * 55)
    if args.share or IN_COLAB:
        print('🚀 Lancement avec lien public...')
        print('📱 Le lien ci-dessous fonctionne sur téléphone (72h)')
        print('=' * 55 + '\n')
        demo.launch(share=True, server_port=args.port, show_error=True)
    else:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = '127.0.0.1'
        print(f'🚀 Lancement local → http://localhost:{args.port}')
        print(f'📱 Sur téléphone (même Wi-Fi) → http://{ip}:{args.port}')
        print('   Pour un lien public : python app_gradio.py --share')
        print('=' * 55 + '\n')
        demo.launch(
            server_name='0.0.0.0',
            server_port=args.port,
            share=False,
            show_error=True,
        )