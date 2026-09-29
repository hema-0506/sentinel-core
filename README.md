"""
Defense app Python Inference Service
- Runs entirely on CPU (no NVIDIA toolkit needed)
- Loads checkpoint trained on Kaggle
- Exposes /conceal and /reveal endpoints for the Go backend
- Texture gate enforced server-side here as well
"""

import os
import io
import json
import uuid
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Force CPU ────────────────────────────────────────────────
# On an i3 with no CUDA, this is always the case anyway,
# but being explicit prevents accidental GPU attempts.
DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)   # use all cores on i3

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("Defense")

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "./models/Defense_final.pth")
IMAGE_SIZE   = int(os.getenv("IMAGE_SIZE", "256"))
NUM_SECRETS  = int(os.getenv("NUM_SECRETS", "5"))
TEXTURE_MIN  = float(os.getenv("TEXTURE_MIN", "50.0"))
OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "/tmp/defense_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# MODEL DEFINITION
# Must match exactly what you trained on Kaggle.
# If you used the aligned version (with PermutationBlock +
# affine ACB), use DefenseAligned below.
# If you used the original additive ACB version, use DefenseOriginal.
# Set MODEL_VARIANT env var to "aligned" or "original".
# ════════════════════════════════════════════════════════════

MODEL_VARIANT = os.getenv("MODEL_VARIANT", "aligned")


# ── Haar wavelet ─────────────────────────────────────────────

class Haar:
    @staticmethod
    def dwt(x):
        x1=x[:,:,0::2,0::2]; x2=x[:,:,1::2,0::2]
        x3=x[:,:,0::2,1::2]; x4=x[:,:,1::2,1::2]
        LL=(x1+x2+x3+x4)/4;  HL=(-x1-x2+x3+x4)/4
        LH=(-x1+x2-x3+x4)/4; HH=(x1-x2-x3+x4)/4
        return torch.cat([LL,HL,LH,HH],dim=1)

    @staticmethod
    def iwt(x):
        B,C4,H,W=x.shape; C=C4//4
        LL,HL,LH,HH=x[:,0:C],x[:,C:2*C],x[:,2*C:3*C],x[:,3*C:4*C]
        out=torch.zeros(B,C,H*2,W*2,device=x.device,dtype=x.dtype)
        out[:,:,0::2,0::2]=LL-HL-LH+HH; out[:,:,1::2,0::2]=LL-HL+LH-HH
        out[:,:,0::2,1::2]=LL+HL-LH-HH; out[:,:,1::2,1::2]=LL+HL+LH+HH
        return out


# ── ALIGNED model (PermutationBlock + affine ACB + IHNN) ─────

class PermutationBlock(nn.Module):
    def __init__(self, num_channels, seed):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed)
        perm = torch.randperm(num_channels, generator=g)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))
    def forward(self, x, reverse=False):
        return x[:, self.inv_perm if reverse else self.perm]

class SubNet(nn.Module):
    def __init__(self, ch, growth=24, dense_layers=3):
        super().__init__()
        layers = []; c = ch
        for _ in range(dense_layers):
            layers += [nn.Conv2d(c, c+growth, 3, 1, 1), nn.LeakyReLU(0.2, True)]
            c += growth
        proj = nn.Conv2d(c, ch, 1)
        nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
        layers.append(proj)
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ACBAligned(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.psi = SubNet(ch); self.phi = SubNet(ch)
        self.rho = SubNet(ch); self.eta = SubNet(ch)

    @staticmethod
    def _s(raw):
        return torch.exp(torch.tanh(raw.float()) * 0.5)

    def forward(self, Xc, Xs, rev=False):
        if not rev:
            s1=self._s(self.psi(Xs)); t1=self.phi(Xs).float()
            Xst=Xc.float()*s1+t1
            s2=self._s(self.rho(Xst)); t2=self.eta(Xst).float()
            R=Xs.float()*s2+t2
            return Xst.to(Xc), R.to(Xc)
        else:
            Xst,Z=Xc.float(),Xs.float()
            s2=self._s(self.rho(Xst)); t2=self.eta(Xst).float()
            Xs_rec=(Z-t2)/s2
            s1=self._s(self.psi(Xs_rec)); t1=self.phi(Xs_rec).float()
            Xc_rec=(Xst-t1)/s1
            return Xc_rec.to(Xc), Xs_rec.to(Xc)

class IHNNAligned(nn.Module):
    def __init__(self, stage_idx, num_acb=6):
        super().__init__()
        wch = 12
        self.perm_c = PermutationBlock(wch, 42 + stage_idx)
        self.perm_s = PermutationBlock(wch, 1042 + stage_idx)
        self.acbs   = nn.ModuleList([ACBAligned(wch) for _ in range(num_acb)])

    def forward(self, xcover, xsecret):
        Xc = self.perm_c(Haar.dwt(xcover))
        Xs = self.perm_s(Haar.dwt(xsecret))
        for acb in self.acbs: Xc, Xs = acb(Xc, Xs, rev=False)
        return Haar.iwt(self.perm_c(Xc, reverse=True)), Haar.iwt(self.perm_s(Xs, reverse=True))

    def reverse(self, xstego, r):
        Xst = self.perm_c(Haar.dwt(xstego))
        R   = self.perm_s(Haar.dwt(r))
        for acb in reversed(self.acbs): Xst, R = acb(Xst, R, rev=True)
        return Haar.iwt(self.perm_c(Xst, reverse=True)), Haar.iwt(self.perm_s(R, reverse=True))

class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Flatten(),
            nn.Linear(ch,max(ch//4,8)),nn.ReLU(),
            nn.Linear(max(ch//4,8),ch),nn.Sigmoid())
    def forward(self, x):
        return x*self.fc(self.pool(x)).view(x.size(0),-1,1,1)

class ImportanceMapAligned(nn.Module):
    def __init__(self, in_ch=3, feat=24):
        super().__init__()
        self.enc=nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch,feat,3,1,1),nn.ReLU(True),SEBlock(feat))
            for _ in range(3)])
        self.dec=nn.Sequential(
            nn.Conv2d(feat*3,feat*3,3,1,1),nn.ReLU(True),
            nn.Conv2d(feat*3,in_ch,1),nn.Sigmoid())
    def forward(self,xcover,xstego,xsecret):
        fs=[e(x) for e,x in zip(self.enc,[xcover,xstego,xsecret])]
        return self.dec(torch.cat(fs,1))

class DefenseAligned(nn.Module):
    def __init__(self, num_secrets=5):
        super().__init__()
        self.num_secrets = num_secrets
        self.ihnns = nn.ModuleList([IHNNAligned(j) for j in range(num_secrets)])
        self.imps  = nn.ModuleList([ImportanceMapAligned() for _ in range(num_secrets-1)])

    def conceal(self, xcover, secrets):
        stego_list, r_list = [], []
        cur = xcover
        for j, ihnn in enumerate(self.ihnns):
            if j > 0:
                ximp = self.imps[j-1](xcover, stego_list[j-1], secrets[j])
                cur  = cur + ximp * 0.1
            xstego, r = ihnn(cur, secrets[j])
            stego_list.append(xstego); r_list.append(r); cur = xstego
        return stego_list, r_list

    def reveal(self, stego_list, r_list):
        secrets_rec = []; cur = stego_list[-1]
        for j in reversed(range(self.num_secrets)):
            xcr, xsr = self.ihnns[j].reverse(cur, r_list[j])
            secrets_rec.insert(0, xsr); cur = xcr
        return cur, secrets_rec


# ── ORIGINAL model (additive ACB, from first training code) ──

class SubNetOriginal(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch,ch,3,padding=1))
    def forward(self, x): return self.net(x)

class ACBOriginal(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.f = SubNetOriginal(ch)
    def forward(self, x, reverse=False):
        x1,x2=torch.chunk(x,2,dim=1)
        if not reverse: return torch.cat([x1, x2+self.f(x1)], 1)
        else:           return torch.cat([x1, x2-self.f(x1)], 1)

class SingleIHNNOriginal(nn.Module):
    def __init__(self, blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([ACBOriginal(12) for _ in range(blocks)])
    def forward(self, cover_feat, secret_feat):
        x = torch.cat([cover_feat, secret_feat], dim=1)
        for b in self.blocks: x = b(x)
        return torch.chunk(x, 2, dim=1)
    def reverse(self, cover_feat, hidden_feat):
        x = torch.cat([cover_feat, hidden_feat], dim=1)
        for b in reversed(self.blocks): x = b(x, reverse=True)
        return torch.chunk(x, 2, dim=1)

class ImportanceModuleOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12,32,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,12,1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

def generate_permutation(channels, seed=42):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randperm(channels, generator=g)

class DefenseOriginal(nn.Module):
    def __init__(self, num_secrets=5, seed=42):
        super().__init__()
        self.num_secrets = num_secrets; self.seed = seed
        self.stages    = nn.ModuleList([SingleIHNNOriginal() for _ in range(num_secrets)])
        self.importance = ImportanceModuleOriginal()

    def conceal(self, cover, secrets):
        cover_feat = Haar.dwt(cover)
        perm = generate_permutation(cover_feat.size(1), self.seed)
        cover_feat = cover_feat[:, perm]
        stego_list = []; r_list = []
        for i in range(self.num_secrets):
            secret_feat = Haar.dwt(secrets[i])
            imp = self.importance(cover_feat)
            mod_secret = secret_feat * imp
            cover_feat, hidden = self.stages[i](cover_feat, mod_secret)
            stego_list.append(Haar.iwt(cover_feat[:, torch.argsort(perm)]))
            r_list.append(hidden)
        cover_feat = cover_feat[:, torch.argsort(perm)]
        stego = Haar.iwt(cover_feat)
        stego_list[-1] = stego
        return stego_list, r_list, perm

    def reveal(self, stego_list, r_list, perm=None):
        if perm is None:
            perm = generate_permutation(12, self.seed)
        cover_feat = Haar.dwt(stego_list[-1])
        cover_feat = cover_feat[:, perm]
        recovered = []
        for i in reversed(range(self.num_secrets)):
            cover_feat, secret_feat = self.stages[i].reverse(cover_feat, r_list[i])
            recovered.insert(0, Haar.iwt(secret_feat))
        cover_feat = cover_feat[:, torch.argsort(perm)]
        return Haar.iwt(cover_feat), recovered


# ════════════════════════════════════════════════════════════
# MODEL LOADER
# Handles the most common Kaggle checkpoint layouts:
#   1. torch.save(model.state_dict(), path)           → plain dict
#   2. torch.save({'model': model.state_dict()}, path) → wrapped
#   3. torch.save(model, path)                         → full object
# ════════════════════════════════════════════════════════════

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    if not Path(MODEL_PATH).exists():
        log.warning(f"Model file not found at {MODEL_PATH} — running in DEMO mode (untrained weights)")
        _model = _build_fresh_model()
        return _model

    log.info(f"Loading model from {MODEL_PATH} on CPU …")
    try:
        # Always map to CPU regardless of what device it was trained on
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

        _model = _build_fresh_model()

        # Detect checkpoint layout
        if isinstance(checkpoint, dict):
            # Try common wrapper keys first
            state_dict = (checkpoint.get("model")
                       or checkpoint.get("state_dict")
                       or checkpoint.get("model_state_dict")
                       or checkpoint)          # bare state_dict

            # Strip any "module." prefix from DataParallel training
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            missing, unexpected = _model.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"Missing keys ({len(missing)}): {missing[:5]} …")
            if unexpected:
                log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        else:
            # torch.save(model, ...) — full object saved
            _model = checkpoint.cpu()

        _model.eval()
        log.info("Model loaded successfully")

    except Exception as e:
        log.error(f"Failed to load checkpoint: {e} — using fresh weights")
        _model = _build_fresh_model()

    return _model


def _build_fresh_model():
    if MODEL_VARIANT == "original":
        m = DefenseOriginal(num_secrets=NUM_SECRETS)
    else:
        m = DefenseAligned(num_secrets=NUM_SECRETS)
    m.eval()
    return m


# ════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════════

def pil_to_tensor(img: Image.Image, size=IMAGE_SIZE) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).detach().float().clamp(0, 1).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))

def save_image(t: torch.Tensor, name: str) -> str:
    path = OUTPUT_DIR / name
    tensor_to_pil(t).save(str(path), "PNG")
    return str(path)

def compute_texture_score(img: Image.Image) -> float:
    arr = np.array(img.convert("RGB").resize((256, 256)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float(), b.float()).item()
    if mse < 1e-10: return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def r_list_to_json(r_list):
    """Serialize r_list tensors to a JSON-safe structure."""
    return [r.detach().cpu().numpy().tolist() for r in r_list]

def r_list_from_json(data):
    """Deserialize r_list from JSON back to tensors."""
    return [torch.tensor(r, dtype=torch.float32) for r in data]


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model_variant": MODEL_VARIANT,
        "model_loaded": Path(MODEL_PATH).exists(),
        "num_secrets": NUM_SECRETS,
    })


@app.route("/conceal", methods=["POST"])
def conceal():
    """
    Expects multipart/form-data:
      cover      : image file
      secret_0   : image file
      secret_1   : image file
      ...
      secret_N-1 : image file (N = 2..5)

    Returns JSON:
      { stego_path, r_list, texture_score }
      or { error }
    """
    # ── Parse cover ──────────────────────────────────────────
    if "cover" not in request.files:
        return jsonify({"error": "cover image required"}), 400
    cover_pil = Image.open(request.files["cover"].stream)

    # ── Texture gate (server-side enforcement) ───────────────
    texture_score = compute_texture_score(cover_pil)
    if texture_score < TEXTURE_MIN:
        return jsonify({
            "error": f"Cover image texture score {texture_score:.1f} is below minimum {TEXTURE_MIN}. "
                     "Please use a more textured image (landscapes, buildings, crowds work well)."
        }), 422

    # ── Parse secrets ────────────────────────────────────────
    secret_pils = []
    for i in range(NUM_SECRETS):
        key = f"secret_{i}"
        if key in request.files:
            secret_pils.append(Image.open(request.files[key].stream))

    if len(secret_pils) < 2:
        return jsonify({"error": "At least 2 secret images required"}), 400

    # Pad to NUM_SECRETS with blank images if fewer uploaded
    while len(secret_pils) < NUM_SECRETS:
        secret_pils.append(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0)))

    # ── Convert to tensors ───────────────────────────────────
    cover_t   = pil_to_tensor(cover_pil)
    secret_ts = [pil_to_tensor(s) for s in secret_pils]

    # ── Inference ────────────────────────────────────────────
    model = load_model()
    job_id = uuid.uuid4().hex[:12]

    try:
        with torch.no_grad():
            if MODEL_VARIANT == "original":
                stego_list, r_list, perm = model.conceal(cover_t, secret_ts)
                # For original variant, store perm index as first r_list item
                perm_tensor = perm.float().unsqueeze(0)
                r_serialized = r_list_to_json([perm_tensor] + r_list)
            else:
                stego_list, r_list = model.conceal(cover_t, secret_ts)
                r_serialized = r_list_to_json(r_list)

        stego = stego_list[-1]
        stego_path = save_image(stego, f"stego_{job_id}.png")
        log.info(f"Conceal job {job_id} done  texture={texture_score:.1f}")

        return jsonify({
            "stego_path":    stego_path,
            "r_list":        r_serialized,
            "texture_score": texture_score,
        })

    except Exception as e:
        log.exception("Conceal failed")
        return jsonify({"error": str(e)}), 500


@app.route("/reveal", methods=["POST"])
def reveal():
    """
    Expects multipart/form-data:
      stego  : image file
      r_list : JSON string (the serialized r_list from /conceal)

    Returns JSON:
      { secrets_paths: [...], psnr_scores: [...] }
      or { error }
    """
    if "stego" not in request.files:
        return jsonify({"error": "stego image required"}), 400
    if "r_list" not in request.form:
        return jsonify({"error": "r_list key required"}), 400

    stego_pil = Image.open(request.files["stego"].stream)
    stego_t   = pil_to_tensor(stego_pil)

    try:
        r_data = json.loads(request.form["r_list"])
    except (json.JSONDecodeError, ValueError):
        return jsonify({"error": "r_list is not valid JSON"}), 400

    try:
        r_tensors = r_list_from_json(r_data)
    except Exception as e:
        return jsonify({"error": f"Failed to deserialize r_list: {e}"}), 400

    model = load_model()
    job_id = uuid.uuid4().hex[:12]

    try:
        with torch.no_grad():
            if MODEL_VARIANT == "original":
                # First tensor is the saved perm
                perm = r_tensors[0].long().squeeze()
                r_list = r_tensors[1:]
                _, secrets_rec = model.reveal([stego_t], r_list, perm)
            else:
                stego_list_mock = [stego_t]  # reveal only needs final stego
                _, secrets_rec = model.reveal(stego_list_mock, r_tensors)

        # Save recovered secrets (only real ones — non-blank)
        paths, psnrs = [], []
        for i, s in enumerate(secrets_rec):
            path = save_image(s, f"secret_{job_id}_{i}.png")
            paths.append(path)
            # We don't have the original here so PSNR is approximate
            psnrs.append(0.0)

        log.info(f"Reveal job {job_id} done — {len(paths)} secrets recovered")
        return jsonify({
            "secrets_paths": paths,
            "psnr_scores":   psnrs,
        })

    except Exception as e:
        log.exception("Reveal failed")
        return jsonify({"error": str(e)}), 500


@app.route("/texture_check", methods=["POST"])
def texture_check():
    """Quick endpoint to check texture score before full conceal."""
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400
    img = Image.open(request.files["image"].stream)
    score = compute_texture_score(img)
    return jsonify({
        "score":   score,
        "ok":      score >= TEXTURE_MIN,
        "minimum": TEXTURE_MIN,
        "message": "Sufficient" if score >= TEXTURE_MIN else
                   f"Too low — needs {TEXTURE_MIN}, got {score:.1f}",
    })


# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info(f"Device: {DEVICE}")
    log.info(f"Model variant: {MODEL_VARIANT}")
    log.info(f"Torch threads: {torch.get_num_threads()}")
    load_model()   # warm up on startup
    app.run(host="0.0.0.0", port=5001, debug=False)