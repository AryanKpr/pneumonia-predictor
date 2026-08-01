import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from model import Net, preprocess

def generate_gradcam(image: Image.Image, model: Net, device: torch.device) -> str:
    tensor = preprocess(image).to(device)

    gradients = []
    activations = []

    # hooks to capture conv2 output and its gradients
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = model.conv2.register_forward_hook(forward_hook)
    bh = model.conv2.register_full_backward_hook(backward_hook)

    output = model(tensor)
    model.zero_grad()
    class_score = output[0, output.argmax().item()]
    class_score.backward()

    fh.remove()
    bh.remove()

    # compute heatmap
    grads  = gradients[0].squeeze()           # shape: [16, H, W]
    acts   = activations[0].squeeze()         # shape: [16, H, W]
    weights = grads.mean(dim=[1, 2])          # global avg pool over spatial dims

    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0-1

    # resize heatmap to match original image and overlay
    orig = np.array(image.convert("RGB").resize((256, 256)))
    heatmap = cv2.resize(cam, (256, 256))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # encode to base64 so it can be sent in JSON
    pil_overlay = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")