import torch
from gsplat import rasterization
dev = "cuda"
N = 500
torch.manual_seed(0)
means = torch.randn(N, 3, device=dev); means[:, 2] += 6.0
quats = torch.randn(N, 4, device=dev); quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=dev) * 0.1
opac = torch.rand(N, device=dev)
colors = torch.rand(N, 3, device=dev)
viewmats = torch.eye(4, device=dev)[None]
K = torch.tensor([[[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]]], device=dev)

radial = torch.tensor([[0.01, 0., 0., 0., 0., 0.]], device=dev, requires_grad=True)
tang   = torch.tensor([[0., 0.]], device=dev, requires_grad=True)

img, alpha, meta = rasterization(
    means, quats, scales, opac, colors, viewmats, K, 320, 240,
    rasterize_mode="classic", with_ut=True, with_eval3d=True,
    radial_coeffs=radial, tangential_coeffs=tang, packed=False, near_plane=0.01)
img.sum().backward()
print("radial.grad      :", radial.grad)
print("tangential.grad  :", tang.grad)
print("=> radial LEARNABLE:", radial.grad is not None and radial.grad.abs().sum().item() > 0)
print("=> tangent LEARNABLE:", tang.grad is not None and tang.grad.abs().sum().item() > 0)
