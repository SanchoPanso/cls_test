import torch
print(torch.cuda.is_available())

src_path = '/home/achernikov/CLS/DATA/models/hair_type/v__3_all_eff_48_0.001/checkpoints/epoch=32-step=15543.pt'
dst_path = '/home/achernikov/CLS/DATA/models/hair_type/v__3_all_eff_48_0.001/checkpoints/epoch=32-step=15543_fp16.pt'

extra_files = {"num2label.txt": ""}  # values will be replaced with data
model = torch.jit.load(src_path, 'cuda', _extra_files=extra_files)
model = model.eval()
print('extra_files:', extra_files)

inputs = torch.randn((1, 3, 640, 480), dtype=torch.float32, device='cuda')
with torch.no_grad():
    outputs = model(inputs)
print('fp32 output:', outputs)

model_fp16 = model.half()
inputs = inputs.half()

with torch.no_grad():
    outputs = model(inputs)
print('fp16 output:', outputs)

torch.jit.save(model_fp16, dst_path, _extra_files=extra_files)