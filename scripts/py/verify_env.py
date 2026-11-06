"""打印 see_through 环境的关键依赖版本（供 setup_env.ps1 调用）。"""
import sys

print('python', sys.version.split()[0])
import torch
print('torch', torch.__version__)
print('cuda_build', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
for name in ('transformers', 'diffusers', 'accelerate', 'psd_tools'):
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(name, 'IMPORT FAILED:', e)
try:
    import bitsandbytes as bnb
    print('bitsandbytes', bnb.__version__)
except Exception as e:
    print('bitsandbytes IMPORT FAILED:', e)
