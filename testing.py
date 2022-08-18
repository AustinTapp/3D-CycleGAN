#import SimpleITK as sitk
#import torch.cuda

#label = sitk.ReadImage("C:/Users/pmilab/Desktop/3D CycleGan Data/MRI/Landman_1399_20110819_366886505_101_WIP_SCOUT_SHC32_SCOUT_SHC32.nii.gz")
#print(torch.cuda.is_available())

import torch

print(torch.cuda.is_available())
print("print count", torch.cuda.device_count())
device = torch.cuda.current_device()
print("a5000", device)
print("other:", torch.cuda.device(1))
print(torch.cuda.device(device))
print("other", torch.cuda.get_device_name(1))
print(torch.cuda.get_device_name(device))