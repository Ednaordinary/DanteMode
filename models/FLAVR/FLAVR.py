
import tqdm
import torch
from PIL import Image
import numpy as np
import torchvision

from .FLAVR_arch import UNet_3D_3D

def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, vid):
        return resize(vid, self.size, interpolation_mode="bilinear")

def video_transform(videoTensor , downscale=1):
    
    T , H , W = videoTensor.size(0), videoTensor.size(1) , videoTensor.size(2)
    downscale = int(downscale * 8)
    resizes = 8*(H//downscale) , 8*(W//downscale)
    transforms = torchvision.transforms.Compose([ToTensorVideo() , Resize(resizes)])
    videoTensor = transforms(videoTensor)
    
    return videoTensor , resizes

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def to_tensor(clip):
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "clip tensor should have data type uint8. Got %s" % str(clip.dtype)
        )
    return clip.float().permute(3, 0, 1, 2) / 255.0

class ToTensorVideo(object):
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor(clip)

class FLAVRModel:
    def __init__(self, path):
        self.model = UNet_3D_3D("unet_18" , n_inputs=4, n_outputs=7,  joinType="concat" , upmode="transpose")
        saved_state_dict = torch.load(path, weights_only=True)
        self.model.load_state_dict(saved_state_dict)
        self.model.eval()
    def __call__(self, frames):
        self.model.cuda()
        images = [torch.Tensor(np.asarray(f)).type(torch.uint8) for f in frames]
        videoTensor = torch.stack(images)
        videoTensor = videoTensor.squeeze()
        print(videoTensor.shape)
        idxs = torch.Tensor(range(len(videoTensor))).type(torch.long).view(1,-1).unfold(1,size=4,step=1).squeeze(0)
        videoTensor , resizes = video_transform(videoTensor , 1)
        frames = torch.unbind(videoTensor , 1)
        n_inputs = len(frames)
        width = 8
        outputs = []
        outputs.append(frames[idxs[0][1]])
        for i in tqdm.tqdm(range(len(idxs))):
            yield (i, len(idxs))
            idxSet = idxs[i]
            inputs = [frames[idx_].cuda().unsqueeze(0) for idx_ in idxSet]
            with torch.no_grad():
                outputFrame = self.model(inputs)
            outputFrame = [of.squeeze(0).cpu().data for of in outputFrame]
            outputs.extend(outputFrame)
            outputs.append(inputs[2].squeeze(0).cpu().data)
        outputs = [Image.fromarray(output.data.mul(255.).clamp(0,255).round().permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for output in outputs]
        self.model.cpu()
        yield outputs
