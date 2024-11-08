import torch
import torch.nn as nn
import math

# BatchNorm helper function
def get_norm(norm, num_features):
    return nn.BatchNorm2d(num_features)

class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=1024, out_channels_list=[192, 384, 768, 1536]):
        super(FeaturePyramid, self).__init__()
        self.stages = nn.ModuleList()

        # Scale factors to match the output dimensions for res2, res3, res4, and res5
        scale_factors = [4.0, 2.0, 1.0, 0.5]
        strides = [4, 2, 1, 0.5]  # strides matching each output resolution

        for idx, scale in enumerate(scale_factors):
            out_dim = in_channels  # Start from input channels (1024)
            out_channels = out_channels_list[idx]

            if scale == 4.0:  # Upsample for res2
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    get_norm("batch", in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2),
                ]
            elif scale == 2.0:  # Upsample for res3
                layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)]
            elif scale == 1.0:  # Identity for res4
                layers = []  # Identity layer (no transformation for res4)
            elif scale == 0.5:  # Downsample for res5
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            # Add 1x1 conv + norm + 3x3 conv with the required output channels
            if scale == 4.0 or scale == 2.0:
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                        get_norm("batch", out_channels),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        get_norm("batch", out_channels),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    ]
                )
                
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        res2 = self.stages[0](x)  # b, 192, 96, 96
        res3 = self.stages[1](x)  # b, 384, 48, 48
        res4 = self.stages[2](x)  # b, 768, 24, 24
        res5 = self.stages[3](x)  # b, 1536, 12, 12

        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}

# 示例用法
input_tensor = torch.randn(1, 1024, 24, 24)
model = FeaturePyramid()
output = model(input_tensor)
for k, v in output.items():
    print(f"{k}: {v.shape}")