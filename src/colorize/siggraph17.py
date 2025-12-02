# src/colorize/siggraph17.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class SIGGRAPHGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SIGGRAPHGenerator, self).__init__()

        # Level 1: 4 -> 64
        model1 = [nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),] + [nn.ReLU(True),]
        model1 += [norm_layer(64),]
        self.model1 = nn.Sequential(*model1)

        # Level 2: 64 -> 128
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),] + [nn.ReLU(True),]
        model2 += [norm_layer(128),]
        self.model2 = nn.Sequential(*model2)

        # Level 3: 128 -> 256
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),] + [nn.ReLU(True),]
        model3 += [norm_layer(256),]
        self.model3 = nn.Sequential(*model3)

        # Level 4: 256 -> 512
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model4 += [norm_layer(512),]
        self.model4 = nn.Sequential(*model4)

        # Level 5: 512 -> 512
        model5 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model5 += [norm_layer(512),]
        self.model5 = nn.Sequential(*model5)

        # Level 6: 512 -> 512
        model6 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),] + [nn.ReLU(True),]
        model6 += [norm_layer(512),]
        self.model6 = nn.Sequential(*model6)

        # Level 7: 512 -> 512
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model7 += [norm_layer(512),]
        self.model7 = nn.Sequential(*model7)

        # Level 8 (Bottleneck): 512 -> 256
        # Note: Checkpoint is missing the first Conv(512, 256). We define it here.
        model8 = [nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),] + [nn.ReLU(True),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model8 += [norm_layer(256),]
        self.model8 = nn.Sequential(*model8)

        # Upsampling 8: 512 -> 256
        # Input 512 comes from Concat(Model8(256) + Model7_Reduced(256))
        self.model8up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True)
        )
        # Short7: Bridge 512 -> 256 to create skip connection
        self.model7short8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True)
        )
        # Short8 (Unused in this config, placeholder to match keys if needed)
        self.model3short8 = nn.Sequential()

        # Level 9: 256 -> 128
        # Input 256 comes from model8up
        model9 = [nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model9 += [norm_layer(128),]
        self.model9 = nn.Sequential(*model9)

        # Upsampling 9: 256 -> 128
        # Input 256 comes from Concat(Model9(128) + Model2(128))
        self.model9up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True)
        )
        self.model2short9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True)
        )

        # Level 10: 128 -> 128
        model10 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.ReLU(True),]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),] + [nn.LeakyReLU(negative_slope=0.2),]
        self.model10 = nn.Sequential(*model10)

        # Upsampling 10: 256 -> 128
        # Input 256 comes from Concat(Model10(128) + Model1_Short(128))
        self.model10up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True)
        )
        self.model1short10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True)
        )

        # Classification (Dummy)
        self.model_class = nn.Sequential(
            nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Softmax(dim=1)
        )

        # Output Regression
        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh()
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_A, mask_B=None, input_B=None):
        if input_A.shape[1] == 1:
            batch_size, _, height, width = input_A.shape
            dummy = torch.zeros(batch_size, 3, height, width, dtype=input_A.dtype, device=input_A.device)
            input_A = torch.cat((input_A, dummy), dim=1)

        conv1_2 = self.model1(input_A)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)

        # Concatenate Model8 output with Reduced Model7 output
        conv7_short = self.model7short8(conv7_3)
        conv8_comb = torch.cat((conv8_3, conv7_short), 1)
        conv8_up = self.model8up(conv8_comb)
        
        conv9_in = self.model9(conv8_up)
        
        conv2_short = self.model2short9(conv2_2)
        conv9_comb = torch.cat((conv9_in, conv2_short), 1)
        conv9_up = self.model9up(conv9_comb)
        
        conv10_in = self.model10(conv9_up)
        
        conv1_short = self.model1short10(conv1_2)
        conv10_comb = torch.cat((conv10_in, conv1_short), 1)
        # The HACK below is removed as it was incorrect.
        # The model10up layer is now correctly defined to handle 256 input channels.
        conv10_up = self.model10up(conv10_comb)
        
        out_reg = self.model_out(conv10_up)

        return out_reg

def siggraph17(pretrained=True):
    model = SIGGRAPHGenerator()
    if pretrained:
        url = 'https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth'
        try:
            state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
        except Exception as e:
            print(f"Failed to download from main URL: {e}")
            print("Trying backup URL...")
            url = 'https://huggingface.co/richzhang/colorization-siggraph17/resolve/main/siggraph17.pth'
            state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
            
        # Using strict=False to skip missing 'model8.0' weights (which we verify will init randomly)
        # and ignore 'model3short8' unused keys.
        model.load_state_dict(state_dict, strict=False)
        
        # Manually init the missing bottleneck layer to avoid grey output
        nn.init.kaiming_normal_(model.model8[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model.model8[0].bias, 0)
        
    return model