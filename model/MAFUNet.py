import torch
import torch.nn as nn
from HAM import CDGF, SDGF, HAM


class ACA(nn.Module):
    def __init__(self, c_list):
        super().__init__()
        c_list_sum = sum(c_list)
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool2d(2)

        self.reduce_dim = nn.Conv2d(c_list_sum * (1 * 1 + 2 * 2), c_list_sum, kernel_size=1, groups=4)

        self.att1 = nn.Linear(c_list_sum, c_list[0])
        self.att2 = nn.Linear(c_list_sum, c_list[1])
        self.att3 = nn.Linear(c_list_sum, c_list[2])
        self.att4 = nn.Linear(c_list_sum, c_list[3])
        self.att5 = nn.Linear(c_list_sum, c_list[4])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        pooled_list = []
        inputs = [t1, t2, t3, t4, t5]
        for t in inputs:
            pooled_1 = self.avgpool_1(t).flatten(1)  # [B, C]
            pooled_2 = self.avgpool_2(t).flatten(1)  # [B, C*4]
            pooled_list.append(torch.cat([pooled_1, pooled_2], dim=1))  # [B, C*(1+4)]

        att_feat = torch.cat(pooled_list, dim=1)  # [B, c_list_sum * (1+4)]

        att_feat_reshaped = att_feat.unsqueeze(-1).unsqueeze(-1)
        att_feat_reduced = self.reduce_dim(att_feat_reshaped).squeeze(-1).squeeze(-1)

        att1 = self.sigmoid(self.att1(att_feat_reduced)).unsqueeze(-1).unsqueeze(-1).expand_as(t1)
        att2 = self.sigmoid(self.att2(att_feat_reduced)).unsqueeze(-1).unsqueeze(-1).expand_as(t2)
        att3 = self.sigmoid(self.att3(att_feat_reduced)).unsqueeze(-1).unsqueeze(-1).expand_as(t3)
        att4 = self.sigmoid(self.att4(att_feat_reduced)).unsqueeze(-1).unsqueeze(-1).expand_as(t4)
        att5 = self.sigmoid(self.att5(att_feat_reduced)).unsqueeze(-1).unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5


class ASA(nn.Module):
    def __init__(self):
        super().__init__()
        self.level_convs = nn.ModuleList([
            nn.ModuleDict({
                'conv1': nn.Sequential(
                    nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                ),
                'conv2': nn.Sequential(
                    nn.Conv2d(2, 1, 3, stride=1, padding=1),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                )
            }) for _ in range(5)
        ])

        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor([0.5])) for _ in range(5)
        ])

    def forward(self, t1, t2, t3, t4, t5):
        features = [t1, t2, t3, t4, t5]
        att_outputs = []

        for idx, (feat, convs) in enumerate(zip(features, self.level_convs)):
            avg_out = torch.mean(feat, dim=1, keepdim=True)
            max_out, _ = torch.max(feat, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)

            att1 = convs['conv1'](att)
            att2 = convs['conv2'](att)

            att = self.weights[idx] * att1 + (1 - self.weights[idx]) * att2

            att_outputs.append(att)

        return att_outputs[0], att_outputs[1], att_outputs[2], att_outputs[3], att_outputs[4]


class ACA_ASA(nn.Module):
    def __init__(self, c_list):
        super().__init__()

        self.catt = ACA(c_list)
        self.satt = ASA()
        self.sdgf1 = SDGF()
        self.sdgf2 = SDGF()
        self.sdgf3 = SDGF()
        self.sdgf4 = SDGF()
        self.sdgf5 = SDGF()
        self.cdgf1 = CDGF()
        self.cdgf2 = CDGF()
        self.cdgf3 = CDGF()
        self.cdgf4 = CDGF()
        self.cdgf5 = CDGF()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        t1, t2, t3, t4, t5 = (self.sdgf1([t1, r1]),
                              self.sdgf2([t2, r2]),
                              self.sdgf3([t3, r3]),
                              self.sdgf4([t4, r4]),
                              self.sdgf5([t5, r5]))

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        x1, x2, x3, x4, x5 = (self.cdgf1([r1_, t1]),
                              self.cdgf2([r2_, t2]),
                              self.cdgf3([r3_, t3]),
                              self.cdgf4([r4_, t4]),
                              self.cdgf5([r5_, t5]))

        return x1, x2, x3, x4, x5


class DC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.MAF = MAF(in_channels=in_channels)
        self.channel = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.maxpool(x)
        x_c = x
        x_h = x
        x = self.MAF(x_c, x_h)
        x = self.channel(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DC(in_channels)
        self.channel = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.channel(x)
        return x


class MAF(nn.Module):
    def __init__(self, in_channels, reduction_ratio_a=16):
        super(MAF, self).__init__()
        self.HAM = HAM(in_channels)
        self.ResDoubleConv = DC(in_channels)
        self.fc_a = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 2 // reduction_ratio_a),
            nn.Linear(in_channels * 2 // reduction_ratio_a, in_channels),
            nn.Sigmoid()
        )

        self.fc_s = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
        )
        self.sigmoid_s = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_c, x_h):
        x_c = self.ResDoubleConv(x_c)
        x_h = self.HAM(x_h)

        x_concat_ch = torch.cat((x_c, x_h), dim=1)
        B, C_concat, H, W = x_concat_ch.size()
        a_input = x_concat_ch.mean(dim=[2, 3])
        a = self.fc_a(a_input)
        a = a.sigmoid().view(B, -1, 1, 1)

        x_ch = a * x_c + (1 - a) * x_h

        x_add_ch_c = self.relu(x_ch + x_c)
        s_input = x_add_ch_c.mean(dim=1, keepdim=True)
        s_fc = self.fc_s(s_input)
        s = self.sigmoid_s(s_fc)
        output = s * x_c + (1 - s) * x_ch
        return output


class MAFUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.c_list = [16, 32, 64, 128, 256, 512]

        self.aca_asa = ACA_ASA(self.c_list[:-1])

        self.initial_conv = nn.Conv2d(in_channels, self.c_list[0], kernel_size=3, padding=1)

        self.down1 = Down(self.c_list[0], self.c_list[1])

        self.down2 = Down(self.c_list[1], self.c_list[2])

        self.down3 = Down(self.c_list[2], self.c_list[3])

        self.down4 = Down(self.c_list[3], self.c_list[4])

        self.down5 = Down(self.c_list[4], self.c_list[5])

        self.up1 = Up(self.c_list[5], self.c_list[4])

        self.up2 = Up(self.c_list[4], self.c_list[3])

        self.up3 = Up(self.c_list[3], self.c_list[2])

        self.up4 = Up(self.c_list[2], self.c_list[1])

        self.up5 = Up(self.c_list[1], self.c_list[0])

        self.outc = nn.Conv2d(self.c_list[0], num_classes, kernel_size=1)

        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x1, x2, x3, x4, x5 = self.aca_asa(x1, x2, x3, x4, x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    model = MAFUNet()
    model = model.cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    y = model(input_tensor)
    print(y.shape)
