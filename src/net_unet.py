# only works for square images (from cropping)

import torch
from torch import nn


class EncoderBlock(nn.Module): 

    def __init__(self, pos, chn_in, chn_out, k_conv=3, k_mp=2, s_mp=2): 
        ''' pos: outer or inner block
            k_mp: max-pooling layer kernel size
            s_mp: max-pooling layer stride '''

        super(EncoderBlock, self).__init__()

        self.pos = pos

        self.encoder = nn.Sequential(
            nn.Conv2d(chn_in, chn_out, k_conv),
            nn.ReLU(True),
            nn.Conv2d(chn_out, chn_out, k_conv),
            nn.ReLU(True), 
        )

        self.max_pool = nn.MaxPool2d(k_mp, s_mp)


    def forward(self, x): 
        ''' Add max pooling layer if inner block. '''        

        if self.pos == 'inner': 
            x = self.max_pool(x)

        return self.encoder(x)


class DecoderBlock(nn.Module): 

    def __init__(self, pos, chn_in, chn_out, chn_convt=None, k_conv=3, k_convt=2, k_mp=2, s_mp=2): 
        ''' pos: outer, inner, or middle block '''

        super(DecoderBlock, self).__init__()
        
        self.pos = pos

        self.decoder = nn.Sequential(
            nn.Conv2d(chn_in, chn_out, k_conv),
            nn.ReLU(True),
            nn.Conv2d(chn_out, chn_out, k_conv),
            nn.ReLU(True), 
        )

        self.max_pool = nn.MaxPool2d(k_mp, s_mp)

        if pos in ('middle', 'inner'): 
            self.convt = nn.ConvTranspose2d(chn_out, chn_convt, k_convt, stride=2)
        elif pos == 'outer': 
            self.conv = nn.Conv2d(chn_out, 2, 1)
            self.softmax = nn.Softmax(dim=1)


    def crop_skip_layer(self, x, skip): 
        ''' crop skip connection to match size '''

        size_diff = skip.size()[2] - x.size()[2] 
        extra_pad = 1 if size_diff % 2 != 0 else 0

        pad_size = size_diff//2

        skip = skip[:, :, (pad_size+extra_pad):-pad_size, (pad_size+extra_pad):-pad_size]

        assert skip.size() == x.size(), f'size of skip connection ({skip.size()}) does not match layer size ({x.size()})!'

        return skip


    def forward(self, x, skip=None): 
        ''' Add conv trans layer if inner block. '''

        if self.pos == 'inner': 
            x = self.decoder(x)
            x = self.convt(x)
            skip = self.crop_skip_layer(x, skip)
            return torch.cat([x, skip], 1)

        elif self.pos == 'outer': 
            x = self.decoder(x)
            x = self.conv(x)
            return self.softmax(x)

        elif self.pos == 'middle': 
            x = self.max_pool(x)
            x = self.decoder(x)
            x = self.convt(x)
            skip = self.crop_skip_layer(x, skip)
            return torch.cat([x, skip], 1)


class UNet(nn.Module): 

    def __init__(self, n_bands=3, k=3): 
        ''' n_bands: number of bands/channels of input image '''

        super(UNet, self).__init__()

        #           0    1    2    3    4
        self.chn = (64, 128, 256, 512, 1024)

        self.encoder1 = EncoderBlock('outer', n_bands, self.chn[0])
        self.encoder2 = EncoderBlock('inner', self.chn[0], self.chn[1])
        self.encoder3 = EncoderBlock('inner', self.chn[1], self.chn[2])
        self.encoder4 = EncoderBlock('inner', self.chn[2], self.chn[3])

        self.middle = DecoderBlock('middle', self.chn[3], self.chn[4], self.chn[3]) # 512, 1024

        self.decoder1 = DecoderBlock('inner', self.chn[4], self.chn[3], self.chn[2]) # 1024, 512
        self.decoder2 = DecoderBlock('inner', self.chn[3], self.chn[2], self.chn[1]) # 512, 256
        self.decoder3 = DecoderBlock('inner', self.chn[2], self.chn[1], self.chn[0])
        self.decoder4 = DecoderBlock('outer', self.chn[1], self.chn[0])


    def forward(self, x): 

        x = x.float()
        
        # encoder
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        # middle
        x = self.middle(skip4, skip4)

        # decoder
        x = self.decoder1(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)
        x = self.decoder4(x)

        return x



if __name__ == '__main__': 

    from torchinfo import summary

    model = UNet()
    img_size = 400
    input_size = (1,3,img_size,img_size)
    sample = torch.zeros(input_size)
    x = model(sample)

    # print model info
    summary(model, input_size)

    print('input: ', sample.size())
    print('output: ', x.size())