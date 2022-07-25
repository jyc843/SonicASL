import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ### **** CNN For in-ear facial gesture recognition ****
        self.conv1 = nn.Conv1d(512, 1024,  kernel_size = 4)
        self.norm1 = nn.InstanceNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size = 16,stride =2)
        self.norm2 = nn.InstanceNorm1d(1024)
        self.conv3 = nn.Conv1d(1024, 512,  kernel_size = 4)
        self.norm3 = nn.InstanceNorm1d(512)

        ### **** CNN For hand gestures recognition ****
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, (0,0)]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        #ipdb.set_trace()
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((4, 4), (4, 4), (0, 0)))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((4, 4), (4, 4), (0, 0)))  # 128x8x32
        convRelu(2, True)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((4, 4), (4, 2), (1, 1)))  # 256x4x16
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((4, 4), (4, 2), (1, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((4, 4), (4, 2), (1, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh , nh, nclass))


    def forward(self, x):

        print('the size of input x', x.shape)
        conv = self.cnn(x[:,:,0:512,:])
        b, c, h, w = conv.size()
        print('the size of conv', conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        x   = x.squeeze(1) # b *512 * width

        out = self.conv1(x[:,512:1024,:])
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)

        conv_upd = torch.cat((conv,out),1)

        conv_upd = conv_upd.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv_upd), dim=2)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(config):

    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)

    return model
