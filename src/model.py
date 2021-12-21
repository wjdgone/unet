import torch

from net_unet import UNet

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')


class Model(): 

    def __init__(self, opt): 

        self.opt = opt

        # initialize network
        self.net = UNet()
        self.net.to(device)

        # set loss function 
        self.loss_fn = torch.nn.BCELoss()
        
        # set optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(self.opt['lr']))


    def load_model(self, model_path): 
        self.net.load_state_dict(torch.load(model_path))


    def train_step(self, train_imgs, train_lbls): 

        self.net.train()

        self.train_imgs = train_imgs
        self.train_lbls = train_lbls

        # forward pass
        self.forward_step()

        # backward pass
        self.backward_step()


    def forward_step(self): 

        # prepare for backprop by zeroing gradients
        self.optimizer.zero_grad()

        # forward
        pred = self.net(self.train_imgs)
        self.train_pred = torch.argmax(pred, dim=1, keepdim=True)
        # print(self.train_pred)

        # crop labels down to output size 
        self.train_imgs.size()[-1]
        pad_size = (self.train_imgs.size()[-1] - self.train_pred.size()[-1])//2
        self.train_lbls = self.train_lbls[:, :, pad_size:-pad_size, pad_size:-pad_size]


    def backward_step(self): 

        self.train_pred = self.train_pred.to(torch.float32).to(device)
        self.train_lbls = self.train_lbls.to(torch.float32).to(device)

        self.train_pred = torch.autograd.Variable(self.train_pred, requires_grad=True)
        self.train_lbls = torch.autograd.Variable(self.train_lbls, requires_grad=True)

        loss = self.loss_fn(self.train_pred, self.train_lbls)
        loss.backward()
        self.optimizer.step()


    def test_step(self, imgs, lbls): 

        self.net.eval()

        # feed into network
        pred = self.net(imgs)
        pred = torch.argmax(pred, dim=1)

        imgs = torch.squeeze(imgs, 0)
        lbls = torch.squeeze(lbls, 0)

        img_size = 388
        tot_size = 572
        pad_size = (tot_size - img_size)//2

        imgs = imgs[:, pad_size:-pad_size, pad_size:-pad_size]
        lbls = lbls[:, pad_size:-pad_size, pad_size:-pad_size]

        return imgs, lbls, pred