import torch.nn as nn
from torchsummary import summary
import torch

# Proper init according to the paper
def weight_init_gan(layer:nn.Module):
    if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.ConvTranspose2d):
        nn.init.normal_(layer.weight.data,0,0.02)
    
    if isinstance(layer,nn.BatchNorm2d): 
        nn.init.normal_(layer.weight.data,1.0,0.02)
        nn.init.constant_(layer.bias.data, 0)
    
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model=nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,64*2,4,2,1),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64*2,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*4,64*8,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*8,1,4,1,0,bias=False),
            nn.Sigmoid()

        )

        self.apply(weight_init_gan)

    def forward(self,x)->torch.Tensor:
        return self.model(x)
    
class Generator(nn.Module):
     def __init__(self) -> None:
        super().__init__()
        
        self.model=nn.Sequential(
            nn.ConvTranspose2d(100,64*8,4,1,0,bias=False,),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d( 64*4,64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d( 64*2,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d( 64,3,4,2,1,bias=False),
            nn.Tanh()
        )
        self.apply(weight_init_gan)
     def forward(self,x)->torch.Tensor:
        return self.model(x)

def get_gen_disc()->tuple[Generator, torch.optim.Adam, Discriminator, torch.optim.Adam, nn.BCELoss]:
    model_gen = Generator().to('cuda')
    optim_gen = torch.optim.Adam(params=model_gen.parameters(),lr=0.0002, betas=(0.5, 0.999))
    model_disc = Discriminator().to('cuda')
    optim_disc =  torch.optim.Adam(params=model_disc.parameters(),lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()
    return model_gen,optim_gen,model_disc,optim_disc,loss_fn


def calc_disc_loss(model_disc:Discriminator,loss_fn,batch:torch.Tensor,targets:torch.Tensor):
    preds = model_disc(batch)
    preds = preds.squeeze() # (batch_size,)
    return loss_fn(preds,targets)

def get_synth_images(model_gen:Generator,nb_images:int,train=False,device='cuda'):
    if train:
        model_gen.train()
        z = torch.randn(size=(nb_images,100,1,1)).to(device)
        return model_gen(z)
    else:
        model_gen.eval()
        with torch.no_grad():
            z = torch.randn(size=(nb_images,100,1,1)).to(device)
            return model_gen(z)

def disc_train_step(model_disc:Discriminator,loss_fn,real_data_batch :torch.Tensor, fake_data_batch:torch.Tensor,device='cuda'):
    model_disc.train()
    len_real = real_data_batch.shape[0]
    targets_real = torch.ones(len_real,dtype=torch.float32).to(device)

    len_fake = fake_data_batch.shape[0]
    targets_fake = torch.zeros(len_fake,dtype=torch.float32).to(device)

    loss_real , loss_fake = calc_disc_loss(model_disc,loss_fn,real_data_batch,targets_real),\
                         calc_disc_loss(model_disc,loss_fn,fake_data_batch,targets_fake)
    
    return loss_fake+loss_real

def gen_train_step(model_gen:Generator,model_disc:Discriminator,nb_images:int,loss_fn,device='cuda'):
    synth_imgs = get_synth_images(model_gen,nb_images,train=True)

    # get preds
    targets_fake = torch.ones(nb_images).to(device)

    model_disc.eval()
    
    loss = calc_disc_loss(model_disc,loss_fn,synth_imgs,targets_fake)

    return loss


if __name__=="__main__":
    model_disc = Discriminator().to('cuda')
    model_gen = Generator().to('cuda')
    image_shape = (3,64,64)
    z_shape = (100,1,1)

    print(summary(model_disc,image_shape))

    print(summary(model_gen,z_shape))
