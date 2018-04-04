require 'hdf5'
require('optim')
require('os')
require('cunn')
require('paths')
require 'image'
require 'xlua'
require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'cudnn'
require 'cunn'
require 'cutorch'
require './models/world_coord_to_normal'

g_model = torch.load("/root/442project/src/results/hourglass3/lr_001_bs_4/model_period3_30000.t7");

g_model:evaluate();



color = torch.Tensor();    
color:resize(1, 3, 128, 128); 



for i = 0, 100 do

    c =image.load("../../data/train/color/".. tostring(i) ..".png");
    n =image.load("../../data/train/normal/".. tostring(i) ..".png");
    m =image.load("../../data/train/mask/".. tostring(i) ..".png");
    color[{1,{}}]:copy(c);


    out = g_model:forward(color:cuda());  
    im=out[{1,{}}]
    n=torch.cdiv(n,torch.norm(n,2,1):repeatTensor(3,1,1))
    im=torch.acos(torch.clamp(torch.sum(torch.cmul(im, n:cuda()),1),-1,1))
    im=im/3.15
    im=torch.cmul(im,m:cuda())
    image.save("out_2/" .. tostring(i) .. ".png", im)
    print(i)

end
