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

g_model = torch.load("/root/442project/src/results/hourglass3/new_lr_001_bs_4/model_period1_10000.t7");

g_model:evaluate();



color = torch.Tensor();    
color:resize(1, 4, 128, 128); 



for i = 0, 1999 do

    c =image.load("../../data/test/color/".. tostring(i) ..".png");
    m =image.load("../../data/test/mask/".. tostring(i) ..".png");
    c=torch.cat(c,m,1)
    color[{1,{}}]:copy(c);


    out = g_model:forward(color:cuda());  
    im=out[{1,{}}]
    image.save("out/" .. tostring(i) .. ".png", im)
    print(i)

end
