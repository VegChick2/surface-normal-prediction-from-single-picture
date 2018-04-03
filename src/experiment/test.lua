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

g_model = torch.load("/root/442project/src/results/hourglass3/Ours/model_period1_5000.t7");

g_model:evaluate();



color = torch.Tensor();    
color:resize(1, 3, 128, 128); 



for i = 0, 1999 do

    c =image.load("../../data/test/color/".. tostring(i) ..".png");

    color[{1,{}}]:copy(c);


    out = g_model:forward(color:cuda());  
    im=out[{1,{}}]
    image.save("out/" .. tostring(i) .. ".png", im)
    print(i)

end
