require('../common/NYU_params')
require('./DataPointer')
require 'image'
require 'xlua'



gmask=torch.Tensor();
gmask:resize(g_args.bs, 1, g_input_height, g_input_width);

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(relative_depth_filename,n_total)    
    print(">>>>>>>>>>>>>>>>> Using DataLoader")         
    
    self.data_ptr_relative_depth = DataPointer(n_total)
    self.relative_depth_filename=relative_depth_filename

    print(string.format('DataLoader init: \n \t%d relative depth samples \n ', n_total))
end



function DataLoader:close()
end





function DataLoader:load_indices( depth_indices)
    local n_depth
    if depth_indices ~= nil then
        n_depth = depth_indices:size(1)
    else
        n_depth = 0
    end


    local batch_size = n_depth

    local color = torch.Tensor();    
    color:resize(batch_size, 4, g_input_height, g_input_width); 



    local normal = torch.Tensor();    
    normal:resize(batch_size, 3, g_input_height, g_input_width); 
    
    --local mask = torch.Tensor();
    --normal:resize(batch_size, 1, g_input_height, g_input_width);

    

    -- Read the relative depth data
    mysum=0
    for i = 1, n_depth do    
        
        local idx = depth_indices[i]-1
        local c =image.load(self.relative_depth_filename ..  string.format("/color/%d.png", idx));
        local n = image.load(self.relative_depth_filename ..  string.format("/normal/%d.png", idx));
        local m = image.load(self.relative_depth_filename ..  string.format("/mask/%d.png", idx));
        c=torch.cat(c,m,1)
        local re
        re = torch.norm(n,2,1)
                                                                    
        n[1]=torch.cdiv(n[1],re);
                                                                    
        n[2]=torch.cdiv(n[2],re);
                                                                     
        n[3]=torch.cdiv(n[3],re);


        n[1]=torch.cmul(n[1],m);

        n[2]=torch.cmul(n[2],m);
        n[3]=torch.cmul(n[3],m);

        --print(string.format("Loading %s", idx))
        mysum=mysum+torch.sum(m)  
        -- read the input image
        color[{i,{}}]:copy(c);    -- Note that the image read is in the range of 0~1
        normal[{i,{}}]:copy(n);
        gmask[{i,{}}]:copy(m)    
        --mask[{i,{}}]:copy(image.load(self.relative_depth_filename .. m));  

        
    end       
    

    return color:cuda(), normal:cuda()
end


function DataLoader:load_next_batch(batch_size)

    
    local depth_indices = self.data_ptr_relative_depth:load_next_batch(batch_size)

    return self:load_indices( depth_indices )
end



function DataLoader:reset()
    self.current_pos = 1
end
