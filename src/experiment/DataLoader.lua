require('../common/NYU_params')
require('./DataPointer')
require 'image'
require 'xlua'






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
    color:resize(batch_size, 3, g_input_height, g_input_width); 



    local normal = torch.Tensor();    
    normal:resize(batch_size, 3, g_input_height, g_input_width); 


    

    -- Read the relative depth data
    for i = 1, n_depth do    
        
        local idx = depth_indices[i]-1
        local img_name = string.format("/color/%d.png", idx);
        local img_name2 = string.format("/normal/%d.png", idx);
       

        print(string.format("Loading %s", img_name))
    
        -- read the input image
        color[{i,{}}]:copy(image.load(self.relative_depth_filename .. img_name));    -- Note that the image read is in the range of 0~1
        normal[{i,{}}]:copy(image.load(self.relative_depth_filename .. img_name2));  

        
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
