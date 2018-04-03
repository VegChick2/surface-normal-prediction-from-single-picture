local train_depth_path = nil 
local valid_depth_path = nil 

local folderpath = '../../data/'

if g_args.t_depth_file ~= '' then
	train_depth_path = folderpath .. g_args.t_depth_file
end

if g_args.v_depth_file ~= '' then
	valid_depth_path = folderpath .. g_args.v_depth_file
end



if train_depth_path == nil then
	print("Error: Missing training file for depth!")
	os.exit()
end

if valid_depth_path == nil then
	print("Error: Missing validation file for depth!")
	os.exit()
end

------------------------------------------------------------------------------------------------------------------


function TrainDataLoader()	
	return DataLoader(train_depth_path,20000)   		
end

function ValidDataLoader()
    return DataLoader(valid_depth_path,2000)
end

