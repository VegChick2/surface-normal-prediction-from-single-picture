from PIL import Image
import numpy as np
import lmdb
import caffe


# Let's pretend this is interesting data
N=20000

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size =1024*1024*1024*5

env = lmdb.open('/home/qty/442tp2/eecs442challenge/train/mylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        image0 = Image.open("/home/qty/442tp2/eecs442challenge/train/color/"+str(i)+".png")
        image1 = Image.open("/home/qty/442tp2/eecs442challenge/train/normal/"+str(i)+".png")
        image2 = Image.open("/home/qty/442tp2/eecs442challenge/train/mask/"+str(i)+".png")
        image0 = np.transpose(image0, (2, 0, 1))
        image1 = np.transpose(image1, (2, 0, 1))
        image2 = np.expand_dims(image2, 2)
        image2 = np.transpose(image2, (2, 0, 1))
        image0=image0/255
        image1=image1/255
        image2=image2/255
        X=np.concatenate((image0, image1, image2), axis=0)
        datum = caffe.io.array_to_datum(X, 0)
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
