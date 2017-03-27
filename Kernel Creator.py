''' Sev's New Convolutional Edge Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | a | b | c | ->  | 1 | 7 | 3 | ~~~ Sev's "base" notation for 3x3 convolutions
 | d | e | f | ->  | 5 | 9 | 6 | 
 | g | h | i | ->  | 2 | 8 | 4 | 

 | a | b | -> | 1 | 2 |  ~~~ Sev's "base" notation for 2x2 convolutions
 | c | d | -> | 3 | 4 |  

Functions
~~~~~~~~~~~
1. bw -> takes np array of image, converts to np array of BW image
2. ed2 -> convolutional edge detector
3. cv3 -> general 3x3 convol, takes a list of 9 weights, [[1,2,3],[4,5...
4. sev_detect -> sev's experimental 2x2 edge detector
5. gen_conv2 -> general convol, takes a list of 4 weights [[1,2],[3,4]]
6. inv -> inverses the colors
7. plot_im -> plots images and graphs for kernel_guesser
8. ed00 -> simple black and white edge detector
9. kernel_guesser -> tensorflow kernel guessing algorithm

TODO
~~~~~~~~~~~
1. add canny edge detection
2. figure out the numerical intervals of the intensities and corresponding color
3. measure entropy of images to auto detect parameters! 
'''
import time
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import tensorflow as tf

def s(base): # simple way to show image instead of typing out the whole thing ;)
    im.fromarray(base).show()
def bw(base):
    '''takes np array of image, converts to np array of BW image'''
    R = base[:,:,0]
    G = base[:,:,1]
    B = base[:,:,2]
    return np.round( np.sum( [R,G,B], axis=0 ) / 3.0 )
    #im.fromarray(np.round(np.sum([R,G,B],axis=0)/3.0).show()#to display BW
def inv(base):
    return 255-base

red = [] # SEV'S COLOR ZONE
green = []
blue = []
for i in range(256): # First part of color wheel
    red.append(255)
    blue.append(0)
    green.append(i)
for i in range(256):
    red.append(256-i)
    blue.append(0)
    green.append(255)
for i in range(256):
    red.append(0)
    blue.append(i)
    green.append(255)
for i in range(256):
    red.append(0)
    blue.append(255)
    green.append(256-i)
for i in range(256):
    red.append(i)
    blue.append(255)
    green.append(0)
for i in range(256): 
    red.append(255)
    blue.append(256-i)
    green.append(0)
colors = np.transpose(np.array([red,green,blue,np.tile(255,1536)]))
#colors2 = np.asarray(np.reshape(colors,[48,32,4]),dtype='uint8')
#im.fromarray(colors2).show()   

base = '' # directory to your image
base = im.open(base)
base = np.asarray(base)

dir1 = '' # directory to your folder to save graph output

# K E R N E L S
gaus = [0,1,0,2,4,2,0,1,0] # Gaussian
sharp = [0,-1,0,-1,5,-1,0,-1,0] # sharpen
blur = [1,1,1,1,1,1,1,1,1] # blur
enh = [0,0,0,-1,1,0,0,0,0] # edge enhance
edge = [0,1,0,1,-4,1,0,1,0] # gimp edge detect
emb = [-2,-1,0,-1,1,1,0,1,2]
lap = [0,-1,0,-1,4,-1,0,-1,0]


def ent(base,t=20):
    '''sevs attempt to implement an entropy measure'''
    base1 = base[:-1,:-1]
    base2 = base[:-1,1:]
    base3 = base[1:,:-1]
    base4 = base[1:,1:]
    v = np.subtract(base1,base2)
    h = np.subtract(base1,base3)
    d = np.subtract(base1,base4)
    vd = v
    hd = h
    dd = d
    vd[np.abs(vd)>t]=0
    vd[np.abs(vd)<=t]=1
    hd[np.abs(hd)>t]=0
    hd[np.abs(hd)<=t]=0
    dd[np.abs(dd)>t]=0
    dd[np.abs(dd)<=t]=1
    vn = np.multiply(v,vd)
    hn = np.multiply(h,hd)
    dn = np.multiply(d,dn)
    vert = np.round( np.sum([-1*base1,-1*base2,base3,base4], \
                                 axis = 0) / 4.0 )
    horz = np.round( np.sum([-1*base1,base2,-1*base3,base4], \
                                 axis = 0) / 4.0 )
    horz[horz==0]=0
    mag = np.sqrt((np.sum([np.square(vert),np.square(horz)])))
    print((np.mean(mag),np.std(mag),'mean std of mag'))
    mag[mag>t]=0
    
    e1 = im.fromarray(edge_vert).show()
    e2 = im.fromarray(edge_horiz).show()

def ed(base,b=0,t=10,c=0): # c=0,1 canny T/F 
    canny_bot = b # ~~~~~~~// H Y P E R   P A R M E T E R S //
    canny_top = t 
    base1 = base[:-2,:-2]
    base2 = base[2:,:-2]
    base3 = base[:-2,2:]
    base4 = base[2:,2:]
    base5 = base[1:-1,:-2]
    base6 = base[1:-1,2:]
    base7 = base[:-2,1:-1]
    base8 = base[2:,1:-1]
    edge_vert = np.round( np.sum([-1*base1,-1*base2,base3,base4, \
                                  -2*base5,2*base6],axis=0) / 8.0 )
    edge_horiz = np.round( np.sum([-1*base1,-1*base3,-2*base7,2*base8, \
                                   base2,base4],axis=0) / 8.0 )
    total_grad = np.sqrt(np.sum([np.square(edge_vert),np.square(edge_horiz)],axis=0))
    total_grad = total_grad*(255/(np.max(total_grad)+1))
    print((np.mean(total_grad),np.std(total_grad),np.min(total_grad),np.max(total_grad)))
    #s(total_grad)

    before_canny = total_grad
    total_grad[total_grad<canny_bot]=0

    edge_horiz[ edge_horiz == 0 ] = 0.1
    tan = np.arctan( np.divide(edge_vert,edge_horiz))
    theta=np.array(((np.pi/2.0)+tan)*488,dtype='uint16') #pythagorean magnitude
    print((np.mean(theta),np.std(theta),np.min(theta),np.max(theta)))
    im_grad_colors = np.asarray(colors[theta],dtype = 'uint8')

    cols = []
    tg2 = total_grad
    sev = 40
    tg2[tg2<sev] = 0
    tg2[tg2>=sev] = 255
    
    s(tg2)
    for i in range(3):
        cols.append(np.multiply(im_grad_colors[:,:,i],tg2/np.max(tg2)))#total_grad/(np.max(total_grad)+1)))
    im_grad2 = np.asarray([cols[0],cols[1],cols[2],im_grad_colors[:,:,3]+100],dtype = 'uint8')
    print(np.mean(im_grad_colors[:,:,3]))
    im_grad2 = np.moveaxis(im_grad2,0,-1)
    s(im_grad2)

















    

def ed0(base,b=0,t=10,d=5,n=256,c=0): # c=0,1 canny T/F 
    '''takes np array of BW image, returns edge detect based on 3x3 conv
    GOOD PARAMS:   0,20,10   0,25,15
    BEST VALUES FOR sev_norm are {1,2,10},50,90,130,140,270,300, and cyclycally on
    '''
    base = bw(base) # NEW FEATURE: AUTO CONVERTS ARRAY TO BW
    canny_bot = b #canny lower bound ~~~~~~~// H Y P E R   P A R M E T E R S //
    canny_top = t #canny upper bound ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    canny_delta = d #how many intensity units defines a connected edge, see **
    sev_norm = n #change relative magnitude of colors for small gradient
    kk = 255 # gradient normalization (stretches [0,max]->[0,255] )~~~~~~~~~~~~
    print((np.mean(base[0]),np.std(base[0]),'mean,std'))
    base1 = base[:-2,:-2]#~~~~~~~~~~^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    base2 = base[2:,:-2]#~~~~~~~~~~/   /-\    /-\   \~~~~~~~~~~~~~~~~~~~~~~~~~~
    base3 = base[:-2,2:]#~~~~~~~~~/   |O) |  |O) |   \~~~~~~~~~~~~~~~~~~~~~~~~~
    base4 = base[2:,2:]#~~~~~~~~C|    |___|  |___|    |D~~~~~~~~~~~~~~~~~~~~~~~
    base5 = base[1:-1,:-2]#~~~~~~~\     "" o o  ""   /~~~~~~~~~~~~~~~~~~~~~~~~~
    base6 = base[1:-1,2:]#~~~~~~~~~~\     <V_V>     /~~~~~~~~~~~~~~~~~~~~~~~~~~
    base7 = base[:-2,1:-1]#~~~~~~~~~~~-____________-~~~~~~~~~~~~~~~~~~~~~~~~~~~
    base8 = base[2:,1:-1]#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    edge_vert = np.round( np.sum([-1*base1,-1*base2,base3,base4, \
                                  -2*base5,2*base6],axis=0) / 8.0 )
    edge_horiz = np.round( np.sum([-1*base1,-1*base3,-2*base7,2*base8, \
                                   base2,base4],axis=0) / 8.0 )
    total_grad = np.sqrt(np.sum([np.square(edge_vert),np.square(edge_horiz)],axis=0))

    total_grad = total_grad*(255/(np.max(total_grad)+1))
    s(total_grad)
    print((np.min(total_grad),np.max(total_grad),'min/ax total grad'))
    
    before_canny = total_grad
    total_grad[total_grad<canny_bot]=0
    if c == 1:
        total_grad[total_grad<canny_bot]=0#decide whether to do this first or after
        total_grad_maybe=total_grad#Keep original values here to be re-added back
        base=total_grad#base will be binary matrix to discard jumpy changes
        total_grad[total_grad<=canny_top]=0#Remove intermediate values from orig 
        base[base>canny_top]=canny_top#base is the set of intermediate values to be filtered
        base1=base[:-1,:-1]# ** My method of Canny implementation: remove lower
        base2=base[:-1,1:]#threshold, then isolate the middle layer of possibly 
        base3=base[1:,:-1]#connected edges, and measure deltas between each pixel
        base4=base[1:,1:]#to see which fall within the "connection criterion"
        delta_right=np.absolute(base1-base2)#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(delta_right.shape)
        delta_right[delta_right>canny_delta]=0#Side note: my implementation is
        delta_bot=np.absolute(base1-base3)#actually pretty genius because it would include
        delta_bot[delta_bot>canny_delta]=0#lines that fade into the distance
        delta_botright=np.absolute(base1-base4)#because itensity delta is relative to its 
        delta_botright[delta_botright>canny_delta]=0#neighbor instead of global
        delta_right=np.insert(delta_right,len(delta_right),0,axis=0)
        delta_right=np.insert(delta_right,0,0,axis=1)
        delta_bot=np.insert(delta_bot,len(delta_bot[0]),0,axis=1)
        delta_bot=np.insert(delta_bot,0,0,axis=0)
        delta_botright=np.insert(delta_botright,0,0,axis=0)
        delta_botright=np.insert(delta_botright,0,0,axis=1)
        delta_keep=np.sign(np.sum([delta_right,delta_bot,delta_botright],axis=0))
        delta_keep=np.multiply(total_grad_maybe,delta_keep)
        total_grad=np.sum([total_grad,delta_keep],axis=0)
        #im.fromarray(total_grad).show() # * * * * * *
        print((np.min(total_grad),np.max(total_grad),'min/ax total grad'))
        #plt.hist(total_grad.flatten())
        #plt.show()
    #after_canny = im.fromarray(total_grad-before_canny).show() #should show noise
    #after_canny2 = im.fromarray(before_canny-total_grad).show()
    
    edge_horiz[ edge_horiz == 0 ] = 0.1
    tan = np.arctan( np.divide(edge_vert,edge_horiz))
    print((np.min(tan),np.max(tan),'min/max tan'))
    theta=np.array(((np.pi/2.0)+tan)*488,dtype='uint16') #pythagorean magnitude
    print((np.min(theta),np.max(theta),'min/max theta'))
    print(theta.shape)
    im_grad_colors = np.asarray(colors[theta],dtype = 'uint8')
    print((im_grad_colors.shape,'im_grad_colors shape'))
    #im.fromarray(im_grad_colors).show() # all the gradients
    print(total_grad.shape)
    cols = []
    k = kk/(np.max(total_grad)+1) # WORKING BEAUTIFUL GRADIENT
    for i in range(3):
        cols.append(np.multiply(im_grad_colors[:,:,i],k*total_grad/sev_norm))
    im_grad2 = np.asarray([cols[0],cols[1],cols[2],im_grad_colors[:,:,3]],dtype = 'uint8')
    im_grad2 = np.moveaxis(im_grad2,0,-1)
    im.fromarray(im_grad2).show()
    #return im_grad2







def cv3(base, weights):
    '''takes np array of BW image, returns weighted 3x3 conv'''
    base1 = base[:-2,:-2]
    base2 = base[2:,:-2]
    base3 = base[:-2,2:]
    base4 = base[2:,2:]
    base5 = base[1:-1,:-2]
    base6 = base[1:-1,2:]
    base7 = base[:-2,1:-1]
    base8 = base[2:,1:-1]
    base9 = base[1:-1,1:-1]
    convolution = np.round( np.sum([ weights[0]*base1 , weights[1]*base7, \
                                     weights[2]*base3 , weights[3]*base5, \
                                     weights[4]*base9 , weights[5]*base6, \
                                     weights[6]*base2 , weights[7]*base8, \
                        weights[8]*base9],axis=0)/np.sum(np.abs(weights)))
    return np.asarray(convolution,dtype='uint8') #e1 = im.fromarray(convolution).show()
    



# U N D E R   C O N S T R U C T I O N

def sev_detect(base):
    '''takes np array of BW image, returns edge detect based on 3x3 conv'''
    base1 = base[:-1,:-1]
    base2 = base[:-1,1:]
    base3 = base[1:,:-1]
    base4 = base[1:,1:]
    edge_vert = np.round( np.sum([-1*base1,-1*base2,base3,base4], \
                                 axis = 0) / 4.0 )
    edge_horiz = np.round( np.sum([-1*base1,base2,-1*base3,base4], \
                                 axis = 0) / 4.0 )
    e1 = im.fromarray(edge_vert).show()
    e2 = im.fromarray(edge_horiz).show()



def gen_conv2(base,weights):
    '''takes np array of BW image, returns weighted 2x2 conv'''
    base1 = base[:-1,:-1]
    base2 = base[:-1,1:]
    base3 = base[1:,:-1]
    base4 = base[1:,1:]
    convolution = np.round( np.sum([ weights[0]*base1,weights[1]*base2, \
                                     weights[2]*base3,weights[3]*base4], \
                                 axis = 0) / np.sum(np.abs(weights)) )
    return convolution #e1 = im.fromarray(convolution).show()





def plot_im(original, uh, loss1, loss2, e, original_image, f1, base1):
    a=fig.add_subplot(2,3,1)
    plt.plot(loss1, color = 'red', label='Adam Regularized loss')
    plt.plot(loss2, color = 'blue', label='Pixel loss')
    plt.legend()

    a=fig.add_subplot(2,3,2)
    plt.axis('off')
    plt.imshow(uh,cmap='gray')
    a.set_title('Guess, Epoch: '+ str(e))


    a=fig.add_subplot(2,3,3)
    plt.axis('off')
    plt.imshow(base1 ,cmap='gray')
    a.set_title('Desired Output')

    
    a=fig.add_subplot(2,3,4)
    plt.axis('off')
    a.set_title('Original Kernel')
    plt.text(0,0,'[-1,-2,-1]',fontsize=15)
    plt.text(0,0.3,'[0,0,0]',fontsize=15)
    plt.text(0,0.6,'[1,2,1]',fontsize=15)

    a=fig.add_subplot(2,3,5)
    plt.axis('off')
    a.set_title('Kernel Guess')
    f2 = []
    for i in range(3):
        row = []
        for j in range(3):
            new_el = round(f1[i][j],2)
            row.append(new_el)
        f2.append(row)
        
    plt.text(0,0,f2[0],fontsize=15)
    plt.text(0,0.3,f2[1],fontsize=15)
    plt.text(0,0.6,f2[2],fontsize=15)

    a=fig.add_subplot(2,3,6)
    plt.axis('off')
    plt.imshow(original_image,cmap='gray')
    a.set_title('Original Image')

    plt.savefig(dir1 + 'grafu' + str(e) + '.jpg')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

def ed00(base,b=0,t=10,d=5,n=256,c=0): # simple black and white edge detector
    base = bw(base) 
    base1 = base[:-2,:-2]
    base2 = base[2:,:-2]
    base3 = base[:-2,2:]
    base4 = base[2:,2:]
    base5 = base[1:-1,:-2]
    base6 = base[1:-1,2:]
    base7 = base[:-2,1:-1]
    base8 = base[2:,1:-1]
    edge_vert = np.round( np.sum([-1*base1,-1*base2,base3,base4, \
                                  -2*base5,2*base6],axis=0) / 8.0 )
    edge_horiz = np.round( np.sum([-1*base1,-1*base3,-2*base7,2*base8, \
                                   base2,base4],axis=0) / 8.0 )
    total_grad = np.sqrt(np.sum([np.square(edge_vert),np.square(edge_horiz)],axis=0))

    total_grad = total_grad*(255/(np.max(total_grad)+1))
    return total_grad

fig = plt.figure(figsize=(10,7))
def kernel_guesser(base):
    base1 = ed00(base)
    height = base1.shape[0]
    width = base1.shape[1]

    y = np.array([base1[:,:]],dtype='float32') 
    y = np.expand_dims(y, axis = 3)
    x = bw(base)
    x2 = bw(base)
    x = np.array([x[:,:]],dtype='float32') 
    x = np.expand_dims(x, axis = 3)
    height_x = x.shape[1]
    width_x = x.shape[2]
    x_placeholder = tf.placeholder(tf.float32,[1, height_x, width_x,1])
    y_placeholder = tf.placeholder(tf.float32,[1, height, width,1])
    x_in = tf.unstack(x_placeholder)
    y_input = tf.unstack(y_placeholder)

    conv_filter1 = tf.Variable(np.random.rand(3, 3, 1, 1), dtype = tf.float32)
    conv_filter2 = tf.Variable(np.random.rand(3, 3, 1, 1), dtype = tf.float32)
    #conv_filter2 = tf.transpose(conv_filter1 ,[0,2,1,3])

    padding1 = 'VALID'
    strides1 = [1,1,1,1]
    edge_vert = tf.nn.conv2d(x_in, conv_filter1, strides1,padding=padding1)
    edge_vert = edge_vert / tf.reduce_sum(conv_filter1)
    
    edge_horiz = tf.nn.conv2d(x_in, conv_filter1, strides1,padding=padding1)
    edge_horiz = edge_horiz / tf.reduce_sum(conv_filter2)

    total_grad = tf.sqrt(tf.add(tf.square(edge_vert),tf.square(edge_horiz)))
    max_val = tf.reduce_max(total_grad)
    total_grad = total_grad*(255/(max_val+1))

    R = 0.001 #0.001 
    loss = tf.reduce_mean(tf.square(y_input - total_grad)) + R*tf.square(tf.reduce_mean(edge_horiz+edge_vert))
    loss2 = tf.reduce_mean(y_input - total_grad)

    train_step1 = tf.train.AdamOptimizer(0.1).minimize(loss) # 0.1 works
    epochs = 1000
    with tf.Session() as sess:       
        loss_list1 = []
        loss_list2 = []
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            _train_step, _total_grad ,_loss1, _loss2 ,mv ,\
            f1, = sess.run([ train_step1, total_grad, loss, loss2,\
                                    max_val,conv_filter1[:,:,0,0]], 
                feed_dict={ x_placeholder:x,
                            y_placeholder:y})
            loss_list1.append(_loss1)
            loss_list2.append(abs(_loss2*100))
            if e%2==0:
                print(f1)
                print(mv, 'max value')
                uh = np.array(_total_grad, dtype = 'uint8')
                uh = np.squeeze(uh)
                plot_im(base1, uh, loss_list1, loss_list2, e,x2,f1,base1)


kernel_guesser(base)












