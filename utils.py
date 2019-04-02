import math
import datetime
import vtk_util
import numpy as np
import patching_utils
import tensorflow as tf
import concurrent.futures
from vtk.util.numpy_support import vtk_to_numpy

def convert_to_image_4(path,image_params,varsM,trainflag):
    data = vtk_util.read_vti(path.filepathMLEM)
    # container_extents = (parsedlisttrain['xmin'].values[n],parsedlisttrain['xmax'].values[n], parsedlisttrain['ymin'].values[n],parsedlisttrain['ymax'].values[n], parsedlisttrain['zmin'].values[n], parsedlisttrain['zmax'].values[n])
    container_extents = (242,502,128,1810,155,480)
    container_image = vtk_util.extractVOI(data,*container_extents)
    nx,ny,nz = container_image.GetDimensions()
    imageMLEM = vtk_to_numpy(container_image.GetPointData().GetScalars()).astype(np.float32)
    imageMLEM = imageMLEM.reshape(nz,ny,nx,1)

    data26 = vtk_util.read_vti(path.filepath26)
    container_image26 = vtk_util.extractVOI(data26,*container_extents)
    nx26,ny26,nz26 = container_image26.GetDimensions()
    image26 = vtk_to_numpy(container_image26.GetPointData().GetScalars()).astype(np.float32)
    image26 = image26.reshape(nz26,ny26,nx26,1)

    dataJSS = vtk_util.read_vti(path.filepathJSS)
    container_imageJSS = vtk_util.extractVOI(dataJSS,*container_extents)
    nxJSS,nyJSS,nzJSS = container_imageJSS.GetDimensions()
    imageJSS = vtk_to_numpy(container_imageJSS.GetPointData().GetScalars()).astype(np.float32)
    imageJSS = imageJSS.reshape(nzJSS,nyJSS,nxJSS,1)

    dataFR = vtk_util.read_vti(path.filepathFR)
    # print(path.filepathFR)
    container_imageFR = vtk_util.extractVOI(dataFR,*container_extents)

    nxFR, nyFR, nzFR = container_imageFR.GetDimensions()

    feature_set = varsM
    imageFR = np.empty([nzFR,nyFR,nxFR,0])
    # print(path.filepathFR)
    for feature in feature_set:
        # print('feature', feature)
        interim = vtk_to_numpy(container_imageFR.GetPointData().GetArray(feature)).astype(np.float32)
        interim = interim.reshape(nzFR,nyFR,nxFR,1)
        imageFR = np.concatenate((imageFR,interim),axis=3)

    image = np.array(np.concatenate([imageMLEM,image26,imageJSS,imageFR],axis=3))
    # image = crop_or_pad_along_axis(image,50,2)
    # image = crop_or_pad_along_axis_random(image,140,1)
    # image = crop_or_pad_along_axis_random(image,50,0)
    # print('image_shape', image.shape)
    return image


def convert_to_image_manual(path,image_params,varsM,trainflag):
    new_data = np.loadtxt(path.filepath)
    # the original cut!!
    # image = new_data.reshape(48,340,64, 9)
    image = new_data.reshape(image_params.ORIG_IMAGE_Z,image_params.ORIG_IMAGE_Y,image_params.ORIG_IMAGE_X,9)
    # variable order MLEM, 26, JSS, 'FOMFrac', 'GeoMeanTheta', 'LogPTheta', 'MikesFOM', 'NumStoppedTracks', 'PoCALogScat',
    # 'StoppedMean', 'StoppedThroughRatio', 'GeoMeanFractionalScattering'
    # MLEM, 26, JSS, 'LogPTheta', 'GeoMeanPThetaSquared', 'MeanDoCA', 'NumStoppedTracks', 'GeoMeanFractionalScattering','StoppedMeanP'
    image = patching_utils.crop_or_pad_along_axis_random(image,image_params.CROP_PAD_IMAGE_Y,1)
    image = image[:,:,:,(0,1,2)]
    return image


def convert_to_image(path,image_params,varsM,trainflag):
    # container_extents = (242, 502, 130, 1810, 155, 480) #should be 130
    random_ystart = np.random.randint(130,318)
    random_yend = random_ystart + 1680
    container_extents = (242,502,random_ystart,random_yend,155,480)
    dataFR = vtk_util.read_vti(path.filepath)
    container_imageFR = vtk_util.extractVOI(dataFR,*container_extents)
    nxFR,nyFR,nzFR = container_imageFR.GetDimensions()

    feature_set = varsM
    image = np.empty([nzFR * nyFR * nxFR,0])
    # print(path.filepath)
    for feature in feature_set:
        try:
            # print('feature', feature)
            interim = vtk_to_numpy(container_imageFR.GetPointData().GetArray(feature)).astype(np.float32)
            interim = interim.reshape(nzFR * nyFR * nxFR,1)
            image = np.concatenate((image,interim),axis=1)
        except:
            print(path)
            image = image.reshape(nzFR,nyFR,nxFR,len(feature_set))
            image = image.transpose(2,1,0,3)
            image = image.reshape(nzFR * nyFR * nxFR,len(feature_set))

    if trainflag == 'Train11':
        choice = np.random.randint(1,20)
        if choice > 10:
            image = image
        if choice == 1:
            image = np.flipud(image)
        if choice == 2:
            # image = np.rot90(image)
            image = image
        if choice == 3:
            image = ndimage.gaussian_filter(image,1.5)
        if choice == 4:
            image = ndimage.gaussian_filter(image,0.5)
        if choice == 5:
            image += np.random.randint(0,100) / 1000.0
        if choice == 6:
            # row, col, ch = image.shape
            s_vs_p = 0.35
            amount = 0.1
            # salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [tuple(np.random.randint(0,i - 1,int(num_salt))) for i in image.shape]
            image[coords] = 0
        if choice == 7:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            image = np.random.poisson(image * vals) / np.float(vals)
        if choice == 8:
            image = ndimage.gaussian_filter(image,1)
        if choice == 9:
            image = ndimage.gaussian_filter(image,2)
        if choice == 10:
            # row, col, ch = image.shape
            s_vs_p = 0.35
            amount = 0.30
            # salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [tuple(np.random.randint(0,i - 1,int(num_salt))) for i in image.shape]
            image[coords] = 1
        # image = tf.expand_dims(image, axis = 0)
        return image

def convert_to_image2(path, image_params, varsM, trainflag):
    np.random.seed()
    # container_extents = (242, 502, 130, 1810, 155, 480) #should be 130
    #print(path)
    begin = datetime.datetime.now()
    random_ystart = np.random.randint(130, 318)
    #print(random_ystart)
    random_yend = random_ystart + 1680
    #print(random_yend)
    #print(np.random.get_state()[1][0])
    container_extents = (242, 502, random_ystart, random_yend, 155, 480)
    #print(path)
    dataFR = vtk_util.read_vti(path.filepath)
    container_imageFR = vtk_util.extractVOI(dataFR, *container_extents)
    nxFR, nyFR, nzFR = container_imageFR.GetDimensions()
    feature_set = varsM
    image = np.empty([nzFR * nyFR * nxFR, 0])
    #print(path.filepath)
    for feature in feature_set:
        # print('feature', feature)
        interim = vtk_to_numpy(container_imageFR.GetPointData().GetArray(feature)).astype(np.float32)
        interim = interim.reshape(nzFR * nyFR * nxFR, 1)
        image = np.concatenate((image, interim), axis=1)
    if trainflag == 'Train11':
        choice = np.random.randint(1, 20)
        if choice > 10:
            image = image
        if choice == 1:
            image = np.flipud(image)
        if choice == 2:
            # image = np.rot90(image)
            image = image
        if choice == 3:
            image = ndimage.gaussian_filter(image, 1.5)
        if choice == 4:
            image = ndimage.gaussian_filter(image, 0.5)
        if choice == 5:
            image += np.random.randint(0, 100) / 1000.0
        if choice == 6:
            # row, col, ch = image.shape
            s_vs_p = 0.35
            amount = 0.1
            # salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [tuple(np.random.randint(0, i - 1, int(num_salt))) for i in image.shape]
            image[coords] = 0
        if choice == 7:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            image = np.random.poisson(image * vals) / np.float(vals)
        if choice == 8:
            image = ndimage.gaussian_filter(image, 1)
        if choice == 9:
            image = ndimage.gaussian_filter(image, 2)
        if choice == 10:
            # row, col, ch = image.shape
            s_vs_p = 0.35
            amount = 0.30
            # salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [tuple(np.random.randint(0, i - 1, int(num_salt))) for i in image.shape]
            image[coords] = 1
    # image = tf.expand_dims(image, axis = 0)
    end = datetime.datetime.now()
    #print(end-begin)
    return image


def get_batches_fn_random(batch_size,image_path,params,varsM,batch_number,trainflag='Train'):
    '''
        Create batches, restricting the number of scans per run included.
    '''
    if trainflag == 'Train':
        runs_batch = image_path.iloc[(batch_number * batch_size):(batch_number * batch_size + batch_size),:]
        shuffled = runs_batch
    else:
        shuffled = image_path.sample(n=batch_size,replace=False)
        # shuffled = shuffled.apply(np.random.permutation, axis=0)
    # runs_batch = runs_batch.apply(np.random.permutation, axis=0)
    #print(shuffled)
    # shuffled = runs_batch.sample(n=batch_size, replace=False)
    # shuffled = shuffled.apply(np.random.permutation, axis=0)
    images = []
    for index,row in shuffled.iterrows():
        if trainflag == 'Train':
            image = convert_to_image2(row,params.image_params,varsM,'Train')
            #print(image.shape)
            if len(image) == 0:
                one_random_path = runs_batch.sample(n=1)
                image = convert_to_image2(one_random_path,params.image_params,varsM,'Train')
        else:
            image = convert_to_image2(row,params.image_params,varsM,'Test')
        images.append(image)
    images = np.array(images)
    #print('images', images.shape)
    return images

def fast_batch(batch_size, image_path, params, varsM, trainflag='Train'):
    runs_available = image_path.run.nunique()
    scans_per_run_batch = int(math.ceil(batch_size / runs_available)) + 1
    runs_batch = image_path
    runs_batch = runs_batch.apply(np.random.permutation, axis=0)
    shuffled = runs_batch.sample(n=batch_size, replace=False)
    shuffled = shuffled.apply(np.random.permutation, axis=0)
    rows = []
    for index, row in shuffled.iterrows():
        rows.append(row[4])
    images = []
    begin = datetime.datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_image = {executor.submit(convert_to_image2, path, params.image_params, varsM, 'Train'): path for path in rows}
        finished, pending = concurrent.futures.wait(future_to_image)
        for future in finished:
            image = future.result()
            images.append(image)
    #print(len(images))
    images = np.array(images)
    #print(images)
    end = datetime.datetime.now()
    #print("Fast batch: ", str(end-begin))
    # print('images', images.shape)
    return images

def edges_length_P(P,z,y,x,k):
    '''
    conv per map
    '''
    z1 = z
    z = tf.cast(z,tf.int32)
    y = tf.cast(y,tf.int32)
    x = tf.cast(x,tf.int32)
    k1 = tf.cast(k,tf.int32)
    # make the convolution ourselves https://en.wikipedia.org/wiki/Kernel_(image_processing)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # 4
    convolution = tf.zeros((k1,y - 2,x - 2))  # z-2, sliding on z!
    P = tf.reshape(P,shape=(k1,z,y,x))
    # slide the convolution on z
    for m1 in range(8,15):
        interim_map = tf.zeros((k1,y - 2,x - 2))  # z-2,
        for i in range(3):  # kernel shape
            ic = i - 1  # -1,0,1
            for j in range(3):
                jc = j - 1  # -1,0,1
                interim_map += tf.to_float(kernel[i,j]) * P[:,m1,1 + jc:y - 1 + jc,1 + ic:x - 1 + ic]
        convolution += interim_map  # *class_ratio_penalty(meanR(P[:,:,:,m1]),tf.to_float(k1),1)
    convolution = tf.square(convolution)  # make it positive and more exagerated
    return tf.reduce_sum(convolution) / (tf.reduce_sum(
        P[:,8:15,:,:]))  # tf.exp(-tf.reduce_sum(convolution)/(tf.reduce_sum(P)*tf.reduce_sum(P))) #sum in all axes


def features_similarity(u,v,n_vars,n_clusters,x,y,z):
    u = tf.reduce_sum(tf.to_float(u))
    v = tf.reduce_sum(tf.to_float(v))
    di = tf.square((u - v))
    delta_huber = tf.to_float(0.1)

    def f1(): return tf.constant(10.0)

    def f2(): return tf.constant(1.0)

    def f3(): return tf.constant(5.0)

    sigma_col = tf.to_float(n_clusters * n_vars * n_vars)  # 15
    d = di  # *sigma_col #+
    return (d)  # tf.divide(tf.to_float(15.0), d) #tf.exp(-d) #+ 1.0*tf.exp(-d)


def variance_within(u):
    mean,var = tf.nn.moments(u,axes=[0])
    return tf.to_float(var)


def meanR(X):
    return tf.expand_dims(tf.reduce_mean(X),axis=0)


def cov(X,Y):
    return mean((X - mean(X)) * (Y - mean(X)))


def var(X):
    return cov(X,X)


def wmeanN(X,w,w_sumT):
    w_sum = tf.reduce_sum(tf.to_float(w)) + tf.to_float(0.0001)
    w = tf.to_float(w)
    w = tf.expand_dims(w,1)
    return tf.reduce_sum(X * (w) / w_sum,axis=0)


def wmeanCentr(X,w,w_sumT):
    w_sum = tf.reduce_sum(tf.to_float(w)) + tf.to_float(0.0001)
    w = tf.to_float(w)
    w = tf.expand_dims(w,1)
    return tf.reduce_sum(tf.reduce_sum(X * (w),axis=0))


def wmeanD(X,w,w_sumT):
    return wmeanN(wmeanN(X,w,w_sumT),w,w_sumT)


def wvarN(X,w,w_sumT):
    # return wmeanN(tf.squared_difference(X,wmeanN(X,w,w_sumT)),w, w_sumT)
    centers = X * tf.expand_dims(w,axis=1)
    w_sum = tf.reduce_sum(tf.to_float(w)) + tf.to_float(0.0001)
    # return wmeanN(tf.square(X-centers),w, w_sumT)
    return tf.reduce_sum(tf.square(X - centers) / w_sum,axis=0)


def class_ratio_penalty_pos(r,n_classes,n_variables):
    # return tf.to_float(2*7) - tf.exp(r*7)
    return 1  # tf.to_float(1) - tf.exp(-r*5.0)
    # return tf.to_float(1) + tf.square(r - 0.5)/tf.to_float(n_classes*n_variables)
    # return tf.to_float(1) - tf.square(r - 0.5)#/tf.to_float(n_variables)


def class_ratio_penalty_neg(r,n_classes,n_variables):
    # return tf.to_float(2*7) - tf.exp(r*7)
    return 1  # tf.exp(-r/5.0)


def to_features(img,z,y,x,n):
    '''
    X_col=img
    X_pos=np.zeros((x,y,z,3))
    for i in range(X_pos.shape[0]):
        X_pos[i,:,:,0]=np.float(i+1)/np.float(x)
    for i in range(X_pos.shape[1]):
        X_pos[:,i,:,1]=np.float(i+1)/np.float(y)
    for i in range(X_pos.shape[2]):
        X_pos[:,:,i,2]=np.float(i+1)/np.float(z) #tf.to_float(i+1)/tf.to_float(x)
    X_pos=tf.convert_to_tensor(X_pos,dtype='float32')
    X=tf.concat([X_col,X_pos],axis=3)
    X=tf.reshape(X,(z*y*x,n+3)) #was z*y*x,n+3
    #return X
    '''
    return tf.reshape(img,(z * y * x,n))


def class_mult(b,a):
    return tf.reduce_sum(a * b)


def class_min(a):
    return tf.reduce_min(a)


def centroids_similarity_loss(y_true,y_pred,z,y,x,k,n_variables):
    img = y_true  # tf.reshape(y_true, shape=(x,y,z,n_variables))
    print('img',img.shape)
    # P = y_pred
    img1 = tf.reshape(y_true, shape=(x * y * z, n_variables))  # tf.transpose(y_true, (2,1,0,3))
    P = tf.reshape(y_pred,shape=(z * y * x,k))
    y_pred1 = tf.transpose(y_pred,(3,0,1,2))
    P1 = tf.reshape(y_pred1,shape=(k,z * y * x))
    X = to_features(img,z,y,x,n_variables)
    w_sumTotal = tf.reduce_sum(P)
    centroids = tf.map_fn(lambda i: wmeanN(X,i,w_sumTotal),(P1),dtype='float32')
    variances = tf.map_fn(lambda i: wvarN(img1,i,w_sumTotal),(P1),dtype='float32')
    class_mins = tf.map_fn(elems=np.arange(k),fn=lambda i: class_min(P[:,i]),dtype='float32')
    loss = tf.to_float(0)  # tf.zeros([1], tf.float32)

    for k1 in np.arange(0,k):
        centroid1 = centroids[k1]
        # print('centroid1 {}',centroid1.get_shape())
        centroid_within_similarity = tf.to_float(0)
        for k2 in np.arange(k1 + 1,k):
            centroid2 = centroids[k2]
            # print('centroid2 {}',centroid2.get_shape())
            centroid_within_similarity += features_similarity(centroid1,centroid2,n_variables,k,x,y,z)
        loss -= centroid_within_similarity  # *class_ratio_penalty_pos(class_ratios[k1],k, n_variables)
        centroid_between_similarity = (
        (tf.reduce_sum(variances[k1]) / tf.to_float(k)))  # *class_ratio_penalty_pos(class_ratios[k1],k, n_variables)
        loss += centroid_between_similarity  # /(centroid_between_similarity + 1*tf.abs(centroid_within_similarity)+ tf.to_float(0.001))
        loss += tf.reduce_sum(class_mins[k1] * tf.to_float(k))
    return loss  # + edges_length_P(P1,z, y, x, k )/tf.to_float(1) #k*(k-1)


def target_distribution(q):
    weight = q ** 2 / tf.reduce_sum(q)
    return (weight) / tf.reduce_sum(weight)


def kl_div(y_true,y_pred,z,y,x,k,n_variables):
    img = tf.reshape(y_true,shape=(z * y * x,n_variables))
    P = tf.reshape(y_pred,shape=(z * y * x,k))
    # P= target_distribution(P)
    prob_a = tf.nn.softmax(img)
    cr_aa = tf.nn.softmax_cross_entropy_with_logits(labels=prob_a,logits=prob_a)
    cr_ab = tf.nn.softmax_cross_entropy_with_logits(labels=prob_a,logits=P)
    kl_ab = tf.reduce_sum(cr_ab - cr_aa)
    return kl_ab


def centroids_similarity_loss_no_variance(y_true,y_pred,z,y,x,k,n_variables):
    img = tf.reshape(y_true,shape=(z * y * x,n_variables))
    # y_pred1 = tf.transpose(y_pred, (2,1,0,3))
    P = tf.reshape(y_pred,shape=(z * y * x,k))
    # X=tf.reshape(img, shape=(z*y*x,n_variables))
    X = to_features(y_true,z,y,x,n_variables)
    centroids = tf.matmul(tf.transpose(P),X) / tf.matmul(tf.transpose(P),tf.ones((z * y * x,1)))
    loss = tf.to_float(0,name='loss')
    for k1 in range(k):
        centroid_inter_similarity_mean = tf.to_float(0)
        for k2 in range(k1 + 1,k):
            loss += features_similarity(centroids[k1],centroids[k2],n_variables,k)
    return loss + edges_length_P(P,z,y,x,k) / tf.to_float(
        k)  # (loss + edges_length(P,z, y, x, k ))/(n_variables*k*(k-1)/2)


def tf_pad_along_axis(array,target_length,axis=0):
    pad_size = target_length - tf.shape(array)[axis]
    paddings = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    paddings[axis] = [0,pad_size]
    b = tf.pad(array,paddings=paddings,mode='CONSTANT',constant_values=-2)
    return b


def tf_crop_or_pad_along_axis(array,target_length,axis):
    def f1():
        return tf_pad_along_axis(array,target_length,axis=axis)

    def f2():
        number_to_crop = target_length - tf.shape(array)[axis]
        start_sl = [0,0,0,0,0]
        size_sl = [-1,tf.shape(array)[1],tf.shape(array)[2],tf.shape(array)[3],-1]
        size_sl[axis] = target_length
        return tf.slice(array,begin=start_sl,size=size_sl)

    array = tf.cond(tf.less(tf.shape(array)[axis],target_length),f1,f2)
    return array


def shape_tensor(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0,len(s))])


def tf_equalize_histogram(image):
    '''
    expects normalized image and returns normalized image, all features
    '''
    image = image * tf.to_float(255)
    values_range = tf.constant([0.,255.],dtype=tf.float32)  # 255. was tf.constant
    histogram = tf.histogram_fixed_width(tf.to_float(image),values_range,256)
    cdf = tf.cumsum(histogram)
    # cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    ix = tf.reduce_min(tf.where(cdf > 0))
    cdf_min = tf.gather(cdf,ix)

    img_shape = tf.shape(image)
    pix_cnt = img_shape[1]
    px_map = tf.round(tf.to_float(cdf - cdf_min) * 256 / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map,tf.int32)
    image = tf.cast(image,tf.int32)
    eq_hist = tf.gather(px_map,image)
    eq_hist = tf.to_float(eq_hist) / tf.to_float(256.0)
    # eq_hist = (tf.to_float(eq_hist) - tf.expand_dims(tf.to_float(tf.reduce_min(eq_hist, axis = 1)), axis = 1))/(tf.expand_dims(tf.to_float(tf.reduce_max(eq_hist, axis = 1)), axis = 1)- tf.expand_dims(tf.to_float(tf.reduce_min(eq_hist, axis = 1)), axis = 1))
    # eq_hist = tf.reshape(eq_hist, shape=img_shape)
    return eq_hist


def image_preprocessing(image,req_z,req_y,req_x,n_vars,lorg_z,lorg_y,lorg_x):
    new_image = tf.reshape(image,shape=(-1,lorg_x,lorg_y * lorg_z,n_vars))
    # new_image = tf.image.adjust_contrast(new_image, contrast_factor = 2)
    new_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),new_image)
    new_image = tf.reshape(image,shape=(-1,lorg_x,lorg_y,lorg_z,n_vars))
    new_image = tf_crop_or_pad_along_axis(new_image,req_x,1)
    new_image = tf_crop_or_pad_along_axis(new_image,req_y,2)
    new_image = tf_crop_or_pad_along_axis(new_image,req_z,3)
    # new_image = tf.transpose(new_image, perm = [0,3,2,1,4]) #- needs to be transposed added transpose in read to match c++
    new_image = tf.reshape(new_image,shape=(-1,req_x,req_y,req_z,n_vars))  # was x, z
    return new_image
