import pandas as pd
import numpy as np
import random
import tensorflow as tf

import multiprocessing
from vtk.util.numpy_support import vtk_to_numpy


import argparse
#import vtk_util
from itertools import product

# ORIG
ORIG_IMAGE_X=47
ORIG_IMAGE_Y=117
ORIG_IMAGE_Z=52

# CROP OR PAD
CROP_PAD_IMAGE_X=48
#CROP_PAD_IMAGE_Y=121
CROP_PAD_IMAGE_Y=119
CROP_PAD_IMAGE_Z=48

# FOLD BLOCK
## FOLD SIZE
#NUM_FOLDS_X=1
#NUM_FOLDS_Y=4
#NUM_FOLDS_Z=2

## FOLD OVERLAP
NUM_VOXEL_OVERLAP_X=0
NUM_VOXEL_OVERLAP_Y=3
NUM_VOXEL_OVERLAP_Z=0

# AXIS ORDING
DIMENSION_X=2
DIMENSION_Y=1
DIMENSION_Z=0

CONTAINER_XMIN=257
CONTAINER_XMAX=487.5
CONTAINER_YMIN=590
CONTAINER_YMAX=1175.5
CONTAINER_ZMIN=292.5
CONTAINER_ZMAX=552.5

GRAPH_INPUT_SHAPE = ( 24, 32, 48 ) 


def find_number_of_patches(patch_length,patch_overlap,total_length):
    num_folds = 0
    while (patch_length - patch_overlap) * num_folds + patch_length <= total_length:
        num_folds += 1
    return num_folds

def get_index_pairs_for_patches(patch_length,patch_overlap,num_patches):
    return map(lambda i: ((patch_length - patch_overlap) * i
                         ,(patch_length - patch_overlap) * i + patch_length)
        , range(num_patches))

def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=1)
    return b

def crop_or_pad_along_axis(array,target_length,axis):

    if array.shape[axis] < target_length:
        array = pad_along_axis(array,target_length,axis=axis)
    elif array.shape[axis] > target_length:
        number_to_crop = array.shape[axis] - target_length
        crop_indexes = range(array.shape[axis] - number_to_crop,array.shape[axis])
        array = np.delete(array,crop_indexes,axis=axis)
    return array

def preprocess_image(vtk_image):
    nx,ny,nz = vtk_image.GetDimensions()
    
    # Extract image from vtk to numpy array.
    # This is 1-D with (z,y,x) ordering
    image = vtk_to_numpy(vtk_image.GetPointData().GetScalars()).astype(np.float32)
    
    # Normalize the data. 
    image = (image - image.min())/(image.max()- image.min())
    image = image - image.mean()

    # Reshape. The dimension ordering is (z,y,x)
    image = image.reshape(nz,ny,nx)
    
    # Transpose. 
    # Use the explicit form. For an n-D array, if axes are given, their 
    # order indicates how the axes are permuted. For example
    # if DIMENSION_Z,DIMENSION_Y,DIMENSION_X = (0,1,2) the transpose will not permute the axis (no transpose)
    # if DIMENSION_Z,DIMENSION_Y,DIMENSION_X = (2,1,0) the transpose will do (z,y,x) -> (x,y,z)
    image = image.transpose(DIMENSION_Z,DIMENSION_Y,DIMENSION_X)

    # Crop or pad the volume so the deconvolutions work
    image = crop_or_pad_along_axis(image,CROP_PAD_IMAGE_X,DIMENSION_X)
    image = crop_or_pad_along_axis(image,CROP_PAD_IMAGE_Y,DIMENSION_Y)
    image = crop_or_pad_along_axis(image,CROP_PAD_IMAGE_Z,DIMENSION_Z)

    return image

def create_patches(image):
    # Get all the patch slices
    patch_slices = create_patch_slices()

    # Extract patches
    patch_images = np.array(map(lambda patch_slice: image[patch_slice], patch_slices))

    # Pop out a dimension for the channel
    patch_images = np.expand_dims(patch_images,axis=-1)

    return patch_images

def create_patch_slices():
    
    # Calculate number of patches for a given length and overlap
    NUM_PATCHES_X = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_X] , NUM_VOXEL_OVERLAP_X, CROP_PAD_IMAGE_X)
    NUM_PATCHES_Y = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_Y] , NUM_VOXEL_OVERLAP_Y, CROP_PAD_IMAGE_Y)
    NUM_PATCHES_Z = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_Z] , NUM_VOXEL_OVERLAP_Z, CROP_PAD_IMAGE_Z)

    # Get the start and stop index pairs for the patches
    patch_index_pairs_x = get_index_pairs_for_patches(GRAPH_INPUT_SHAPE[DIMENSION_X] , NUM_VOXEL_OVERLAP_X, NUM_PATCHES_X)
    patch_index_pairs_y = get_index_pairs_for_patches(GRAPH_INPUT_SHAPE[DIMENSION_Y] , NUM_VOXEL_OVERLAP_Y, NUM_PATCHES_Y)
    patch_index_pairs_z = get_index_pairs_for_patches(GRAPH_INPUT_SHAPE[DIMENSION_Z] , NUM_VOXEL_OVERLAP_Z, NUM_PATCHES_Z)

    # Convert start/stop index pairs to slice objects to cut image later
    patch_slices_x = map(lambda i: slice(*i),patch_index_pairs_x)
    patch_slices_y = map(lambda i: slice(*i),patch_index_pairs_y)
    patch_slices_z = map(lambda i: slice(*i),patch_index_pairs_z)

    # This will create an ordered tuple that will have the correct
    # patch slice ordering relative to the dimension ordering
    patch_slices_correct_dimension_ordering,_ = zip(*sorted(zip(
                        (patch_slices_x,patch_slices_y, patch_slices_z)
                       ,(DIMENSION_X,   DIMENSION_Y,    DIMENSION_Z   )
                    ),key=lambda (slices,dim): dim))

    # Calculate the product of image indexes for all dimensions.
    patch_slices = product(*patch_slices_correct_dimension_ordering)

    return patch_slices

def stitch_patches(patch_images,overlap_op=np.maximum):

    # Get all the patch slices
    patch_slices = list(create_patch_slices())

    patch_images = patch_images.squeeze() # Reverse expand dimensions
    
    # Get image size
    image_size = map(lambda axis_slices: max(axis_slices,key= lambda s: s.stop).stop, zip(*patch_slices))
    
    # Start with an image of zeros
    image = np.zeros(image_size)
    
    # Loop through once to fill in the patch
    for patch_index,patch_slice in enumerate(patch_slices):
        image[patch_slice] = patch_images[patch_index]

    # Loop through again and apply a operation
    # This is for the overlapping regions
    # For example:
    #     np.maximum
    #     np.logical_or
    #     np.logical_and
    for patch_index,patch_slice in enumerate(patch_slices):
        image[patch_slice] = overlap_op(image[patch_slice],patch_images[patch_index])
        
    return image

def get_random_pair_for_patch(patch_length,container_dim):
    if patch_length<container_dim:
        start_dim = random.randint(0, (container_dim - patch_length))
        return [(start_dim, start_dim+patch_length)]
    else:
        return [(0, container_dim)]
    
def create_random_patch_slice():
    # Calculate number of patches for a given length and overlap
    NUM_PATCHES_X = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_X] , NUM_VOXEL_OVERLAP_X, CROP_PAD_IMAGE_X)
    NUM_PATCHES_Y = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_Y] , NUM_VOXEL_OVERLAP_Y, CROP_PAD_IMAGE_Y)
    NUM_PATCHES_Z = find_number_of_patches(GRAPH_INPUT_SHAPE[DIMENSION_Z] , NUM_VOXEL_OVERLAP_Z, CROP_PAD_IMAGE_Z)
    # Get the start and stop index pairs for the patches
    patch_index_pairs_x = get_random_pair_for_patch(GRAPH_INPUT_SHAPE[DIMENSION_X], CROP_PAD_IMAGE_X)
    patch_index_pairs_y = get_random_pair_for_patch(GRAPH_INPUT_SHAPE[DIMENSION_Y], CROP_PAD_IMAGE_Y)
    patch_index_pairs_z = get_random_pair_for_patch(GRAPH_INPUT_SHAPE[DIMENSION_Z], CROP_PAD_IMAGE_Z)
    # Convert start/stop index pairs to slice objects to cut image later
    patch_slices_x = map(lambda i: slice(*i),patch_index_pairs_x)
    patch_slices_y = map(lambda i: slice(*i),patch_index_pairs_y)
    patch_slices_z = map(lambda i: slice(*i),patch_index_pairs_z)
    # This will create an ordered tuple that will have the correct
    # patch slice ordering relative to the dimension ordering
    patch_slices_correct_dimension_ordering,_ = zip(*sorted(zip(
                        (patch_slices_x,patch_slices_y, patch_slices_z)
                       ,(DIMENSION_X,   DIMENSION_Y,    DIMENSION_Z   )
                    ),key=lambda (slices,dim): dim))
    # Calculate the product of image indexes for all dimensions.
    patch_slices = product(*patch_slices_correct_dimension_ordering)
    return patch_slices    

def create_random_patches(image, random_slice):
    # Get all the patch slices
    # Extract patches
    patch_images = np.array(map(lambda random_slice: image[random_slice], random_slice))

    # Pop out a dimension for the channel
    patch_images = np.expand_dims(patch_images,axis=-1)

    return patch_images
