#!/usr/bin/env python2

import pandas as pd
import vtk

def read_vtp(input_file):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    return reader.GetOutput()

def read_vti(input_file):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    return reader.GetOutput()


def vtp_points_to_df(input_file):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    vtp = reader.GetOutput()
    points = []
    for i in range(vtp.GetNumberOfPoints()):
        points.append(dict(zip(['x','y','z'],vtp.GetPoint(i))))
    df = pd.DataFrame(points)
    df.sort_values(['x','y','z'],inplace=True)
    return df

def vti_to_df(input_file):
    vtk_image_data = read_vti(input_file)
    point_data = vtk_image_data.GetPointData()
    field_data = vtk_image_data.GetFieldData()

    data_dict = {}
    # Add field data
    for i in range(field_data.GetNumberOfArrays()):
        array_name = field_data.GetArrayName(i)
        array = field_data.GetArray(array_name)
        if array is None:
            array = field_data.GetAbstractArray(i)
        data_dict[array_name] = map(array.GetValue,xrange(array.GetNumberOfTuples()))

    # Add point data
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(array_name)
        if array is None:
            continue
        data_dict[array_name] = map(array.GetValue,xrange(array.GetNumberOfTuples()))

    num_rows = max(map(len,data_dict.itervalues()))
    data_dict = { k : v + [None] * (num_rows - len(v)) for (k,v) in data_dict.iteritems() }
    
    return pd.DataFrame(data_dict)

def vtp_to_df(input_file):
    poly_data = read_vtp(input_file)
    field_data = poly_data.GetFieldData()
    data_dict = {}
    # Add field data
    for i in range(field_data.GetNumberOfArrays()):
        array_name = field_data.GetArrayName(i)
        array = field_data.GetArray(array_name)
        if array is None:
            array = field_data.GetAbstractArray(i)
        values_list = map(array.GetValue,xrange(array.GetNumberOfTuples()))        
        data_dict[array_name] = values_list

    num_rows = max(map(len,data_dict.itervalues()))
    data_dict = { k : v + [None] * (num_rows - len(v)) for (k,v) in data_dict.iteritems() }
    return pd.DataFrame(data_dict)
    

def df_to_vtp(df,output_path):
    append = vtk.vtkAppendPolyData()
    
    
    for index, row in df.iterrows():
       
        cube = vtk.vtkCubeSource()
        xcom =  0.5*(row.xmax + row.xmin)
        ycom =  0.5*(row.ymax + row.ymin)
        zcom =  0.5*(row.zmax + row.zmin)
        xsize = row.xmax - row.xmin
        ysize = row.ymax - row.ymin
        zsize = row.zmax - row.zmin
        cube.SetCenter(xcom,ycom,zcom)
        cube.SetXLength(xsize)
        cube.SetYLength(ysize)
        cube.SetZLength(zsize)
        cube.Update()
       
        append.AddInput(cube.GetOutput())
    
    append.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInput(append.GetOutput())
    writer.SetFileName(output_path)
    writer.Write()


def df_to_vti(df,column,output_filename,resample_factor=1
             ,x_column='x'
             ,y_column='y'
             ,z_column='z'):
    ''' Create a vti file from a data frame based on a column/s
        :params:
            image: vtkImageData to write
            output_filename: path to output vti file to write
    '''

    # Get the unique x,y,z values 
    xs,ys,zs = map(lambda col: sorted(df[col].unique().tolist())
                        ,[x_column,y_column,z_column])

    # Get the number in each dimensioin
    nx,ny,nz = map(len,[xs,ys,zs])

    resample_df = get_resample_index_lists(resample_factor,xs,ys,zs)
    volume_size = xs[1] - xs[0]
    resample_volume_size = (xs[1] - xs[0]) * 1.0/resample_factor

    spacing = (resample_volume_size, resample_volume_size, resample_volume_size)
    origin = map(lambda i: i - volume_size/2.0, [xs[0], ys[0], zs[0]])
    dimensions = map(lambda n: resample_factor * n, [nx,ny,nz])
    
    image = create_empty_vtk_image_data(spacing,origin,dimensions)
    dims = image.GetDimensions()
    
    # Handle the case if only one column is passed in
    if isinstance(column,str):
        filter_columns = [column]
    else:
        filter_columns = column

    filter_image_data = {}
    for filter_name in filter_columns:
        filter_image_data[filter_name] = vtk.vtkDoubleArray()
        filter_image_data[filter_name].SetName(filter_name)
        filter_image_data[filter_name].SetNumberOfComponents(1)
        filter_image_data[filter_name].SetNumberOfTuples(image.GetNumberOfPoints())

    #first sort
    df_sorted = df.sort_values([x_column,y_column,z_column])
    df_sorted.dropna(subset=filter_columns,inplace=True)

    #convert to list
    filter_voxel_values_list = df_sorted[filter_columns].values.tolist()

    #for indexing in vti
    ixyz = 0
    #go through entire list
    for filter_voxel_values in filter_voxel_values_list:
        #print 'filter_voxel_values:',filter_voxel_values
        for iixyz in resample_df[resample_df.ixyz==ixyz]['iixyz_list'].values[0]:
            #they are in order
            for ifilter,filter_name in enumerate(filter_columns):
                filter_image_data[filter_name].SetValue(iixyz, filter_voxel_values[ifilter])
        ixyz += 1

    for filter_name in filter_columns:
        image.GetPointData().AddArray(filter_image_data[filter_name])

    # Write the image
    write_vti(image,output_filename)


def create_image(array,spacing,origin,dimensions,array_name):
    image = create_empty_vtk_image_data(spacing,origin,dimensions)
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName(array_name)
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetNumberOfTuples(image.GetNumberOfPoints())

    map(lambda i : vtk_array.SetValue(i,array[i]),range(len(array)))

    image.GetPointData().AddArray(vtk_array)
    return image

    

def get_resample_index_lists(resample_factor
                            ,xs,ys,zs):
    
    xs.sort()
    ys.sort()
    zs.sort()
    
    nxrs = resample_factor * len(xs)
    nyrs = resample_factor * len(ys)
    nzrs = resample_factor * len(zs)

    ixyz=0
    ixyz_iixyz_dict_list = []
    #for iz,z in enumerate(zs):
    for ix,x in enumerate(xs):
        for iy,y in enumerate(ys):
            #for ix,x in enumerate(xs):
            for iz,z in enumerate(zs):
                ixyz_iixyz_dict = {'ixyz':int(ixyz),'iixyz_list':[]}
                # resampling loops
                for iiz in xrange(iz*resample_factor,iz*resample_factor+resample_factor):
                    for iiy in xrange(iy*resample_factor,iy*resample_factor+resample_factor):
                        for iix in xrange(ix*resample_factor,ix*resample_factor+resample_factor):
                            iixyz = iix + iiy*nxrs + iiz*nxrs*nyrs
                            ixyz_iixyz_dict['iixyz_list'].append(iixyz)
                ixyz_iixyz_dict_list.append(ixyz_iixyz_dict)
                ixyz += 1
    return pd.DataFrame(ixyz_iixyz_dict_list) 

def create_empty_vtk_image_data(spacing,origin,dimensions):
    ''' Create a vti image data of doubles
        :params:
            spacing: tuple of spacing for x,y,z
            origin: tuple of x,y,z origin
            dimensions: tuple of number of voxels in nx,ny,nz
        :return:
            vtk image data
    '''
    image = vtk.vtkImageData()
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin) 
    image.SetDimensions(*dimensions)
    return image

def write_vti(image,output_filename):
    ''' Write vtk image data to output vti file
        :params:
            image: vtkImageData to write
            output_filename: path to output vti file to write
    '''
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(image.GetProducerPort())
    else:
        writer.SetInputData(image)
    writer.Write()


def extractVOI(image, b0, b1, b2, b3, b4, b5):
    min_voxel_ijk = [int()]*3
    max_voxel_ijk = [int()]*3
    pcoords = [float()]*3
 
    image.ComputeStructuredCoordinates([b0, b2, b4],min_voxel_ijk,pcoords)
    image.ComputeStructuredCoordinates([b1, b3, b5],max_voxel_ijk,pcoords)
 
    def ifZero(value):
        if value == 0:
            return value
        else:
            return value + 1
 
    start_voxel_ijk = map(ifZero, min_voxel_ijk)
    extents_ijk = list(reduce(lambda i,j : i + j, zip(start_voxel_ijk, max_voxel_ijk)))
 
    extract_voi = vtk.vtkExtractVOI()
    if vtk.vtkVersion.GetVTKMajorVersion() < 6:
        extract_voi.SetInput(image)
    else:
        extract_voi.SetInputData(image)
 
    extract_voi.SetVOI(*extents_ijk)
    extract_voi.Update()
 
    return extract_voi.GetOutput()

import sys

def does_intersect_np(row1,row2):
    ''' xmin, xmax, ymin, ymax, zmin, zmax
        0       1     2    3     4     5
    '''
    intersect_volume = reduce(lambda i,j : i*j \
                        , map(lambda (vmin,vmax) : max(min(*vmax) - max(*vmin) ,0)
                            , [  ((row2[0],row1[0]),(row2[1],row1[1]))
                                ,((row2[2],row1[2]),(row2[3],row1[3]))
                                ,((row2[4],row1[4]),(row2[5],row1[5])) ] ) )    
    return intersect_volume > 0

def percent_volume_of_intersection_np (gridrow,truthrow):
    ''' xmin, xmax, ymin, ymax, zmin, zmax
        0       1     2    3     4     5
    '''
    total_volume    = reduce(lambda i,j : i*j \
                        , map(lambda (vmin,vmax) : vmax - vmin
                            , [  (gridrow[0],gridrow[1])
                                ,(gridrow[2],gridrow[3])
                                ,(gridrow[4],gridrow[5]) ] ) )
    intersect_volume = reduce(lambda i,j : i*j \
                        , map(lambda (vmin,vmax) : max(min(*vmax) - max(*vmin) ,0)
                            , [  ((truthrow[0],gridrow[0]),(truthrow[1],gridrow[1]))
                                ,((truthrow[2],gridrow[2]),(truthrow[3],gridrow[3]))
                                ,((truthrow[4],gridrow[4]),(truthrow[5],gridrow[5])) ] ) )    
    return intersect_volume/total_volume

