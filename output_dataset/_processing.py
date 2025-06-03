import xarray as xr
import numpy as np
import sys 
from sklearn.decomposition import PCA

from typing import List, Tuple, Dict
from typing import Optional, Union,Callable


### TO DO
### Supported data proccesing
# Transpose
# Normalize
# Correlated product? (accept ints and strs?)
# Product? (accept ints and strs?)
# Smooth? -> access to savgol_filter

### Supported axis actions (accept ints and str?)
# shift axis
# centre axis
#

### Supported xarray-functions
# save datasets
# merge datasets
# 

def reverse_coords(coords):
    return list(reversed(list(coords)))
    
def transpose(data_output):
    """
        Transpose the data variables using xarrays
        build-in transpose funciton
        Args:
            datasets: list of xarray Datasets 
        Returns:
            list of xarray datasets: tranposed data
    """
    processed_data = []
    for dataset in data_output.datasets:
        processed_data.append(dataset.transpose())
    data_output.datasets = processed_data
     
def normalize(data_output,inverse=False):
    new_datasets = []
    for dataset in data_output.datasets:
        new_dataset = dataset
        for data_var in dataset.data_vars:
            if not inverse:
                new_dataset[data_var] -= np.min(dataset[data_var])
                new_dataset[data_var] /= np.max(dataset[data_var])
            else:
                new_dataset[data_var] -= np.max(dataset[data_var])
                new_dataset[data_var] *= -1                                
                new_dataset[data_var] /= np.max(dataset[data_var])

        new_datasets.append(new_dataset)
    data_output.datasets=new_datasets

def select(data_output, sel_dict):
    """
        Take a cut along a dimension using xarrays build-in sel function
        Args:
            datasets (list of Datasets): the data to which the sel operation is applied
            sel_dict: dictionary to pass to the .sel() function that specifies the coordinate and value
        Returns:
            list of xarray datasets with reduced dimensionality
    """
    method = 'nearest'
    for key in sel_dict:
        if isinstance(sel_dict[key], slice):
            method = None
    
    for idx,dataset in enumerate(data_output.datasets):
        data_output.datasets[idx] = dataset.sel(sel_dict, method = method)

def average_outerdim(datasets):
    for dataset in datasets:
        outer_coord = list(dataset.coords)[0]
        data_coord = list(dataset.data_vars)[0]
    
        output_values = np.zeros(dataset[data_coord].shape[1:])  
    
        for outer_val in dataset[outer_coord].values:
            subset = dataset.sel({f'{outer_coord}':outer_val}, method = 'nearest')
            output_values += subset[data_coord].values
        output_values = output_values/len(dataset[outer_coord].values)
        dataset[f'average_{data_coord}'] = xr.DataArray(output_values, coords = subset.coords, attrs = dataset[data_coord].attrs)
    return datasets


_supported_adjustments = {
    'centre': 'centre_axis',
    'shift': 'shift_axis',
    'scale': 'scale_axis',
    'multiply': 'scale',
}

def scale_axis(values, multiply_by = 1, shift_by = 0):
    return values * multiply_by + shift_by


def centre_axis(values:list):
    return values - (values[0]+values[-1])/2

def shift_axis(values:list, shift_by = 0):
    return values - shift_by

def _handle_get_default(values:list, **kwargs):
    """
        Default function for axis adjustment. It returns the values unchanged.
        Args:
            values (list): the values to be adjusted
        Returns:
            list: the unchanged values
    """
    raise(AttributeError('No adjustment function specified. Please specify a function or a string that is supported.'))

def adjust_axis(data_output, mapping:Callable|str, adjust:Optional[int|str] = 'all',**kwargs)->list[xr.Dataset]:
    """
        Map one or more axis of datasets to new values
        Args:
            datasets (list of Datasets): the data to which the sel operation is applied
            mapping (Callable): function to which to pass the coordinate that should be adjused
            adjust (int|str): optional, the axis to pass to mapping. By default
                all axis are processed. Can be either the axis index or the axis name.
        Returns:
            list of xarray datasets with new axis
    """
    
    if isinstance(mapping,str):
        adjust_func_name = _supported_adjustments.get(mapping, '_handle_get_default')
        adjust_func = getattr(sys.modules[__name__],adjust_func_name)
        mapping = adjust_func

    for idx,dataset in enumerate(data_output.datasets):
        if adjust == 'all':
            data_coords = reverse_coords(dataset.coords)
        elif isinstance(adjust, int):
            data_coords = [reverse_coords(dataset.coords)[adjust]]
        else:
            data_coords = [dataset.coords[adjust].name]
        
        new_coord_dict = {}
        for coord_key in reversed(data_coords):
            old_coord_da= dataset[coord_key]
            old_coord_attrs = dataset[coord_key].attrs
            coord_values = old_coord_da.values
            new_values = mapping(coord_values,**kwargs)
            new_coord_dict[coord_key] = (f'{coord_key}',new_values,old_coord_attrs)

        old_attrs = dataset.attrs
        new_dataset = dataset.assign_coords(new_coord_dict)
        new_dataset.attrs = old_attrs
        data_output.datasets[idx] = new_dataset

def rotate_PCA(data_output,data_key,shift_median=False):
    """Performs PCA on the data variable in the datasets and rotates the data accordingly.

    Args:
        data_output (DataOutput): DataOutput object to process.
        data_key (str): string key to the data variable in the datasets.
        shift_median (bool, optional): Shifts the projected data so that the
            median of each component is zero. Defaults to False.
    """
    for dataset in data_output.datasets:
        data_array = dataset[data_key]
        x_data = np.real(data_array.values)
        y_data = np.imag(data_array.values)
        X = x_data.flatten()
        Y = y_data.flatten()
        shape = np.shape(x_data)
        stacked_XY = np.column_stack((X, Y))
        # perform PCA
        pca = PCA(n_components=2, copy=True, svd_solver='full').fit(stacked_XY)
        # Do the projection manually
        # fix the found component to always have positive x-component.
        pca_component = [
            np.sign(pca.components_[0][0])*pca.components_[0][0],
            np.sign(pca.components_[0][0])*pca.components_[0][1],
            ]
        pca_component_perp = [
            -pca_component[1],
            pca_component[0]
            ]

        # project the coordinates onto the PCA components
        X_rot = np.dot(stacked_XY, pca_component)
        Y_rot = np.dot(stacked_XY, pca_component_perp)
        if shift_median:
            # Shift the projected data so that the median of each component is zero.
            X_rot = X_rot - np.median(X_rot)
            Y_rot = Y_rot - np.median(Y_rot)

        X_rot = np.reshape(X_rot,shape)
        Y_rot = np.reshape(Y_rot,shape)
        dataset[f"{data_key}_PCA_X"] = xr.DataArray(X_rot, coords = data_array.coords,attrs=data_array.attrs)
        dataset[f"{data_key}_PCA_Y"] = xr.DataArray(Y_rot, coords = data_array.coords,attrs=data_array.attrs)
        dataset[f"{data_key}_PCA_X"].attrs["long_name"] = f"{data_key}_PCA_X"
        dataset[f"{data_key}_PCA_Y"].attrs["long_name"] = f"{data_key}_PCA_Y"

def adjust_coordinate_offset(dataset, coord_key, offset):
    new_coord_dict = {}
    old_coord_da= dataset[coord_key]
    old_coord_attrs = dataset[coord_key].attrs
    coord_values = old_coord_da.values
    new_values = coord_values - offset
    new_coord_dict[coord_key] = (f'{coord_key}',new_values,old_coord_attrs)
    old_attrs = dataset.attrs
    new_dataset = dataset.assign_coords(new_coord_dict)
    new_dataset.attrs = old_attrs
    return new_dataset

def adjust_data_offset(dataset,data_key,offset):
    data_array = dataset[data_key]
    old_attrs = dataset[data_key].attrs
    data_array = data_array - offset
    dataset[data_key] = data_array
    dataset[data_key].attrs = old_attrs
    return dataset

def multiply(data_output: 'DataOutput' , multiplier: int):

    ## Loop over all datasets
    for idx,dataset in enumerate(data_output.datasets):
        new_dataset = dataset*multiplier

        ## Override the old dataset with the new dataset
        data_output.datasets[idx] = new_dataset
