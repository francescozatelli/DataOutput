import json
import os
from typing import List, Tuple, Dict
from typing import Optional, Union,Callable
from importlib import reload

from IPython.core.display import Markdown
from IPython.display import display

import ipywidgets as widgets

def get_file_path(filename:str)->str:
	""" 
		Retrieve file path relative to package location, with correct os format
		Ensures parameters and configs are saved in correct location

		Args:
			filename (string): the name of the file whose path to obtain
		Returns:
			String containing the path to the file
	"""
	module_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(module_dir, filename)

def load_dictionary(filename:str)->dict:
	"""
		Reads the data from json files, 
		or create file with default settings

		Args:
			filename (string): name of the file to load, including extension
		Returns:
			dictionary: content of the file if it existed, default values otherwise
	"""
	file_path = get_file_path(filename)
	if os.path.exists(file_path):
		with open(file_path, 'r') as f:
			return json.load(f)
	else:
		with open(file_path, 'w') as f:
			default = {}
			json.dump(default, f)
			return default
			
def save_dictionary(dictionary:dict, filename:str)->None:
	"""
		Save a dictionary to a json file
		Args:
			dictionary: the dict with values to save
			file_path: str containing path to json file to store data
	"""
	file_path = get_file_path(filename)
	with open(file_path, 'w') as f:
		json.dump(dictionary, f,indent=4)

class ParameterMeta(type):
    """Metaclass to allow Parameter['name'] lookup at the class level."""
    def __getitem__(cls, name):
        return cls._instances[name]  # Raises KeyError if not found

class Parameter(metaclass=ParameterMeta):
	_instances = {}  # Store instances by name
	filename_params = "verbose_params.json" ## Hardcoded parameter filename

	def __init__(self, param_name, verbose_name = None, unit = '-', scale = 1, offset = 0):
		if param_name in Parameter._instances:
			raise ValueError(f"A Parameter with name '{param_name}' already exists.")
		self.param_name = str(param_name)
		self.verbose_name = str(verbose_name) if verbose_name is not None else str(param_name)
		self.unit = str(unit) if unit is not None else '-'
		self.scale = scale if scale is not None else 1
		self.offset = offset if offset is not None else 0
		Parameter._instances[param_name] = self  # Store the instance
		#Parameter.save() # Save to the dictionary

	def __repr__(self):
		return f"Parameter(name={self.verbose_name})"
	
	def _repr_markdown_(self):
		return self.as_label() + f' - Scale: {self.scale:.2e}, Offset: {self.offset:.2e}'
	
	def as_label(self, with_unit: bool=True)->str:
		"""
			Construct an axis label for a parameter
			by combining its verbose name and its unit

			Args:
				param_key (string): the dictionary key value of the parmater to load
				add_unit (bool)
		"""
		if with_unit:
			return  f"{self.verbose_name} ({self.unit})"
		else:
			return f"{self.verbose_name}"
		
	def update(self,**kwargs):
		if not kwargs:
			self.update_widget()
		else:
			for key,item in kwargs.items():
				if hasattr(self,key):
					setattr(self,key,item)
				else:
					raise KeyError(f"{self.__class__.__name__} has not attribute {key}")
			Parameter.save() ## Dump all data into the parameter file

	def update_widget(self):
		# Define four interactive widgets
		name = widgets.Text(description="Label", value=f"{self.verbose_name}")
		unit = widgets.Text(description="Unit", value=f"{self.unit}")
		scale = widgets.Text(description="Scale", value=f"{self.scale}")
		offset = widgets.Text(description="Offset", value=f"{self.offset}")

		# Submit button
		button = widgets.Button(description="Save")

		# Function to handle submission
		def on_submit(b):
			self.verbose_name = name.value
			self.unit = unit.value
			self.scale = float(scale.value)
			self.offset = float(offset.value)
			Parameter.save()

		button.on_click(on_submit)
		# Display widgets
		display(name, unit, scale, offset, button)	

	@classmethod
	def reload(cls):
		cls._instances = {}
		verbose_param_dict = load_dictionary(cls.filename_params)
		for key,item in verbose_param_dict.items():
			Parameter(key, verbose_name = item.get('verbose_name'),unit=item.get('unit'), scale=item.get('scale'), offset=item.get('offset'))				

	@classmethod
	def load(cls):
		verbose_param_dict = load_dictionary(cls.filename_params)
		for key,item in verbose_param_dict.items():
			Parameter(key, verbose_name = item.get('verbose_name'),unit=item.get('unit'), scale=item.get('scale'), offset=item.get('offset'))

	@classmethod
	def save(cls):
		params_as_dict = {}
		for param_name,parameter in cls._instances.items():
			params_as_dict[parameter.param_name] = {}
			for attr in ['verbose_name','unit','scale','offset']:
				value = getattr(parameter, attr)
				params_as_dict[parameter.param_name][attr]=value
		save_dictionary(params_as_dict,cls.filename_params)

	@classmethod
	def all(cls):
		for param_key,parameter in cls._instances.items():
			display(Markdown(cls._repr_markdown_(parameter)))

Parameter.load()
