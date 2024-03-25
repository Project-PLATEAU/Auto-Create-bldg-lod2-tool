from .plobj import plobj
from .plutils import *
from .ploptions import ploptions
import numpy as np
import copy
from lxml import etree

class plluse(plobj):
	def __init__(self,filename=None, options=ploptions()):
		super().__init__()
		self.kindstr = 'luse'
		if filename is not None:
			self.loadFile(filename, options=options)
	def loadFile(self,filename, options=ploptions()):
		pass
