# encoding:utf-8

# csdaiwei@foxmail.com

import pdb

import pyocr
import pyocr.builders

from PIL import Image


# load ocr tools
tool = pyocr.get_available_tools()[0]		#tesseract
lang = tool.get_available_languages()[0]	#eng
assert lang == 'eng'


#load, crop and enhance target image
img = Image.open('spec.1231.t0.s0.jpg')
region = (20, 55, 300, 70)
cropimg = img.crop(region)
cropimg = cropimg.resize((cropimg.size[0]*5, cropimg.size[1]*5), Image.BILINEAR)	#important

# recognize
txt = tool.image_to_string(cropimg, lang=lang, builder=pyocr.builders.TextBuilder())
print txt