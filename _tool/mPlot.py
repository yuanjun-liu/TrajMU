import math,sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import matplotlib.colors as mcolor
font_times = 'Times New Roman'
plt.rcParams['font.size'] = 8  
matplotlib.rcParams['mathtext.fontset'] = 'stix'
if sys.platform != 'linux':
    matplotlib.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = font_times
    matplotlib.rcParams['font.serif'] = font_times
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams["mathtext.fontset"] = "cm"

def set_figsize(width, height, dpi):
    plt.rcParams['figure.figsize'] = (width, height)
    plt.rcParams['savefig.dpi'] = dpi  
    plt.rcParams['figure.dpi'] = dpi  
bar_hatch=['/','//','///','\\\\','\\\\\\', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] 
