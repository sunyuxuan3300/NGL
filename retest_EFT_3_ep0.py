"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation (slightly modified).
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import numpy as np
import gvar as gv 
import sys
import time  # 添加时间模块导入
from multiprocessing import Pool  # 添加Pool的导入

from numpy import log
from numpy import abs
from numpy import sin
from numpy import cos
from numpy import pi
from numba import jit
import datetime
import os
import platform
import psutil
import socket

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True

# 全局参数
r = 2
part = 'real'
eulerGamma = 0.5772156649

# 参数配置
class IntegrationConfig:
    """积分计算的配置类"""
    def __init__(self):
        
        # 类参数
        self.r = 2
        self.part = 'real'
        self.eulerGamma = 0.5772156649
        
        # 积分区间
        self.integration_ranges = [
            [0., 2*np.pi],  # x0
            [0., 1.],       # x1
            [0., 1.],       # x2
            [0., 1.],       # x3
            [0., 1.],       # x4
            [0., 1.],       # x5
            [0., 1.],       # x6
            [0., 1.]        # x7
        ]
        
        # 训练阶段参数
        self.training_iterations = 5    # nitn for training
        self.training_evaluations = 2e7  # neval for training
        
        # 计算阶段参数
        self.final_iterations = 30      # nitn for final integration
        self.final_evaluations = 2e7     # neval for final integration
        
        # 并行计算参数
        self.n_cores = 100               # 并行核心数
        self.n_iterations = 10           # 重复计算次数
        
        # 输出控制
        self.save_intermediate = True    # 是否保存中间结果
        self.save_training = False       # 是否保存训练结果
        self.verbose = True             # 是否显示详细输出
        
        # 文件命名格式
        self.result_dir_format = 'results_{part}_r-{r}_{timestamp}'
        self.result_file_format = 'final_result_{part}_r-{r}_cores-{cores}_iters-{iters}'
        
        # 添加x0分布计算的配置
        self.x0_distribution = {
            'n_points': 100,           # x0采样点数
            'x0_range': [0, 2*np.pi],  # x0的范围
            'plot_size': (10, 6),      # 图像大小
            'save_intermediate': True,  # 是否保存中间结果
            'plot_style': {
                'fmt': 'o',            # 数据点样式
                'capsize': 5,          # 误差棒大小
                'markersize': 3,       # 标记大小
                'elinewidth': 1,       # 误差线宽度
                'grid': True           # 是否显示网格
            },
            # 积分参数
            'training_iterations': 10,  # 训练迭代次数
            'training_evaluations': 1e7,# 训练采样点数
            'final_iterations': 30,     # 最终迭代次数
            'final_evaluations': 2e7    # 最终采样点数
        }
        
        # 添加图像配置
        self.plot_config = {
            'dpi': 300,                # 图像分辨率
            'format': 'png',           # 保存格式
            'xlabel': 'x[0]',          # x轴标签
            'ylabel': 'Integral Value',# y轴标签
            'title': 'Integration Result vs x[0]' # 图像标题
        }
        
    def get_timestamp(self):
        """获取当前时间戳"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_result_dir(self):
        """获取结果目录名"""
        return self.result_dir_format.format(
            part=self.part,
            r=format(self.r, '.1e'),
            timestamp=self.get_timestamp()
        )
    
    def get_result_filename(self, cores, iters):
        """获取结果文件名"""
        return self.result_file_format.format(
            part=self.part,
            r=format(self.r, '.1e'),
            cores=cores,
            iters=iters
        )
    
    def to_dict(self):
        """将配置转换为字典形式"""
        return {
            'r': self.r,
            'part': self.part,
            'eulerGamma': self.eulerGamma,
            'integration_ranges': self.integration_ranges,
            'training_iterations': self.training_iterations,
            'training_evaluations': self.training_evaluations,
            'final_iterations': self.final_iterations,
            'final_evaluations': self.final_evaluations,
            'n_cores': self.n_cores,
            'n_iterations': self.n_iterations,
            'save_intermediate': self.save_intermediate,
            'save_training': self.save_training,
            'verbose': self.verbose
        }


@jit(nopython=True)
def f(x):
    x0, x1, x2, x3, x4, x5, x6, x7 = x
    
    result = 128*np.pi**4*(-(((-1+2*x1)*(2*x3*x4+4*np.pi**2*x3*x4-2*np.pi**2*x3**2*x4-2*np.\
pi**2*x3*x4**2+np.pi**2*x3**2*x4**2-8*x3*np.log(2)-8*x4*np.log(2)+16*\
x3*x4*np.log(2)-4*x3**2*x4*np.log(2)-4*x3*x4**2*np.log(2)+16*x3*x4*np.\
log(2)**2-8*x3**2*x4*np.log(2)**2-8*x3*x4**2*np.log(2)**2+4*x3**2*\
x4**2*np.log(2)**2+x3**2*np.log(16)+x4**2*np.log(16)+8*x3*x4*np.log(1-\
x1)-2*x3**2*x4*np.log(1-x1)-2*x3*x4**2*np.log(1-x1)+16*x3*x4*np.log(2)\
*np.log(1-x1)-8*x3**2*x4*np.log(2)*np.log(1-x1)-8*x3*x4**2*np.log(2)*\
np.log(1-x1)+x3**2*x4**2*np.log(16)*np.log(1-x1)+4*x3*x4*np.log(1-x1)\
**2-2*x3**2*x4*np.log(1-x1)**2-2*x3*x4**2*np.log(1-x1)**2+x3**2*x4**2*\
np.log(1-x1)**2+8*x3*x4*np.log(x1)-2*x3**2*x4*np.log(x1)-2*x3*x4**2*\
np.log(x1)+16*x3*x4*np.log(2)*np.log(x1)-8*x3**2*x4*np.log(2)*np.log(\
x1)-8*x3*x4**2*np.log(2)*np.log(x1)+x3**2*x4**2*np.log(16)*np.log(x1)+\
8*x3*x4*np.log(1-x1)*np.log(x1)-4*x3**2*x4*np.log(1-x1)*np.log(x1)-4*\
x3*x4**2*np.log(1-x1)*np.log(x1)+2*x3**2*x4**2*np.log(1-x1)*np.log(x1)\
+4*x3*x4*np.log(x1)**2-2*x3**2*x4*np.log(x1)**2-2*x3*x4**2*np.log(x1)\
**2+x3**2*x4**2*np.log(x1)**2+8*x3*x4*np.log(1-x2)-2*x3**2*x4*np.log(\
1-x2)-2*x3*x4**2*np.log(1-x2)+16*x3*x4*np.log(2)*np.log(1-x2)-8*x3**2*\
x4*np.log(2)*np.log(1-x2)-8*x3*x4**2*np.log(2)*np.log(1-x2)+x3**2*\
x4**2*np.log(16)*np.log(1-x2)+8*x3*x4*np.log(1-x1)*np.log(1-x2)-4*\
x3**2*x4*np.log(1-x1)*np.log(1-x2)-4*x3*x4**2*np.log(1-x1)*np.log(1-\
x2)+2*x3**2*x4**2*np.log(1-x1)*np.log(1-x2)+8*x3*x4*np.log(x1)*np.log(\
1-x2)-4*x3**2*x4*np.log(x1)*np.log(1-x2)-4*x3*x4**2*np.log(x1)*np.log(\
1-x2)+2*x3**2*x4**2*np.log(x1)*np.log(1-x2)+4*x3*x4*np.log(1-x2)**2-2*\
x3**2*x4*np.log(1-x2)**2-2*x3*x4**2*np.log(1-x2)**2+x3**2*x4**2*np.\
log(1-x2)**2+8*x3*x4*np.log(x2)-2*x3**2*x4*np.log(x2)-2*x3*x4**2*np.\
log(x2)+16*x3*x4*np.log(2)*np.log(x2)-8*x3**2*x4*np.log(2)*np.log(x2)-\
8*x3*x4**2*np.log(2)*np.log(x2)+x3**2*x4**2*np.log(16)*np.log(x2)+8*\
x3*x4*np.log(1-x1)*np.log(x2)-4*x3**2*x4*np.log(1-x1)*np.log(x2)-4*x3*\
x4**2*np.log(1-x1)*np.log(x2)+2*x3**2*x4**2*np.log(1-x1)*np.log(x2)+8*\
x3*x4*np.log(x1)*np.log(x2)-4*x3**2*x4*np.log(x1)*np.log(x2)-4*x3*\
x4**2*np.log(x1)*np.log(x2)+2*x3**2*x4**2*np.log(x1)*np.log(x2)+8*x3*\
x4*np.log(1-x2)*np.log(x2)-4*x3**2*x4*np.log(1-x2)*np.log(x2)-4*x3*\
x4**2*np.log(1-x2)*np.log(x2)+2*x3**2*x4**2*np.log(1-x2)*np.log(x2)+4*\
x3*x4*np.log(x2)**2-2*x3**2*x4*np.log(x2)**2-2*x3*x4**2*np.log(x2)**2+\
x3**2*x4**2*np.log(x2)**2-4*(-2+x4)*x4*np.log(2-x3)-2*x3*(-2+x4)*x4*\
np.log(x3)+8*x3*np.log(2-x4)-4*x3**2*np.log(2-x4)+4*x3*x4*np.log(x4)-\
2*x3**2*x4*np.log(x4)-16*x3*x4*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*\
np.sign(1-2*x2))+4*x3**2*x4*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.\
sign(1-2*x2))+4*x3*x4**2*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(\
1-2*x2))-32*x3*x4*np.log(2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.\
sign(1-2*x2))+16*x3**2*x4*np.log(2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.\
pi*np.sign(1-2*x2))+16*x3*x4**2*np.log(2)*(np.log(np.abs(1-2*x2))+(1j/\
2)*np.pi*np.sign(1-2*x2))-8*x3**2*x4**2*np.log(2)*(np.log(np.abs(1-2*\
x2))+(1j/2)*np.pi*np.sign(1-2*x2))-16*x3*x4*np.log(1-x1)*(np.log(np.\
abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))+8*x3**2*x4*np.log(1-x1)*(\
np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))+8*x3*x4**2*np.\
log(1-x1)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))-4*\
x3**2*x4**2*np.log(1-x1)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(\
1-2*x2))-16*x3*x4*np.log(x1)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.\
sign(1-2*x2))+8*x3**2*x4*np.log(x1)*(np.log(np.abs(1-2*x2))+(1j/2)*np.\
pi*np.sign(1-2*x2))+8*x3*x4**2*np.log(x1)*(np.log(np.abs(1-2*x2))+(1j/\
2)*np.pi*np.sign(1-2*x2))-4*x3**2*x4**2*np.log(x1)*(np.log(np.abs(1-2*\
x2))+(1j/2)*np.pi*np.sign(1-2*x2))-16*x3*x4*np.log(1-x2)*(np.log(np.\
abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))+8*x3**2*x4*np.log(1-x2)*(\
np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))+8*x3*x4**2*np.\
log(1-x2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))-4*\
x3**2*x4**2*np.log(1-x2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(\
1-2*x2))-16*x3*x4*np.log(x2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.\
sign(1-2*x2))+8*x3**2*x4*np.log(x2)*(np.log(np.abs(1-2*x2))+(1j/2)*np.\
pi*np.sign(1-2*x2))+8*x3*x4**2*np.log(x2)*(np.log(np.abs(1-2*x2))+(1j/\
2)*np.pi*np.sign(1-2*x2))-4*x3**2*x4**2*np.log(x2)*(np.log(np.abs(1-2*\
x2))+(1j/2)*np.pi*np.sign(1-2*x2))+16*x3*x4*(np.log(np.abs(1-2*x2))+(\
1j/2)*np.pi*np.sign(1-2*x2))**2-8*x3**2*x4*(np.log(np.abs(1-2*x2))+(\
1j/2)*np.pi*np.sign(1-2*x2))**2-8*x3*x4**2*(np.log(np.abs(1-2*x2))+(\
1j/2)*np.pi*np.sign(1-2*x2))**2+4*x3**2*x4**2*(np.log(np.abs(1-2*x2))+\
(1j/2)*np.pi*np.sign(1-2*x2))**2+8*x3*(np.log(np.abs((1-2*x1)*np.cos(\
x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*\
np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-4*x3**2*(np.log(np.abs((\
1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4*(\
np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(\
1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+2*x3**2*x4*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*\
x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))+2*x3*x4**2*(np.log(np.abs((1-2*x1)*np.cos(x0)-\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.\
cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-16*x3*x4*np.log(2)*(np.log(\
np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*\
x3**2*x4**2*np.log(4)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-\
x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+2*x3**2*x4*np.log(16)*(np.log(np.abs((1-\
2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+2*x3*x4**2*\
np.log(16)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.\
sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*\
x1)*np.sin(x0)))-8*x3*x4*np.log(1-x1)*(np.log(np.abs((1-2*x1)*np.cos(\
x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*\
np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.log(1-x1)*(\
np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(\
1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+4*x3*x4**2*np.log(1-x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*x4**2*np.log(1-x1)*(np.log(\
np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*\
x3*x4*np.log(x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*\
x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.log(x1)*(np.log(np.abs((1-2*x1)*\
np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-\
2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3*x4**2*np.log(\
x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)\
))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.\
sin(x0)))-2*x3**2*x4**2*np.log(x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)-\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.\
cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4*np.log(1-x2)*(np.\
log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/\
2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))\
+4*x3**2*x4*np.log(1-x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt(\
(1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+4*x3*x4**2*np.log(1-x2)*(np.log(np.abs((\
1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*\
x4**2*np.log(1-x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)\
*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))-8*x3*x4*np.log(x2)*(np.log(np.abs((1-2*x1)*np.\
cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*\
x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.log(x2)\
*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+\
(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+4*x3*x4**2*np.log(x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*x4**2*np.log(x2)*(np.log(np.\
abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.\
pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+16*\
x3*x4*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.log(\
np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*\
x3**2*x4*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.\
log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/\
2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))\
-8*x3*x4**2*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.\
log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/\
2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))\
+4*x3**2*x4**2*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(\
np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(\
1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+8*x3*x4*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)\
*x1)*np.sin(x0)))**2-4*x3**2*x4*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*\
np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(\
x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))**2-4*x3*x4**2*(np.log(np.abs((1-\
2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(x0)))**2+2*x3**2*\
x4**2*(np.log(np.abs((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)-2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))**2+8*x3*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-\
x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))-4*x3**2*(np.log(np.abs((1-2*x1)*np.cos(\
x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*\
np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4*(np.log(np.abs((\
1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+2*x3**2*x4*\
(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(\
1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+2*x3*x4**2*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*\
x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))-16*x3*x4*np.log(2)*(np.log(np.abs((1-2*x1)*np.\
cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*\
x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*x4**2*np.log(\
4)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0))\
)+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.\
sin(x0)))+2*x3**2*x4*np.log(16)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*\
np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(\
x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+2*x3*x4**2*np.log(16)*(np.log(\
np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*\
x3*x4*np.log(1-x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)\
*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.log(1-x1)*(np.log(np.abs((1-2*x1)\
*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-\
2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3*x4**2*np.log(\
1-x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))-2*x3**2*x4**2*np.log(1-x1)*(np.log(np.abs((1-2*x1)*np.\
cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*\
x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4*np.log(x1)*(\
np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(\
1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+4*x3**2*x4*np.log(x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3*x4**2*np.log(x1)*(np.log(np.\
abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.\
pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*\
x3**2*x4**2*np.log(x1)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((\
1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4*np.log(1-x2)*(np.log(np.abs((1-2*\
x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign(\
(1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.\
log(1-x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.\
sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*\
x1)*np.sin(x0)))+4*x3*x4**2*np.log(1-x2)*(np.log(np.abs((1-2*x1)*np.\
cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*\
x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*x4**2*np.log(\
1-x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))-8*x3*x4*np.log(x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*\
np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(\
x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3**2*x4*np.log(x2)*(np.log(\
np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*\
x3*x4**2*np.log(x2)*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-\
x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))-2*x3**2*x4**2*np.log(x2)*(np.log(np.abs((\
1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+16*x3*x4*(\
np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.log(np.abs((\
1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3**2*x4*\
(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.log(np.abs((\
1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))-8*x3*x4**2*\
(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.log(np.abs((\
1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.\
sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+4*x3**2*\
x4**2*(np.log(np.abs(1-2*x2))+(1j/2)*np.pi*np.sign(1-2*x2))*(np.log(\
np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*\
np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+8*\
x3*x4*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))**2-4*x3**2*x4*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.\
sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+\
2*np.sqrt((1-x1)*x1)*np.sin(x0)))**2-4*x3*x4**2*(np.log(np.abs((1-2*\
x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))+(1j/2)*np.pi*np.sign(\
(1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(x0)))**2+2*x3**2*\
x4**2*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*np.sin(\
x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*\
np.sin(x0)))**2-8*x3*(np.log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-\
x1)*x1)*(1-2*(1-x4/2))*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.\
cos(x0)+2*np.sqrt((1-x1)*x1)*(1-2*(1-x4/2))*np.sin(x0)))+4*x3**2*(np.\
log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*(1-2*(1-x4/2))*np.\
sin(x0)))+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*\
x1)*(1-2*(1-x4/2))*np.sin(x0)))-8*x3*(np.log(np.abs((1-2*x1)*np.cos(\
x0)+2*np.sqrt((1-x1)*x1)*(1-x4)*np.sin(x0)))+(1j/2)*np.pi*np.sign((1-\
2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*(1-x4)*np.sin(x0)))+4*x3**2*(np.\
log(np.abs((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*(1-x4)*np.sin(x0))\
)+(1j/2)*np.pi*np.sign((1-2*x1)*np.cos(x0)+2*np.sqrt((1-x1)*x1)*(1-x4)\
*np.sin(x0)))))/(1/100000+64*np.pi**6*np.sqrt(1-x1)*np.sqrt(x1)*np.\
sqrt(1-x2)*np.sqrt(x2)*(-2+x3)*x3*(-2+x4)*x4*np.sqrt(2-x5)*np.sqrt(x5)\
*np.sqrt(x6)*(2-x5+2*(-1+2*x1)*np.sqrt(2-x5)*np.sqrt(x5)*np.sqrt(x6)+\
x5*x6))))
    
    if part == 'real':
        return np.real(result)
    else:
        return np.imag(result)

@jit(nopython=True)
def f_fixed_x0(x, x0):
    """带有固定x0的被积函数"""
    full_x = np.zeros(8)
    full_x[0] = x0
    # 逐个赋值而不是使用切片
    for i in range(len(x)):
        full_x[i+1] = x[i]
    return f(full_x)

def integrate_for_x0(x0_val, config=None):
    """对于给定的x0值进行积分"""
    if config is None:
        config = IntegrationConfig()
        
    integ = vegas.Integrator([[0., 1.], [0., 1.], [0., 1.], 
                             [0., 1.], [0., 1.], [0., 1.], [0., 1.]], nproc=1)
    
    def integrand(x):
        return f_fixed_x0(x, x0_val)
    
    # 使用配置中的参数
    # 训练积分器
    integ(integrand, nitn=config.x0_distribution['training_iterations'],
          neval=config.x0_distribution['training_evaluations'])
    
    # 进行最终积分
    result = integ(integrand, nitn=config.x0_distribution['final_iterations'],
                  neval=config.x0_distribution['final_evaluations'])
    
    return result.mean, result.sdev

def process_x0(args):
    """处理单个x0值的函数"""
    x0, config = args
    mean, sdev = integrate_for_x0(x0, config)
    return mean, sdev

def plot_x0_distribution(config=None):
    """
    计算并绘制x0分布
    
    参数:
    config: IntegrationConfig实例，包含所有配置参数
    """
    if config is None:
        config = IntegrationConfig()
        
    import matplotlib.pyplot as plt
    from multiprocessing import Pool, cpu_count
    
    # 生成x0的值
    x0_range = config.x0_distribution['x0_range']
    n_points = config.x0_distribution['n_points']
    x0_values = np.linspace(x0_range[0], x0_range[1], n_points)
    
    # 创建进程池
    n_cores = config.n_cores
    print(f"使用 {n_cores} 个CPU核心进行并行计算")
    
    # 准备参数
    args = [(x0, config) for x0 in x0_values]
    
    # 创建结果目录，修改为 result_x0dist_...
    result_dir = f'result_x0dist_{config.part}_r-{format(config.r, ".1e")}_{config.get_timestamp()}'
    try:
        os.makedirs(result_dir, exist_ok=True)
        print(f"创建结果目录: {os.path.abspath(result_dir)}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        result_dir = '.'
    
    # 使用进程池并行计算
    with Pool(n_cores) as pool:
        # 使用imap来获取实时进度
        results = []
        total = len(x0_values)
        
        for i, (mean, sdev) in enumerate(pool.imap(process_x0, args)):
            print(f"计算进度: {i+1}/{total}")
            results.append((mean, sdev))
            
            if config.x0_distribution['save_intermediate']:
                # 实时保存结果
                integral_values = [r[0] for r in results]
                error_bars = [r[1] for r in results]
                output_file = os.path.join(result_dir, 
                    f'integral_vs_x0_{config.part}_r-{format(config.r, ".1e")}_intermediate.npz')
                np.savez(output_file,
                        x0_values=x0_values[:i+1],
                        integral_values=integral_values,
                        error_bars=error_bars,
                        config=config.to_dict())
    
    # 分离结果
    integral_values, error_bars = zip(*results)
    
    # 绘制结果
    plt.figure(figsize=config.x0_distribution['plot_size'])
    plt.errorbar(x0_values, integral_values, yerr=error_bars, 
                 fmt=config.x0_distribution['plot_style']['fmt'],
                 capsize=config.x0_distribution['plot_style']['capsize'],
                 markersize=config.x0_distribution['plot_style']['markersize'],
                 elinewidth=config.x0_distribution['plot_style']['elinewidth'])
    
    plt.xlabel(config.plot_config['xlabel'])
    plt.ylabel(config.plot_config['ylabel'])
    plt.title(config.plot_config['title'])
    
    # 单独调用 plt.grid()
    if config.x0_distribution['plot_style']['grid']:
        plt.grid(True)
    
    # 保存图像
    plot_file = os.path.join(result_dir, 
        f'integral_vs_x0_{config.part}_r-{format(config.r, ".1e")}.{config.plot_config["format"]}')
    plt.savefig(plot_file, dpi=config.plot_config['dpi'])
    
    # 保存最终数据
    data_file = os.path.join(result_dir, 
        f'integral_vs_x0_{config.part}_r-{format(config.r, ".1e")}_final.npz')
    np.savez(data_file,
             x0_values=x0_values,
             integral_values=integral_values,
             error_bars=error_bars,
             config=config.to_dict())
    
    if config.verbose:
        plt.show()
    
    return x0_values, integral_values, error_bars

def main(seed=None, config=None):
    """
    主积分函数
    
    参数:
    seed: 随机种子
    config: 积分配置
    """
    if config is None:
        config = IntegrationConfig()
        
    # 如果提供了种子，则使用它
    if seed is not None:
        gv.ranseed(seed)
    else:
        gv.ranseed((1, 2, 33))

    start_time = time.time()
    
    # 创建积分器并设置详细输出
    integ = vegas.Integrator(
        config.integration_ranges,
        nproc=1,  # 使用单核，因为我们在外层并行
    )
    
    # 适应性迭代阶段
    if config.verbose:
        print(f"\n=== 开始适应性迭代阶段 (seed={seed}) ===")
    training_result = integ(f, nitn=config.training_iterations, 
                          neval=config.training_evaluations)

    # 最终积分阶段
    if config.verbose:
        print(f"\n=== 开始最终积分阶段 (seed={seed}) ===")
    result = integ(f, nitn=config.final_iterations, 
                  neval=config.final_evaluations)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取每次迭代的结果
    iteration_results = result.itn_results
    
    # 输出当前进程的结果
    if config.verbose:
        print(f"\n=== 种子 {seed} 的计算结果 ===")
        print(result.summary())
        print(f'积分结果 = {result}')
        print(f'计算时间 = {total_time:.2f}秒')
    
    return result.mean, result.sdev, total_time, result, iteration_results, training_result

def run_parallel_integration(n_cores, seed_base=0, result_dir='.', config=None):
    """
    并行运行多个积分实例
    
    参数:
    n_cores: 使用的CPU核心数
    seed_base: 随机种子基础值
    result_dir: 保存结果的目录
    config: 积分配置
    """
    if config is None:
        config = IntegrationConfig()
        
    overall_start = time.time()
    
    print(f"=== 启动并行积分，使用 {n_cores} 个核心，种子基础值 {seed_base} ===")
    
    # 创建不同的随机种子，加入seed_base使每次循环的种子不同
    seeds = [(seed_base + i, seed_base + i+100, seed_base + i+200) for i in range(1, n_cores+1)]
    
    # 创建进程池并运行
    with Pool(n_cores) as pool:
        # 启动所有进程
        results = pool.map(main, seeds)
    
    # 解包结果
    means, sdevs, times = [], [], []
    full_results = []
    iteration_results = []
    training_results = []
    
    for result in results:
        if len(result) >= 6:  # 如果返回了完整的结果元组
            mean, sdev, run_time, full_result, iter_result, train_result = result
            means.append(mean)
            sdevs.append(sdev)
            times.append(run_time)
            full_results.append(full_result)
            iteration_results.append(iter_result)
            training_results.append(train_result)
        else:  # 兼容旧版本返回值
            mean, sdev, run_time = result
            means.append(mean)
            sdevs.append(sdev)
            times.append(run_time)
    
    # 使用方差的倒数作为权重来计算加权平均值
    weights = np.array([1.0/(sdev**2) for sdev in sdevs])
    weights = weights / np.sum(weights)  # 归一化权重
    
    # 计算加权平均值
    final_mean = np.sum(np.array(means) * weights)
    
    # 计算合并后的误差（正确处理独立测量的误差）
    # 对于独立测量，合并后的方差是各个方差倒数之和的倒数
    final_variance = 1.0 / np.sum([1.0/(s**2) for s in sdevs])
    final_sdev = np.sqrt(final_variance)
    
    # 计算总时间
    total_time = time.time() - overall_start
    max_time = max(times)
    avg_time = sum(times) / len(times)
    
    # 计算Q值 (chi^2/dof)
    chi2 = np.sum([(m - final_mean)**2 / s**2 for m, s in zip(means, sdevs)])
    dof = len(means) - 1  # 自由度
    Q = chi2 / dof if dof > 0 else 0
    
    # 收集系统信息
    system_info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'logical_cpu_count': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
    }
    
    # 输出结果
    print("\n=== 并行积分结果 ===")
    print(f"总核心数: {n_cores}")
    print(f"积分结果: {final_mean} ± {final_sdev}")
    print(f"相对误差: {100 * final_sdev / abs(final_mean):.2f}%")
    print(f"Q值: {Q:.2f}")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"最长单次运行时间: {max_time:.2f}秒")
    print(f"平均单次运行时间: {avg_time:.2f}秒")
    print(f"加速比: {max_time/total_time:.2f}x (理想情况下)")
    
    # 计算更多统计量
    mean_array = np.array(means)
    mean_of_means = np.mean(mean_array)
    std_of_means = np.std(mean_array)
    cv_of_means = std_of_means / abs(mean_of_means) if mean_of_means != 0 else float('inf')  # 变异系数
    
    # 构建输出文件路径
    output_file = os.path.join(result_dir, f'parallel_result_{config.part}_r-{format(config.r, ".1e")}_cores-{n_cores}_seedbase-{seed_base}.npz')
    
    # 保存单次结果到文件，包含更多信息
    np.savez(output_file,
             # 基本结果
             mean=final_mean,
             sdev=final_sdev,
             Q=Q,
             # 每个进程的详细结果
             weights=weights,
             individual_means=means,
             individual_sdevs=sdevs,
             individual_times=times,
             # 统计分析
             mean_of_means=mean_of_means,
             std_of_means=std_of_means,
             cv_of_means=cv_of_means,
             # 时间统计
             total_time=total_time,
             max_time=max_time,
             avg_time=avg_time,
             # 系统参数
             n_cores=n_cores,
             seed_base=seed_base,
             seeds=seeds,
             system_info=system_info,
             # 积分参数
             part=config.part,
             r=config.r,
             # 原始数据 (可选，文件可能会很大)
             # full_results=full_results,
             # iteration_results=iteration_results,
             # training_results=training_results,
            )
    
    return final_mean, final_sdev, total_time, Q, (means, sdevs, times)

def run_multiple_integrations(n_cores=None, n_iterations=None, config=None):
    """
    运行多次并行积分并整合所有结果
    
    参数:
    n_cores: 每次积分使用的核心数（可选，优先使用配置中的值）
    n_iterations: 运行并行积分的次数（可选，优先使用配置中的值）
    config: 积分配置
    """
    if config is None:
        config = IntegrationConfig()
    
    # 使用参数值或配置值
    n_cores = n_cores if n_cores is not None else config.n_cores
    n_iterations = n_iterations if n_iterations is not None else config.n_iterations
    
    # 创建结果目录
    result_dir = config.get_result_dir()
    try:
        os.makedirs(result_dir, exist_ok=True)
        print(f"创建结果目录: {os.path.abspath(result_dir)}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        result_dir = '.'  # 如果创建目录失败，使用当前目录
    
    overall_start = time.time()
    
    print(f"=== 开始运行 {n_iterations} 次并行积分 ===")
    
    all_means = []
    all_sdevs = []
    all_times = []
    all_Qs = []
    all_seed_bases = []
    all_detailed_results = []
    
    # 运行n_iterations次并行积分
    for i in range(1, n_iterations+1):
        # 为每次循环创建不同的种子基础值
        seed_base = i * 1000  # 使用循环索引乘以1000作为基础种子
        all_seed_bases.append(seed_base)
        
        print(f"\n=== 进行第 {i}/{n_iterations} 次并行积分，种子基础值: {seed_base} ===")
        if len(all_means) > 0:
            current_mean = np.mean(all_means)
            current_sdev = np.sqrt(1.0 / np.sum([1.0/(s**2) for s in all_sdevs]))
            print(f"当前累积结果: {current_mean} ± {current_sdev} (相对误差: {100 * current_sdev / abs(current_mean):.2f}%)")
        
        # 将结果目录传递给run_parallel_integration函数
        result = run_parallel_integration(n_cores, seed_base, result_dir, config)
        if len(result) >= 5:  # 使用新的返回格式
            mean, sdev, run_time, Q, detailed_results = result
            all_means.append(mean)
            all_sdevs.append(sdev)
            all_times.append(run_time)
            all_Qs.append(Q)
            all_detailed_results.append(detailed_results)
        else:  # 兼容旧版本的返回值
            mean, sdev, run_time = result
            all_means.append(mean)
            all_sdevs.append(sdev)
            all_times.append(run_time)
            all_Qs.append(0)  # 旧版本没有Q值
    
    # 使用方差的倒数作为权重来计算加权平均值
    weights = np.array([1.0/(sdev**2) for sdev in all_sdevs])
    weights = weights / np.sum(weights)  # 归一化权重
    
    # 计算加权平均值
    final_mean = np.sum(np.array(all_means) * weights)
    
    # 计算合并后的误差
    final_variance = 1.0 / np.sum([1.0/(s**2) for s in all_sdevs])
    final_sdev = np.sqrt(final_variance)
    
    # 计算总时间
    total_time = time.time() - overall_start
    avg_time = np.mean(all_times)
    max_time = np.max(all_times)
    min_time = np.min(all_times)
    std_time = np.std(all_times)
    
    # 计算Q值 (chi^2/dof)
    chi2 = np.sum([(m - final_mean)**2 / s**2 for m, s in zip(all_means, all_sdevs)])
    dof = len(all_means) - 1  # 自由度
    Q = chi2 / dof if dof > 0 else 0
    
    # 计算更多统计量
    mean_array = np.array(all_means)
    std_of_means = np.std(mean_array)
    cv_of_means = std_of_means / abs(final_mean) if final_mean != 0 else float('inf')  # 变异系数
    
    # 收集系统信息
    system_info = {
        'hostname': socket.gethostname(),
        'os': platform.system(),
        'os_version': platform.version(),
        'python': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'logical_cpu_count': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 积分参数
    integration_params = {
        'r': config.r,
        'part': config.part,
        'eulerGamma': config.eulerGamma,
        'n_cores': n_cores,
        'n_iterations': n_iterations,
    }
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(f"总运行次数: {n_iterations}")
    print(f"每次运行核心数: {n_cores}")
    print(f"积分结果 = {final_mean} ± {final_sdev}")
    print(f"Q值 = {Q:.2f}")
    print(f"相对误差 = {100 * final_sdev / abs(final_mean):.2f}%")
    print(f"总计算时间 = {total_time:.2f}秒")
    print(f"平均单次时间 = {avg_time:.2f}秒")
    print(f"时间变异系数 = {std_time/avg_time:.2f}")
    print(f"所有结果已保存到目录: {os.path.abspath(result_dir)}")
    
    # 保存结果到文件，更加全面的输出
    output_file = os.path.join(result_dir, f'final_result_{config.part}_r-{format(config.r, ".1e")}_cores-{n_cores}_iters-{n_iterations}.npz')
    np.savez(output_file,
             # 基本结果
             mean=final_mean,
             sdev=final_sdev,
             Q=Q,
             relative_error_percent=100 * final_sdev / abs(final_mean),
             # 每次迭代的结果
             individual_means=all_means,
             individual_sdevs=all_sdevs,
             individual_times=all_times,
             individual_Qs=all_Qs,
             # 统计分析
             weights=weights,
             std_of_means=std_of_means,
             cv_of_means=cv_of_means,
             chi2=chi2,
             dof=dof,
             # 时间统计
             total_time=total_time,
             avg_time=avg_time,
             max_time=max_time,
             min_time=min_time,
             std_time=std_time,
             time_efficiency=avg_time/(n_cores*avg_time/n_iterations) if avg_time > 0 else 0,
             # 系统和随机种子信息
             seed_bases=all_seed_bases,
             n_cores=n_cores,
             n_iterations=n_iterations,
             system_info=system_info,
             integration_params=integration_params,
             # 详细的原始数据 (可选)
             # detailed_results=all_detailed_results,
            )
    
    # 同时保存一个简单的文本结果文件，便于快速查看
    try:
        txt_file = os.path.join(result_dir, f'result_summary_{config.part}_r-{format(config.r, ".1e")}.txt')
        with open(txt_file, 'w') as f:
            f.write(f"=== 积分结果摘要 ===\n")
            f.write(f"计算日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"系统: {platform.system()} {platform.version()}\n")
            f.write(f"参数: part={config.part}, r={config.r}\n")
            f.write(f"并行设置: {n_cores}核心 x {n_iterations}次迭代\n")
            f.write(f"最终结果: {final_mean} ± {final_sdev}\n")
            f.write(f"相对误差: {100 * final_sdev / abs(final_mean):.2f}%\n")
            f.write(f"Q值: {Q:.2f}\n")
            f.write(f"总计算时间: {total_time:.2f}秒\n")
            f.write(f"性能统计: 平均时间={avg_time:.2f}秒, 最长={max_time:.2f}秒, 最短={min_time:.2f}秒\n")
            f.write(f"结果文件目录: {os.path.abspath(result_dir)}\n")
    except Exception as e:
        print(f"无法创建文本摘要文件: {e}")
    
    # 也将最终结果文件复制到当前目录
    try:
        final_output_file = f'final_result_{config.part}_r-{format(config.r, ".1e")}.npz'
        np.savez(final_output_file,
                 mean=final_mean,
                 sdev=final_sdev,
                 Q=Q,
                 relative_error_percent=100 * final_sdev / abs(final_mean),
                 total_time=total_time,
                 n_cores=n_cores,
                 n_iterations=n_iterations,
                 r=config.r,
                 part=config.part,
                 result_dir=result_dir)
    except Exception as e:
        print(f"无法创建当前目录下的结果文件: {e}")
    
    return final_mean, final_sdev, Q, total_time, result_dir

def main2(
    n_cores=None,
    n_iterations=None,
    training_evaluations=None,
    final_evaluations=None,
    x0_distribution_params=None,
    run_x0_distribution=False
):
    """
    主运行函数，支持可选参数配置
    
    参数:
    n_cores: 可选，使用的CPU核心数
    n_iterations: 可选，运行的迭代次数
    training_evaluations: 可选，训练阶段的评估次数
    final_evaluations: 可选，最终阶段的评估次数
    x0_distribution_params: 可选，x0分布计算的参数字典
    run_x0_distribution: 是否运行x0分布计算，默认False
    """
    # 创建配置实例
    config = IntegrationConfig()
    
    # 仅在提供了参数时才更新配置
    if n_cores is not None:
        config.n_cores = n_cores
    if n_iterations is not None:
        config.n_iterations = n_iterations
    if training_evaluations is not None:
        config.training_evaluations = training_evaluations
    if final_evaluations is not None:
        config.final_evaluations = final_evaluations
    
    # 如果提供了x0分布参数，则更新配置
    if x0_distribution_params is not None:
        config.x0_distribution.update(x0_distribution_params)
    
    if run_x0_distribution:
        # 计算x0分布
        x0_values, integral_values, error_bars = plot_x0_distribution(config)
        return x0_values, integral_values, error_bars
    else:
        # 运行常规积分
        result, error, Q, total_time, result_dir = run_multiple_integrations(config=config)
        
        # 输出最终结果
        print("\n=== 最终结果 ===")
        print(f'积分结果 = {result} ± {error}')
        print(f'Q值 = {Q:.2f}')
        print(f'相对误差 = {100 * error / abs(result):.2f}%')
        print(f'总计算时间 = {total_time:.2f}秒')
        
        # 保存最简结果到当前目录的文件
        np.savez('final_result_'+config.part+'_r-'+str(format(config.r, '.1e'))+'.npz',
                 mean=result,
                 sdev=error,
                 Q=Q,
                 relative_error_percent=100 * error / abs(result),
                 total_time=total_time,
                 config=config.to_dict(),
                 result_dir=result_dir)
        
        return result, error, Q, total_time, result_dir

if __name__ == '__main__':
    # 示例：使用默认配置运行
    main2(run_x0_distribution=True)
    
    # 示例：使用自定义配置运行
    # custom_x0_params = {
    #     'n_points': 50,
    #     'training_evaluations': 1e6,
    #     'final_evaluations': 1e7
    # }
    # main2(
    #     n_cores=10,
    #     n_iterations=4,
    #     training_evaluations=1e3,
    #     final_evaluations=1e3,
    #     x0_distribution_params=custom_x0_params,
    #     run_x0_distribution=False
    # )



# Copyright (c) 2013-22 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
