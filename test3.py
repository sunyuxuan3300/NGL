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

from numpy import log
from numpy import abs
from numpy import sin
from numpy import cos
from numpy import pi
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True


@jit(nopython=True)
def f(x):
    x0, x1, x2, x3, x4, x5, x6 = x
    result = (-4*np.pi*(-1+x1**2)*(-1+2*x3)*(4*np.sign(-1+2*x2)+np.sign((1-2*x2-2*x3+4*\
x2*x3-4*np.sqrt(-((-1+x2)*x2))*np.sqrt(-((-1+x3)*x3)))*np.cos(x0)+2*(\
np.sqrt(-((-1+x2)*x2))-2*np.sqrt(-((-1+x2)*x2))*x3+np.sqrt(-((-1+x3)*\
x3))-2*x2*np.sqrt(-((-1+x3)*x3)))*np.sin(x0))+np.sign((1-2*x2-2*x3+4*x2*\
x3+4*np.sqrt(-((-1+x2)*x2))*np.sqrt(-((-1+x3)*x3)))*np.cos(x0)+2*(-np.\
sqrt(-((-1+x2)*x2))+2*np.sqrt(-((-1+x2)*x2))*x3+np.sqrt(-((-1+x3)*x3))\
-2*x2*np.sqrt(-((-1+x3)*x3)))*np.sin(x0))+np.sign((1-2*x2-2*x3+4*x2*x3+4*\
np.sqrt(-((-1+x2)*x2))*np.sqrt(-((-1+x3)*x3)))*np.cos(x0)+2*(np.sqrt(-\
((-1+x2)*x2))-2*np.sqrt(-((-1+x2)*x2))*x3-np.sqrt(-((-1+x3)*x3))+2*x2*\
np.sqrt(-((-1+x3)*x3)))*np.sin(x0))+np.sign((1-2*x2-2*x3+4*x2*x3-4*np.\
sqrt(-((-1+x2)*x2))*np.sqrt(-((-1+x3)*x3)))*np.cos(x0)+2*(-np.sqrt(-((\
-1+x2)*x2))+2*np.sqrt(-((-1+x2)*x2))*x3-np.sqrt(-((-1+x3)*x3))+2*x2*\
np.sqrt(-((-1+x3)*x3)))*np.sin(x0))))/(np.sqrt(-((-1+x2)*x2))*np.sqrt(\
-((-1+x3)*x3))*(1+x1**2+x1*(-2+4*x3))*(x1**2*(-1+x6)-x6))
    return result

@jit(nopython=True)
def f_fixed_x0_jit(x0, x):
    """JIT编译的固定x0的被积函数"""
    full_x = np.zeros(7)
    full_x[0] = x0
    # 逐个赋值替代切片赋值
    for i in range(len(x)):
        full_x[i+1] = x[i]
    return f(full_x)

def integrate_single_point(x0):
    """对单个x0点进行积分"""
    integ = vegas.Integrator([
        [0, 1],      # x[1]
        [0, 1],       # x[2]
        [0, 1],       # x[3]
        [0, 1],       # x[4]
        [0, 1],       # x[5]
        [0, 1],       # x[6]
    ])
    
    def f_wrapper(x):
        return f_fixed_x0_jit(x0, x)
    
    # 进行积分
    integ(f_wrapper, nitn=10, neval=1e6)
    result = integ(f_wrapper, nitn=20, neval=1e6)
    print(f"x0 = {x0:.2f}, result = {result.mean:.6f} ± {result.sdev:.6f}")
    return result.mean, result.sdev

def main():
    # 创建x0的采样点
    x0_points = np.linspace(0, 2*pi, 50)
    
    # 获取CPU核心数
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # 并行计算
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # 并行提交所有积分任务
        future_results = list(executor.map(integrate_single_point, x0_points))
        
        # 收集结果
        for result, error in future_results:
            results.append(result)
            errors.append(error)
    
    # 转换为numpy数组以便后续处理
    results = np.array(results)
    errors = np.array(errors)
    
    # 绘制结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.errorbar(x0_points, results, yerr=errors, fmt='o-', capsize=5, markersize=3)
    plt.xlabel('x[0]')
    plt.ylabel('Integral Value')
    plt.title('Integration Results vs x[0]')
    plt.grid(True)
    plt.show()
    
    # 保存数据到文件
    np.savez('integration_results.npz', 
             x0=x0_points, 
             results=results, 
             errors=errors)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

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
