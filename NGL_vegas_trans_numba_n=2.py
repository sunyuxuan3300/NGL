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

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True


@jit(nopython=True)
def f(x):
    
#    x[0]= (33/12)* np.pi
    
    r1 = 4 - 6*x[1]**4 + 4*x[1]**8 - x[1]**12
    r2 = 4 - 6*x[2]**4 + 4*x[2]**8 - x[2]**12
    v1 = -1 + x[1]**4
    v2 = -1 + x[2]**4
    
    Q1 = 1-8*x[1]**4 + 12*x[1]**8 - 8*x[1]**12 + 2*x[1]**16
    Q2 = 1-8*x[2]**4 + 12*x[2]**8 - 8*x[2]**12 + 2*x[2]**16
    
    F1p = 1 + 4 * np.sqrt(r1) * np.sqrt(r2) * v1**2 * v2**2 * x[1]**2 * x[2]**2 - 2 * r2 * x[2]**4 + 2 * r1 * x[1]**4 * ( -1 + 2 * r2 * x[2]**4)
    F1m = 1 - 4*np.sqrt(r1) * np.sqrt(r2) * v1**2 *v2**2 * x[1]**2 * x[2]**2 - 2*r2 *x[2]**4 + 2* r1 * x[1]**4 * (-1+2*r2*x[2]**4)
    F2p = np.sqrt(r2) * v2**2 * x[2]**2 - 2 * r1 * np.sqrt(r2) * v2**2 * x[1]**4 * x[2]**2 + np.sqrt(r1) * v1**2 * x[1]**2 * (1-2*r2*x[2]**4)
    F2m = np.sqrt(r2) * v2**2 * x[2]**2 - 2 * r1 * np.sqrt(r2) * v2**2 * x[1]**4 * x[2]**2 - np.sqrt(r1) * v1**2 * x[1]**2 * (1-2*r2*x[2]**4)
    
    T1 = 8*x[4] + 8*x[5] - 4*x[4]**2 * x[5] - 4*x[4] * x[5]**2
    T2 = 2 * (-2+x[4]**2) * (-2+x[5]**2) * np.log(r1) + 2 * (-2+x[4]**2) * (-2+x[5]**2) * np.log(r2)
    T3 = 8 * (-2+x[4]**2) * (-2+x[5]**2) * (np.log(x[1]) + np.log(x[2]) + np.log(1-x[1]**4) + np.log(1-x[2]**4))
    T4 = -4 * (-2+x[4]**2) * (-2+x[5]**2) * np.log(np.abs(Q2))
    T5 =  - ((-2+x[4]**2) * (-2+x[5]**2) * np.log( (1/16) *  np.abs( F1p*np.cos(x[0]) - 2*F2m*np.sin(x[0]) ) * np.abs( F1p*np.cos(x[0]) + 2*F2m * np.sin(x[0]) ) * np.abs( F1m*np.cos(x[0]) - 2*F2p*np.sin(x[0]) ) * np.abs( F1m*np.cos(x[0]) + 2*F2p * np.sin(x[0]) )  )) 
    
    nm = (16384*v2*x[1]*(-1 + 9*x[1]**4 - 20*x[1]**8 + 20*x[1]**12 - 10*x[1]**16 + 2*x[1]**20) * x[2] * x[3] * x[6] * np.cos(2*x[0]) * (  T1+T2+T3+T4+ T5 ))
    de = np.pi**3 * np.sqrt(2-x[1]**4) * np.sqrt(2-x[2]**4) * np.sqrt(2-2*x[1]**4+x[1]**8) * np.sqrt(2 - 2*x[2]**4 + x[2]**8) * np.sqrt(2-x[3]**4) * (-2+x[4]**2) * (-2+x[5]**2) * (-2 + 2 * Q1 * x[3]**2 * np.sqrt(2-x[3]**4) *x[6]**2 -x[3]**4 *(-1+x[6]**4))  
    return nm/de













def main():
    # seed the random number generator so results reproducible
    gv.ranseed((1, 2, 33))

    start_time = time.time()  # 记录开始时间
    
    # 创建积分器并设置详细输出
    integ = vegas.Integrator(
        [ [0.,2*np.pi], [0., 1.], [0., 1.], [0., 1.],[0., 1.], [0., 1.], [0., 1.]], 
        nproc=13,
        analyzer=vegas.reporter()
    )

    # 适应性迭代阶段
    print("\n=== 开始适应性迭代阶段 ===")
    training_result = integ(f, nitn=50, neval=1e7, analyzer=vegas.reporter())

    # 最终积分阶段
    print("\n=== 开始最终积分阶段 ===")
    result = integ(f, nitn=200, neval=2e8, analyzer=vegas.reporter())
    
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(result.summary())
    print('积分结果 = %s' % result)
    print('Q值 = %.2f' % result.Q)
    print('相对误差 = %.2f%%' % (100 * result.sdev / abs(result.mean)))
    print('总计算时间 = %.2f秒' % total_time)  # 使用我们自己计算的时间
    
    if SHOW_GRID:
        integ.map.show_grid(20)

if __name__ == '__main__':
    main()


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
