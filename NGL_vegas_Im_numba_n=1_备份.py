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
    # 定义常用的平方根表达式
    sqrt_1mx1 = np.sqrt(1-x[1])
    sqrt_x1 = np.sqrt(x[1])
    sqrt_1mx2 = np.sqrt(1-x[2])
    sqrt_x2 = np.sqrt(x[2])
    sqrt_2mx3 = np.sqrt(2-x[3])
    sqrt_x3 = np.sqrt(x[3])
    sqrt_x6 = np.sqrt(x[6])
    
    # 定义F1m, F1p等复合表达式
    F1m = 1-2*x[1]-4*sqrt_1mx1*sqrt_x1*sqrt_1mx2*sqrt_x2-2*x[2]+4*x[1]*x[2]
    F1p = 1-2*x[1]+4*sqrt_1mx1*sqrt_x1*sqrt_1mx2*sqrt_x2-2*x[2]+4*x[1]*x[2]
    
    # 定义各种Sin项中的表达式
    S1 = 2*sqrt_1mx1*sqrt_x1*(1-2*x[2])+2*sqrt_1mx2*sqrt_x2-4*x[1]*sqrt_1mx2*sqrt_x2
    S2 = 2*(sqrt_1mx1*sqrt_x1*(1-2*x[2])-sqrt_1mx2*sqrt_x2+2*x[1]*sqrt_1mx2*sqrt_x2)
    S3 = 2*sqrt_1mx2*sqrt_x2-4*x[1]*sqrt_1mx2*sqrt_x2+2*sqrt_1mx1*sqrt_x1*(-1+2*x[2])
    S4 = -2*sqrt_1mx2*sqrt_x2+4*x[1]*sqrt_1mx2*sqrt_x2+2*sqrt_1mx1*sqrt_x1*(-1+2*x[2])
    
    # 计算分子部分
    T1 = 8-2*x[4]-2*x[5]+16*np.log(2)-8*x[4]*np.log(2)-8*x[5]*np.log(2)+x[4]*x[5]*np.log(16)
    T2 = 2*(-2+x[4])*(-2+x[5])*(np.log(1-x[1])+np.log(x[1]))
    T3 = (8-4*x[4]-4*x[5]+2*x[4]*x[5])*(np.log(1-x[2])+np.log(x[2]))
    
    # Sign项的系数
    coef1 = 8*np.pi*1j
    coef2 = -4*np.pi*1j*x[4]
    coef3 = -4*np.pi*1j*x[5]
    coef4 = 2*np.pi*1j*x[4]*x[5]
    
    # 计算所有Sign项
    sign_terms = (
        (coef1 + coef2 + coef3 + coef4) * np.sign(-1+2*x[2]) +
        (2*np.pi*1j - np.pi*1j*x[4] - np.pi*1j*x[5] + 0.5*np.pi*1j*x[4]*x[5]) * (
            1*np.sign(-F1m*np.cos(x[0])-S1*np.sin(x[0])) +
            1*np.sign(-F1p*np.cos(x[0])-S2*np.sin(x[0])) +
            1*np.sign(-F1p*np.cos(x[0])-S3*np.sin(x[0])) +
            1*np.sign(-F1m*np.cos(x[0])-S4*np.sin(x[0]))
        )
    )
    
    # 计算分子
    nm = 4*(-1+2*x[1])*np.cos(2*x[0])*(T1+T2+T3+sign_terms)
    
    # 计算分母
    de = (np.pi**3 * sqrt_1mx1 * sqrt_x1 * sqrt_1mx2 * sqrt_x2 * 
          sqrt_2mx3 * sqrt_x3 * (-2+x[4]) * (-2+x[5]) * sqrt_x6 * 
          (2-x[3]+2*(-1+2*x[1])*sqrt_2mx3*sqrt_x3*sqrt_x6+x[3]*x[6]))
    
    # 返回虚部
    return np.imag(nm/de)













def main():
    # seed the random number generator so results reproducible
    gv.ranseed((1, 2, 33))

    start_time = time.time()  # 记录开始时间
    
    # 创建积分器并设置详细输出
    integ = vegas.Integrator(
        [[0., 2*np.pi], [0., 1.], [0., 1.], [0., 1.],[0., 1.], [0., 1.], [0., 1.]], 
        nproc=13,
        analyzer=vegas.reporter()
    )

    # 适应性迭代阶段
    print("\n=== 开始适应性迭代阶段 ===")
    training_result = integ(f, nitn=50, neval=1e7, analyzer=vegas.reporter())

    # 最终积分阶段
    print("\n=== 开始最终积分阶段 ===")
    result = integ(f, nitn=1000, neval=2e8, analyzer=vegas.reporter())
    
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
