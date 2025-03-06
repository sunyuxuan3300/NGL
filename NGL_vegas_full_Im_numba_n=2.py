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
    """
    参数:
    x: 包含7个元素的numpy数组，对应x[0]到x[6]
    """
    # 辅助函数，用于简化重复的sqrt计算
    def sqrt_terms():
        sqrt_1_x1 = np.sqrt(1 - x[1])
        sqrt_x1 = np.sqrt(x[1])
        sqrt_1_x2 = np.sqrt(1 - x[2])
        sqrt_x2 = np.sqrt(x[2])
        return sqrt_1_x1, sqrt_x1, sqrt_1_x2, sqrt_x2
    
    # 获取所有sqrt项
    sqrt_1_x1, sqrt_x1, sqrt_1_x2, sqrt_x2 = sqrt_terms()
    
    # 公共项计算
    common_term1 = 1 - 2*x[1] - 2*x[2] + 4*x[1]*x[2]
    common_term2 = 4*sqrt_1_x1*sqrt_x1*sqrt_1_x2*sqrt_x2
    
    # 计算主要的项（只替换变量相关的log为sign，保持常数的log）
    log_sign_terms = (8 - 2*x[3] - 2*x[4] + 16*np.log(2) - 8*x[3]*np.log(2) - 8*x[4]*np.log(2) + 
                     x[3]*x[4]*np.log(16) + 2*(-2 + x[3])*(-2 + x[4])*np.sign(1 - x[1]) + 
                     2*(-2 + x[3])*(-2 + x[4])*np.sign(x[1]) + 8*np.sign(1 - x[2]) - 
                     4*x[3]*np.sign(1 - x[2]) - 4*x[4]*np.sign(1 - x[2]) + 2*x[3]*x[4]*np.sign(1 - x[2]) + 
                     8*np.sign(x[2]) - 4*x[3]*np.sign(x[2]) - 4*x[4]*np.sign(x[2]) + 2*x[3]*x[4]*np.sign(x[2]) - 
                     16*np.sign(1 - 2*x[2]) + 8*x[3]*np.sign(1 - 2*x[2]) + 
                     8*x[4]*np.sign(1 - 2*x[2]) - 4*x[3]*x[4]*np.sign(1 - 2*x[2]))
    
    # 计算abs项的符号组合（移除嵌套的绝对值）
    term1 = (common_term1 - common_term2)*np.cos(x[0]) + \
            2*(sqrt_1_x1*sqrt_x1*(1-2*x[2]) + sqrt_1_x2*sqrt_x2 - \
               2*x[1]*sqrt_1_x2*sqrt_x2)*np.sin(x[0])
    
    term2 = (common_term1 + common_term2)*np.cos(x[0]) + \
            2*(sqrt_1_x1*sqrt_x1*(1-2*x[2]) - sqrt_1_x2*sqrt_x2 + \
               2*x[1]*sqrt_1_x2*sqrt_x2)*np.sin(x[0])
    
    term3 = (common_term1 + common_term2)*np.cos(x[0]) + \
            2*(sqrt_1_x2*sqrt_x2 - 2*x[1]*sqrt_1_x2*sqrt_x2 + \
               sqrt_1_x1*sqrt_x1*(-1 + 2*x[2]))*np.sin(x[0])
    
    term4 = (common_term1 - common_term2)*np.cos(x[0]) + \
            2*(-sqrt_1_x2*sqrt_x2 + 2*x[1]*sqrt_1_x2*sqrt_x2 + \
               sqrt_1_x1*sqrt_x1*(-1 + 2*x[2]))*np.sin(x[0])
    
    abs_sign_terms = 1*(-4 + 2*x[3] + 2*x[4] - x[3]*x[4])*np.sign(term1) + \
                     1*(-4 + 2*x[3] + 2*x[4] - x[3]*x[4])*np.sign(term2) + \
                     1*(-4 + 2*x[3] + 2*x[4] - x[3]*x[4])*np.sign(term3) + \
                     1*(-4 + 2*x[3] + 2*x[4] - x[3]*x[4])*np.sign(term4)
    
    # 分母项
    denominator = (sqrt_1_x1 * sqrt_x1 * sqrt_1_x2 * sqrt_x2 * 
                  (-2 + x[3]) * (-2 + x[4]) * 
                  (1 + (-2 + 4*x[1])*x[6] + x[6]**2) * 
                  (-x[5] + (-1 + x[5])*x[6]**2))
    
    # 最终结果（使用新的符号项）
    result = -((-1 + 2*x[1]) * (-1 + x[6]**2)  * np.cos(x[0]) * (log_sign_terms + abs_sign_terms)) / denominator / (np.pi**2/4)

    return result














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
