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


@jit  # 使用Numba加速
def safe_sqrt_scalar(t, eps=1e-12):
    """标量版本的数值安全sqrt"""
    t_clipped = max(eps, min(t, 1.0 - eps))
    return np.sqrt(t_clipped)

@jit
def safe_log_scalar(t, eps=1e-12):
    """标量版本的数值安全log"""
    return np.log(max(eps, np.abs(t)))

@jit(nopython=True)
def f(x):
    # 变量别名简化 (x0, x1...x6 对应 x[0], x[1]...x[6])
    x0, x1, x2 = x[0], x[1], x[2]
    x3, x4, x5, x6 = x[3], x[4], x[5], x[6]
    
    
    # 公共中间项
    sqrt_1_x1 = safe_sqrt_scalar(1 - x1)
    sqrt_x1 = safe_sqrt_scalar(x1)
    sqrt_1_x2 = safe_sqrt_scalar(1 - x2)
    sqrt_x2 = safe_sqrt_scalar(x2)
    cross_sqrt = sqrt_1_x1 * sqrt_x1 * sqrt_1_x2 * sqrt_x2  # 重复使用的四根号乘积
    
    # 分子计算 ----------------------------------------------
    # 第一部分：多项式部分
    poly_part = (
        4 - x3 - x4
        - 4 * x3 * np.log(2)
        - 4 * x4 * np.log(2)
        + x3 * x4 * np.log(4)
        + np.log(256)
    )
    
    # 第二部分：对数组合项
    log_term_1 = ( (x3 - 2) * (x4 - 2) ) * (safe_log_scalar(1 - x1) + safe_log_scalar(x1))
    
    log_term_2 = (
        4 * safe_log_scalar(1 - x2)
        - 2 * x3 * safe_log_scalar(1 - x2)
        - 2 * x4 * safe_log_scalar(1 - x2)
        + x3 * x4 * safe_log_scalar(1 - x2)
        + 4 * safe_log_scalar(x2)
        - 2 * x3 * safe_log_scalar(x2)
        - 2 * x4 * safe_log_scalar(x2)
        + x3 * x4 * safe_log_scalar(x2)
    )
    
    # 第三部分：复杂对数项（抽象公共计算）
    def compute_log_block(coeff, inner):
        """处理每个对数块的结构：coeff * log(|inner|)"""
        return coeff * safe_log_scalar(inner)
    
    # 设计对数项的系数模式
    coeff_pattern = (-4 + 2*x3 + 2*x4 - x3*x4)
    
    # 四个不同的对数项参数
    log_arg1 = 1 - 2*x1 - 4*cross_sqrt - 2*x2 + 4*x1*x2
    log_arg2 = 1 - 2*x1 + 4*cross_sqrt - 2*x2 + 4*x1*x2
    log_arg3 = np.cos(x0)*(1 - 2*x2) - 2*sqrt_1_x2*sqrt_x2*np.sin(x0)
    log_arg4 = np.cos(x0)*(1 - 2*x2) + 2*sqrt_1_x2*sqrt_x2*np.sin(x0)
    
    # 组合所有对数项
    log_block = (
        compute_log_block(coeff_pattern, log_arg1)
        + compute_log_block(coeff_pattern, log_arg2)
        + compute_log_block(coeff_pattern, log_arg3)
        + compute_log_block(coeff_pattern, log_arg4)
    )
    
    # 合并分子所有部分
    numerator = (
        -2 * (2*x1 - 1)  # 原始形式为 -2*(-1 + 2x1)
        * (x6**2 - 1)    # 原始形式为 (-1 + x6^2)
        * np.cos(0*x[0]) * (poly_part + log_term_1 + log_term_2 + log_block)
    )
    
    # 分母计算 ----------------------------------------------
    denominator = (
        sqrt_1_x1 * sqrt_x1 
        * sqrt_1_x2 * sqrt_x2
        * (x3 - 2) * (x4 - 2)  # 原始形式为 (-2 + x3)(-2 + x4)
        * (x6**2 + (4*x1 - 2)*x6 + 1)
        * ( (x5 - 1)*x6**2 - x5 )
    )
    
    # 最终结果
    return numerator / denominator / (np.pi**2 / 4)














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
