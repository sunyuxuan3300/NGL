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
    nm = 4*(-1+2*x[1])*np.cos(1*x[0])*(T1+T2+T3+sign_terms)
    
    # 计算分母
    de = (np.pi**3 * sqrt_1mx1 * sqrt_x1 * sqrt_1mx2 * sqrt_x2 * 
          sqrt_2mx3 * sqrt_x3 * (-2+x[4]) * (-2+x[5]) * sqrt_x6 * 
          (2-x[3]+2*(-1+2*x[1])*sqrt_2mx3*sqrt_x3*sqrt_x6+x[3]*x[6]))
    
    # 返回虚部
    return np.imag(nm/de)



@jit(nopython=True)
def f_fixed_x0(x, x0):
    """对于给定的x0值计算被积函数"""
    # 创建完整的x数组，包含固定的x0
    full_x = np.zeros(7)
    full_x[0] = x0
    # 逐个赋值而不是使用切片
    for i in range(len(x)):
        full_x[i+1] = x[i]
    return f(full_x)

def integrate_for_x0(x0):
    """对于给定的x0值进行积分"""
    # 创建积分器
    integ = vegas.Integrator([[0., 1.], [0., 1.], [0., 1.], 
                             [0., 1.], [0., 1.], [0., 1.]], nproc=1)  # 设置nproc=1避免多进程问题
    
    # 创建一个包装函数
    def wrapper(x):
        return f_fixed_x0(x, x0)
    
    # 训练积分器
    integ(wrapper, nitn=10, neval=1e6)
    
    # 进行最终积分
    result = integ(wrapper, nitn=10, neval=1e6)
    
    return result.mean, result.sdev

def process_x0(x0):
    """处理单个x0值的函数"""
    mean, sdev = integrate_for_x0(x0)
    return mean, sdev

def plot_x0_distribution():
    import matplotlib.pyplot as plt
    from multiprocessing import Pool, cpu_count
    
    # 生成x0的值
    x0_values = np.linspace(0, 2*np.pi, 50)
    
    # 创建进程池
    n_cores = cpu_count() - 1  # 留一个核心给系统
    print(f"使用 {n_cores} 个CPU核心进行并行计算")
    
    # 使用进程池并行计算
    with Pool(n_cores) as pool:
        # 使用imap来获取实时进度
        results = []
        total = len(x0_values)
        
        for i, (mean, sdev) in enumerate(pool.imap(process_x0, x0_values)):
            print(f"计算进度: {i+1}/{total}")
            results.append((mean, sdev))
            
            # 实时保存结果
            integral_values = [r[0] for r in results]
            error_bars = [r[1] for r in results]
            np.savez('integration_results.npz', 
                     x0_values=x0_values[:i+1], 
                     integral_values=integral_values,
                     error_bars=error_bars)
    
    # 分离结果
    integral_values, error_bars = zip(*results)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.errorbar(x0_values, integral_values, yerr=error_bars, fmt='o', 
                capsize=5, markersize=3, elinewidth=1)
    plt.xlabel('x[0]')
    plt.ylabel('Integral Value')
    plt.title('Integration Result vs x[0]')
    plt.grid(True)
    plt.savefig('integral_vs_x0.png')
    plt.show()

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
    plot_x0_distribution()


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
