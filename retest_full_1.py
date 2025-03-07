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

r = 1
eulerGamma = 0.5772156649
part = 'imag'

@jit(nopython=True)
def FF1(x2, t5, t5b, t6, t6b, x0, x3, x4):
    return 1-2*x4+r*x2*(1+4*(-1+2*t6)*np.sqrt(1-x3)*np.sqrt(x3)*np.sqrt(1-x4)*\
np.sqrt(x4)-2*x4+x3*(-2+4*x4))*np.cos(x0)+2*r*x2*((1-2*t5)*np.sqrt(1-\
x4)*np.sqrt(x4)+2*(-1+2*t5)*x3*np.sqrt(1-x4)*np.sqrt(x4)+np.sqrt(1-x3)\
*np.sqrt(x3)*(-4*np.sqrt(t5)*np.sqrt(t5b)*np.sqrt(t6)*np.sqrt(t6b)+(-\
1+2*t6)*(-1+2*x4)+t5*(-2+4*t6+4*x4-8*t6*x4)))*np.sin(x0)

@jit(nopython=True)
def FF2(x2, t5, t5b, t6, t6b, x0, x3, x4):
    return (4*(-1+2*t6)*x2*np.sqrt(1-x3)*np.sqrt(x3)*np.sqrt(1-x4)*np.sqrt(x4)+\
x2*(-1+2*x3)*(-1+2*x4)+(r-2*r*x4)*np.cos(x0)+2*r*(1-2*t5)*np.sqrt(1-\
x4)*np.sqrt(x4)*np.sin(x0))/r

@jit(nopython=True)
def FF1binv(x2, t5, t5b, t6, t6b, x0, x3, x4):
    return x2-2*x2*x4+r*(1+4*(-1+2*t6)*np.sqrt(1-x3)*np.sqrt(x3)*np.sqrt(1-x4)*\
np.sqrt(x4)-2*x4+x3*(-2+4*x4))*np.cos(x0)+2*r*((-1+2*t5)*(-1+2*x3)*np.\
sqrt(1-x4)*np.sqrt(x4)+np.sqrt(1-x3)*np.sqrt(x3)*(1-2*t5-2*t6+4*t5*t6-\
4*np.sqrt(t5)*np.sqrt(t5b)*np.sqrt(t6)*np.sqrt(t6b)-2*(-1+2*t5)*(-1+2*\
t6)*x4))*np.sin(x0)

@jit(nopython=True)
def FF2binv(x2, t5, t5b, t6, t6b, x0, x3, x4):
    return (4*(-1+2*t6)*np.sqrt(1-x3)*np.sqrt(x3)*np.sqrt(1-x4)*np.sqrt(x4)+(-1+\
2*x3)*(-1+2*x4)+r*x2*(1-2*x4)*np.cos(x0)+2*r*(1-2*t5)*x2*np.sqrt(1-x4)\
*np.sqrt(x4)*np.sin(x0))/r

@jit(nopython=True)
def f(x):
    x0, x1, x2, x3, x4, x5, x6, x7 = x
    
    result = 1024*np.pi**6 * np.cos(0*x0)*((1-x1**2)*(-8*x2*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+x1**2+x1*(-2+\
4*x3))*(-2+x5)*x5-4*x2*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+x1**2+x1*(\
-2+4*x3))*(-2+x5)*x5*(-2+x6)+4*(x2**2*(-1+2*x3)+x1**6*x2**2*(-1+2*x3)+\
x1**2*(-2+4*x3+11*x2**2*(-1+2*x3)+x2**4*(-2+4*x3)+8*x2*(1-3*x3+3*\
x3**2)+8*x2**3*(1-3*x3+3*x3**2))+x1**4*(-2+4*x3+11*x2**2*(-1+2*x3)+\
x2**4*(-2+4*x3)+8*x2*(1-3*x3+3*x3**2)+8*x2**3*(1-3*x3+3*x3**2))+x1*x2*\
(-3+6*x3+x2**2*(-3+6*x3)+x2*(3-8*x3+8*x3**2))+x1**5*x2*(-3+6*x3+x2**2*\
(-3+6*x3)+x2*(3-8*x3+8*x3**2))+2*x1**3*(2*(1-2*x3)**2+2*x2**4*(1-2*x3)\
**2+5*x2*(-1+2*x3)+5*x2**3*(-1+2*x3)+3*x2**2*(3-8*x3+8*x3**2)))*(2-x5)\
*x5*(2-x6)*x6-8*x2*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+x1**2+x1*(-2+\
4*x3))*(-2+x6)*x6-4*x2*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+x1**2+x1*(\
-2+4*x3))*(-2+x5)*(-2+x6)*x6-8*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+\
x1**2+x1*(-2+4*x3))*(-2+x5)*x5*(-2+x6)*x6-x2*(x1+x2)**2*(1+x1*x2)**2*(\
-1+2*x3)*(1+x1**2+x1*(-2+4*x3))*(-2+x5)*x5*(-2+x6)*x6*(16*eulerGamma-\
24*np.log(2)-4*np.log(1-x3)-4*np.log(x3)-4*np.log(1-x4)-4*np.log(x4)+\
np.log(np.abs(FF1(x2,0,1,0,1,x0,x3,x4)))+np.log(np.abs(FF1(x2,0,1,1,0,x0,x3,x4)))+\
np.log(np.abs(FF1(x2,1,0,0,1,x0,x3,x4)))+np.log(np.abs(FF1(x2,1,0,1,0,x0,x3,x4)))+\
np.log(np.abs(FF1binv(x2,0,1,0,1,x0,x3,x4)))+np.log(np.abs(FF1binv(x2,0,1,1,0,x0,x3,x4)))+\
np.log(np.abs(FF1binv(x2,1,0,0,1,x0,x3,x4)))+np.log(np.abs(FF1binv(x2,1,0,1,0,x0,x3,x4)))+\
np.log(np.abs(FF2(x2,0,1,0,1,x0,x3,x4)))+np.log(np.abs(FF2(x2,0,1,1,0,x0,x3,x4)))+\
np.log(np.abs(FF2(x2,1,0,0,1,x0,x3,x4)))+np.log(np.abs(FF2(x2,1,0,1,0,x0,x3,x4)))+\
np.log(np.abs(FF2binv(x2,0,1,0,1,x0,x3,x4)))+np.log(np.abs(FF2binv(x2,0,1,1,0,x0,x3,x4)))+\
np.log(np.abs(FF2binv(x2,1,0,0,1,x0,x3,x4)))+np.log(np.abs(FF2binv(x2,1,0,1,0,x0,x3,x4)))+\
(1j/2)*np.pi*np.sign(FF1(x2,0,1,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF1(x2,0,1,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF1(x2,1,0,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF1(x2,1,0,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF1binv(x2,0,1,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF1binv(x2,0,1,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF1binv(x2,1,0,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF1binv(x2,1,0,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF2(x2,0,1,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF2(x2,0,1,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF2(x2,1,0,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF2(x2,1,0,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF2binv(x2,0,1,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF2binv(x2,0,1,1,0,x0,x3,x4))+\
(1j/2)*np.pi*np.sign(FF2binv(x2,1,0,0,1,x0,x3,x4))+(1j/2)*np.pi*np.sign(FF2binv(x2,1,0,1,0,x0,x3,x4)))))/(256*np.pi**6*x2*(x1+x2)\
**2*(1+x1*x2)**2*np.sqrt(1-x3)*np.sqrt(x3)*(1+x1**2+x1*(-2+4*x3))**2*\
np.sqrt(1-x4)*np.sqrt(x4)*(-2+x5)*x5*(-2+x6)*x6*(x1**2*(-1+x7)-x7))
    
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

def integrate_for_x0(x0_val):
    """对于给定的x0值进行积分"""
    integ = vegas.Integrator([[0., 1.], [0., 1.], [0., 1.], 
                             [0., 1.], [0., 1.], [0., 1.], [0., 1.]], nproc=1)
    
    def integrand(x):
        return f_fixed_x0(x, x0_val)
    
    # 训练积分器
    integ(integrand, nitn=10, neval=1e6)
    
    # 进行最终积分
    result = integ(integrand, nitn=40, neval=2e7)
    
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
            np.savez('integral_vs_x0_'+part+'_r-'+str(format(r, '.1e'))+'.npz', 
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
    plt.savefig('integral_vs_x0_'+part+'_r-'+str(format(r, '.1e'))+'.png')
    plt.show()

def main():
    # seed the random number generator so results reproducible
    gv.ranseed((1, 2, 33))

    start_time = time.time()  # 记录开始时间
    
    # 创建积分器并设置详细输出
    integ = vegas.Integrator(
        [[0., 2*np.pi], [0., 1.], [0., 1.], [0., 1.],[0., 1.], [0., 1.], [0., 1.], [0., 1.]], 
        nproc=13,
        analyzer=vegas.reporter()
    )

    # 适应性迭代阶段
    print("\n=== 开始适应性迭代阶段 ===")
    training_result = integ(f, nitn=10, neval=1e7, analyzer=vegas.reporter())

    # 最终积分阶段
    print("\n=== 开始最终积分阶段 ===")
    result = integ(f, nitn=40, neval=5e7, analyzer=vegas.reporter())
    
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
