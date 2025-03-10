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

r = 1e8
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
    
    result = 1024*np.pi**6 * np.cos(1*x0)*((1-x1**2)*(-8*x2*(x1+x2)**2*(1+x1*x2)**2*(-1+2*x3)*(1+x1**2+x1*(-2+\
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
    integ(integrand, nitn=10, neval=1e7)
    
    # 进行最终积分
    result = integ(integrand, nitn=50, neval=1e8)
    
    return result.mean, result.sdev

def process_x0(x0):
    """处理单个x0值的函数"""
    mean, sdev = integrate_for_x0(x0)
    return mean, sdev

def plot_x0_distribution():
    import matplotlib.pyplot as plt
    from multiprocessing import Pool, cpu_count
    
    # 生成x0的值
    x0_values = np.linspace(0, 2*np.pi, 100)
    
    # 创建进程池
    n_cores = 100  # 留一个核心给系统
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

def main(seed=None):
    # 如果提供了种子，则使用它
    if seed is not None:
        gv.ranseed(seed)
    else:
        # 否则使用默认种子
        gv.ranseed((1, 2, 33))

    start_time = time.time()  # 记录开始时间
    
    # 创建积分器并设置详细输出
    # 使用较少的核心，因为我们会在外层并行
    integ = vegas.Integrator(
        [[0., 2*np.pi], [0., 1.], [0., 1.], [0., 1.],[0., 1.], [0., 1.], [0., 1.], [0., 1.]], 
        nproc=1,  # 使用单核，因为我们在外层并行
    )
    
    # 适应性迭代阶段 - 减少迭代次数，因为我们会有多次独立运行
    print(f"\n=== 开始适应性迭代阶段 (seed={seed}) ===")
    training_result = integ(f, nitn=10, neval=1e6)

    # 最终积分阶段 - 减少迭代次数，因为我们会有多次独立运行
    print(f"\n=== 开始最终积分阶段 (seed={seed}) ===")
    result = integ(f, nitn=100, neval=2e7)
    
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    
    # 输出当前进程的结果
    print(f"\n=== 种子 {seed} 的计算结果 ===")
    print(result.summary())
    print(f'积分结果 = {result}')
    print(f'计算时间 = {total_time:.2f}秒')
    
    # 返回结果以便进行合并
    return result.mean, result.sdev, total_time

def run_parallel_integration(n_cores):
    """
    并行运行多个积分实例，然后合并结果
    
    使用权重平均法合并多个独立积分结果：
    - 权重正比于各自方差的倒数
    - 最终误差是独立积分误差的合理组合
    """
    from multiprocessing import Pool
    import numpy as np
    
    overall_start = time.time()
    
    print(f"=== 启动并行积分，使用 {n_cores} 个核心 ===")
    
    # 创建不同的随机种子
    seeds = [(i, i+100, i+200) for i in range(1, n_cores+1)]
    
    # 创建进程池并运行
    with Pool(n_cores) as pool:
        # 启动所有进程
        results = pool.map(main, seeds)
    
    # 解包结果
    means, sdevs, times = zip(*results)
    
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
    
    # 输出结果
    print("\n=== 并行积分结果 ===")
    print(f"总核心数: {n_cores}")
    print(f"积分结果: {final_mean} ± {final_sdev}")
    print(f"相对误差: {100 * final_sdev / abs(final_mean):.2f}%")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"最长单次运行时间: {max_time:.2f}秒")
    print(f"加速比: {max_time/total_time:.2f}x (理想情况下)")
    
    # 保存单次结果到文件
    np.savez(f'parallel_result_{part}_r-{format(r, ".1e")}_cores-{n_cores}.npz',
             mean=final_mean,
             sdev=final_sdev,
             weights=weights,
             individual_means=means,
             individual_sdevs=sdevs,
             individual_times=times,
             total_time=total_time,
             n_cores=n_cores)
    
    return final_mean, final_sdev, total_time

def run_multiple_integrations(n_cores, n_iterations):
    """
    运行多次并行积分并整合所有结果
    
    参数:
    n_cores: 每次积分使用的核心数
    n_iterations: 运行并行积分的次数
    """
    import numpy as np
    
    overall_start = time.time()
    
    print(f"=== 开始运行 {n_iterations} 次并行积分 ===")
    
    all_means = []
    all_sdevs = []
    all_times = []
    
    # 运行n_iterations次并行积分
    for i in range(1, n_iterations+1):
        print(f"\n=== 进行第 {i}/{n_iterations} 次并行积分 ===")
        mean, sdev, run_time = run_parallel_integration(n_cores)
        all_means.append(mean)
        all_sdevs.append(sdev)
        all_times.append(run_time)
    
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
    
    # 计算Q值 (chi^2/dof)
    chi2 = np.sum([(m - final_mean)**2 / s**2 for m, s in zip(all_means, all_sdevs)])
    dof = len(all_means) - 1  # 自由度
    Q = chi2 / dof if dof > 0 else 0
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(f"总运行次数: {n_iterations}")
    print(f"每次运行核心数: {n_cores}")
    print(f"积分结果 = {final_mean} ± {final_sdev}")
    print(f"Q值 = {Q:.2f}")
    print(f"相对误差 = {100 * final_sdev / abs(final_mean):.2f}%")
    print(f"总计算时间 = {total_time:.2f}秒")
    
    # 保存结果到文件
    np.savez(f'final_result_{part}_r-{format(r, ".1e")}_cores-{n_cores}_iters-{n_iterations}.npz',
             mean=final_mean,
             sdev=final_sdev,
             Q=Q,
             individual_means=all_means,
             individual_sdevs=all_sdevs,
             individual_times=all_times,
             total_time=total_time,
             n_cores=n_cores,
             n_iterations=n_iterations)
    
    return final_mean, final_sdev, Q, total_time

def main2():    
    # 使用多次并行积分
    result, error, Q, total_time = run_multiple_integrations(n_cores=110, n_iterations=100)
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(f'积分结果 = {result} ± {error}')
    print(f'Q值 = {Q:.2f}')
    print(f'相对误差 = {100 * error / abs(result):.2f}%')
    print(f'总计算时间 = {total_time:.2f}秒')  # 使用我们自己计算的时间
    
    # 保存结果到文件
    np.savez('final_result_'+part+'_r-'+str(format(r, '.1e'))+'.npz',
             mean=result,
             sdev=error,
             Q=Q,
             total_time=total_time)
#    if SHOW_GRID:
#        main.integ.map.show_grid(20)


if __name__ == '__main__':
    main2()



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
