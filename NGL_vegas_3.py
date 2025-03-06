"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation (slightly modified).
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas

import sys



import numpy as np
import gvar as gv 
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime

from numpy import log
from numpy import abs
from numpy import sin
from numpy import cos
from numpy import pi

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True



def f(x):
    # 检查输入变量是否在正确的范围内
    if not (0 <= x[0] <= 2*pi and 
            all(0 <= x[i] <= 1 for i in range(1, 7))):
        print("警告：输入变量超出范围！")
        print(f"x值: {x}")
        return 0
    
    # 变量变换和区间检查
    theta1 = pi/2 * x[1]  # 映射到[0,π/2]
    theta2 = pi/2 * x[2]
    theta3 = pi/2 * x[3]
    theta6 = pi/2 * x[6]
    
    # 变换后的变量
    x1 = np.sin(theta1)**2  # 范围[0,1]
    x2 = np.sin(theta2)**2
    x3 = np.sin(theta3)**2
    x6 = np.sin(theta6)**2
    
    # 检查变换后的变量是否在正确范围内
    if not (0 <= x1 <= 1 and 0 <= x2 <= 1 and 
            0 <= x3 <= 1 and 0 <= x6 <= 1):
        print("警告：变换后的变量超出范围！")
        print(f"变换后的值: x1={x1}, x2={x2}, x3={x3}, x6={x6}")
        return 0
    
    # 雅克比行列式
    jacobian = (pi/2)**4 * np.sin(2*theta1) * np.sin(2*theta2) * \
               np.sin(2*theta3) * np.sin(2*theta6) 
    
    # 使用变换后的变量计算F1m,F1p等
    F1m = (1 - 2*x1 - 4*np.sin(theta1)*np.cos(theta1)*np.sin(theta2)*np.cos(theta2) - 
           2*x2 + 4*x1*x2)
    F1p = (1 - 2*x1 + 4*np.sin(theta1)*np.cos(theta1)*np.sin(theta2)*np.cos(theta2) - 
           2*x2 + 4*x1*x2)
    F2p = (np.sin(theta1)*np.cos(theta1)*(1-2*x2) + 
           np.sin(theta2)*np.cos(theta2) - 
           2*x1*np.sin(theta2)*np.cos(theta2))
    F2m = (np.sin(theta1)*np.cos(theta1)*(1-2*x2) - 
           np.sin(theta2)*np.cos(theta2) + 
           2*x1*np.sin(theta2)*np.cos(theta2))
    F3p = (np.sin(theta2)*np.cos(theta2) - 
           2*x1*np.sin(theta2)*np.cos(theta2) + 
           np.sin(theta1)*np.cos(theta1)*(-1+2*x2))
    F3m = (-np.sin(theta2)*np.cos(theta2) + 
           2*x1*np.sin(theta2)*np.cos(theta2) + 
           np.sin(theta1)*np.cos(theta1)*(-1+2*x2))
    
    
    # T1项
    T1 = (8 - 2*x[4] - 2*x[5] + 
          16*np.log(2) - 8*x[4]*np.log(2) - 8*x[5]*np.log(2) + 
          x[4]*x[5]*np.log(16))
    
    # T2项
    T2 = (2*(-2+x[4])*(-2+x[5])*2*np.log(np.sin(theta1)) + 
          2*(-2+x[4])*(-2+x[5])*2*np.log(np.cos(theta1)) + 
          8*2*np.log(np.cos(theta2)) - 4*x[4]*2*np.log(np.cos(theta2)) - 
          4*x[5]*2*np.log(np.cos(theta2)) + 2*x[4]*x[5]*2*np.log(np.cos(theta2)))
    
    # T3项
    T3 = (8*2*np.log(np.sin(theta2)) - 4*x[4]*2*np.log(np.sin(theta2)) - 4*x[5]*2*np.log(np.sin(theta2)) + 
          2*x[4]*x[5]*2*np.log(np.sin(theta2)) - 
          16*np.log(np.abs(1-2*x2)) + 
          8*x[4]*np.log(np.abs(1-2*x2)) + 
          8*x[5]*np.log(np.abs(1-2*x2)) - 
          4*x[4]*x[5]*np.log(np.abs(1-2*x2)))
    
    # T4项 (分成多个部分以提高可读性)
    T4_part1 = (-4*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) + 
                2*x[4]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) + 
                2*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) - 
                x[4]*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))))
    
    T4_part2 = (-4*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))) + 
                2*x[4]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))) + 
                2*x[5]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))) - 
                x[4]*x[5]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))))
    
    T4_part3 = (-4*np.log(np.abs(F1m*np.cos(x[0]) + 2*F3m*np.sin(x[0]))) + 
                2*x[4]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F3m*np.sin(x[0]))) + 
                2*x[5]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F3m*np.sin(x[0]))) - 
                x[4]*x[5]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F3m*np.sin(x[0]))))
    
    T4_part4 = (-4*np.log(np.abs(F1p*np.cos(x[0]) + 2*F3p*np.sin(x[0]))) + 
                2*x[4]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F3p*np.sin(x[0]))) + 
                2*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F3p*np.sin(x[0]))) - 
                x[4]*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F3p*np.sin(x[0]))))
    
    T4 = T4_part1 + T4_part2 + T4_part3 + T4_part4
    
    # 计算最终结果
    nm = 4*(-1+2*x1)*np.cos(2*x[0])*(T1 + T2 + T3 + T4)
    de = np.pi**3 * np.sin(theta1)*np.cos(theta1) * np.sin(theta2)*np.cos(theta2) * np.sqrt(2-x3) * np.sin(theta3) * (-2+x[4]) * (-2+x[5]) * np.sin(theta6) * (2-x3+2*(-1+2*x1)*np.sqrt(2-x3)*np.sin(theta3)*np.sin(theta6)+x3*x6)
    
    # 返回前检查结果是否为有限值
    result = (nm/de) * jacobian
    if not np.isfinite(result):
        print("警告：计算结果为无穷大或NaN！")
        print(f"nm={nm}, de={de}, jacobian={jacobian}")
        return 0
        
    return result


class ProgressReporter(object):
    def __init__(self, total_iters):
        self.start_time = time.time()
        self.total_iters = total_iters
        self.previous_result = None
    
    def begin(self):
        """在积分开始时调用"""
        pass
    
    def end(self):
        """在积分结束时调用"""
        pass
    
    def __call__(self, results):
        """每次迭代后调用"""
        iter_num = len(results)
        if iter_num > 0:
            last_result = results[-1]
            elapsed_time = time.time() - self.start_time
            
            # 清晰的进度分隔线
            print("\n" + "="*50)
            print(f"进度: 迭代 {iter_num}/{self.total_iters}")
            print(f"完成度: {iter_num/self.total_iters*100:.1f}%")
            print(f"预计剩余时间: {(elapsed_time/iter_num)*(self.total_iters-iter_num)/60:.1f}分钟")
            print("-"*30)
            
            print(f"当前结果: {last_result.mean:.8f} ± {last_result.sdev:.8f}")
            print(f"相对误差: {100 * last_result.sdev / abs(last_result.mean):.4f}%")
            print(f"Q值: {last_result.Q:.4f}")
            print(f"已用时间: {elapsed_time/60:.2f}分钟")
            
            if len(results) >= 2:
                prev_result = results[-2]
                rel_change = abs((last_result.mean - prev_result.mean)/prev_result.mean)
                print(f"相对变化: {rel_change:.2e}")
                
                # 收敛性警告
                if rel_change > 0.01:  # 相对变化超过1%
                    print("\033[93m警告：结果可能尚未收敛\033[0m")
                if last_result.Q < 0.1:
                    print("\033[93m警告：Q值过低，可能需要增加迭代次数\033[0m")

def main():
    # seed the random number generator
    gv.ranseed((1, 2, 33))
    
    print("\n=== 积分初始化 ===")
    print("积分维度: 7")
    print("积分区间:")
    print("x[0]: [0, 2π]")
    print("x[1]-x[6]: [0, 1]")
    print("变量变换说明:")
    print("x[1,2,3,6] -> sin²(πx/2) 变换")
    
    start_time = time.time()
    
    # 创建积分器
    integ = vegas.Integrator(
        [[0., 2*pi], [0., 1.], [0., 1.], [0., 1.],
         [0., 1.], [0., 1.], [0., 1.]], 
        nproc=13
    )
    
    # 适应性迭代阶段
    print("\n=== 开始适应性迭代阶段 ===")
    print(f"迭代次数: 50, 每次采样点数: {5e6:.1e}")
    training_result = integ(f, nitn=50, neval=5e6, 
                          analyzer=vegas.reporter())
    
    print("\n=== 训练阶段完成 ===")
    print(f"训练结果: {training_result.mean:.8f} ± {training_result.sdev:.8f}")
    print(f"训练阶段Q值: {training_result.Q:.4f}")
    
    # 最终积分阶段
    print("\n=== 开始最终积分阶段 ===")
    print(f"迭代次数: 200, 每次采样点数: {2e7:.1e}")
    result = integ(f, nitn=200, neval=2e7, adapt=False,
                  analyzer=vegas.reporter())
    
    # 详细的最终结果报告
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("=== 最终计算报告 ===")
    print("="*50)
    print(result.summary())
    print(f"积分结果 = {result.mean:.8f} ± {result.sdev:.8f}")
    print(f"Q值 = {result.Q:.4f}")
    print(f"相对误差 = {100 * result.sdev / abs(result.mean):.4f}%")
    print(f"总计算时间 = {total_time/60:.2f}分钟")
    print("="*50)
    
    # 保存结果
    save_results(result, total_time)
    

def save_results(result, total_time):
    """保存计算结果到文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'integration_results_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("="*50 + "\n")
        f.write("=== 积分计算结果 ===\n")
        f.write("="*50 + "\n")
        f.write(f"计算时间: {total_time/60:.2f}分钟\n")
        f.write(f"积分结果: {result.mean:.8f} ± {result.sdev:.8f}\n")
        f.write(f"相对误差: {100 * result.sdev / abs(result.mean):.4f}%\n")
        f.write(f"Q值: {result.Q:.4f}\n")
        f.write("\n详细信息:\n")
        f.write(result.summary())
        f.write("\n" + "="*50 + "\n")
    
    print(f"\n结果已保存到文件: {filename}")




# 确保输出到终端
sys.stdout.flush()

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
