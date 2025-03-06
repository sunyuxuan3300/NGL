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
    
    F1m = 1-2*x[1]-4*np.sqrt(1-x[1])*np.sqrt(x[1])*np.sqrt(1-x[2])*np.sqrt(x[2])-2*x[2]+4*x[1]*x[2]
    F1p = 1-2*x[1]+4*np.sqrt(1-x[1])*np.sqrt(x[1])*np.sqrt(1-x[2])*np.sqrt(x[2])-2*x[2]+4*x[1]*x[2]
    F2p = np.sqrt(1-x[1])*np.sqrt(x[1])*(1-2*x[2])+np.sqrt(1-x[2])*np.sqrt(x[2])-2*x[1]*np.sqrt(1-x[2])*np.sqrt(x[2])
    F2m = np.sqrt(1-x[1])*np.sqrt(x[1])*(1-2*x[2])-np.sqrt(1-x[2])*np.sqrt(x[2])+2*x[1]*np.sqrt(1-x[2])*np.sqrt(x[2])
    F3p = np.sqrt(1-x[2])*np.sqrt(x[2])-2*x[1]*np.sqrt(1-x[2])*np.sqrt(x[2])+np.sqrt(1-x[1])*np.sqrt(x[1])*(-1+2*x[2])
    F3m = -np.sqrt(1-x[2])*np.sqrt(x[2])+2*x[1]*np.sqrt(1-x[2])*np.sqrt(x[2])+np.sqrt(1-x[1])*np.sqrt(x[1])*(-1+2*x[2])
    
    T1 = 8-2*x[4]-2*x[5]+16*np.log(2)-8*x[4]*np.log(2)-8*x[5]*np.log(2)+x[4]*x[5]*np.log(16)
    T2 = 2*(-2+x[4])*(-2+x[5])*np.log(1-x[1])+2*(-2+x[4])*(-2+x[5])*np.log(x[1])+8*np.log(1-x[2])-4*x[4]*np.log(1-x[2])-4*x[5]*np.log(1-x[2])+2*x[4]*x[5]*np.log(1-x[2])
    T3 = 8*np.log(x[2])-4*x[4]*np.log(x[2])-4*x[5]*np.log(x[2])+2*x[4]*x[5]*np.log(x[2])-16*np.log(np.abs(1-2*x[2]))+8*x[4]*np.log(np.abs(1-2*x[2]))+8*x[5]*np.log(np.abs(1-2*x[2]))-4*x[4]*x[5]*np.log(np.abs(1-2*x[2]))
    
    T4 = -4*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) + 2*x[4]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) + 2*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) - x[4]*x[5]*np.log(np.abs(F1p*np.cos(x[0]) + 2*F2m*np.sin(x[0]))) - 4*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))) + 2*x[4]*np.log(np.abs(F1m*np.cos(x[0]) + 2*F2p*np.sin(x[0]))) + 2*x[5]*np.log(np.abs(F1m*np.cos(x[0])+2*F2p*np.sin(x[0])))-x[4]*x[5]*np.log(np.abs(F1m*np.cos(x[0])+2*F2p*np.sin(x[0])))- 4 *np.log(np.abs(F1m *np.cos(x[0]) + 2 *F3m *np.sin(x[0]))) + 2 *x[4] *np.log(np.abs(F1m *np.cos(x[0]) + 2 *F3m *np.sin(x[0]))) + 2* x[5]* np.log(np.abs(F1m *np.cos(x[0]) + 2 *F3m *np.sin(x[0]))) - x[4]* x[5]* np.log(np.abs(F1m *np.cos(x[0]) + 2* F3m *np.sin(x[0]))) - 4 *np.log(np.abs(F1p *np.cos(x[0]) + 2* F3p *np.sin(x[0]))) + 2* x[4] *np.log(np.abs(F1p *np.cos(x[0]) + 2* F3p *np.sin(x[0]))) + 2 *x[5] *np.log(np.abs(F1p *np.cos(x[0]) + 2 *F3p *np.sin(x[0]))) - x[4] *x[5] *np.log(np.abs(F1p* np.cos(x[0]) + 2* F3p *np.sin(x[0])))
    
    nm = 4*(-1+2*x[1])*np.cos(2*x[0])*(T1+T2+T3+T4)
    de = np.pi**3 * np.sqrt(1-x[1]) * np.sqrt(x[1]) * np.sqrt(1-x[2]) * np.sqrt(x[2]) * np.sqrt(2-x[3]) * np.sqrt(x[3]) * (-2+x[4]) * (-2+x[5]) * np.sqrt(x[6]) * (2-x[3]+2*(-1+2*x[1])*np.sqrt(2-x[3])*np.sqrt(x[3])*np.sqrt(x[6])+x[3]*x[6])
    
    return nm/de













def main():
    # seed the random number generator so results reproducible
    gv.ranseed((1, 2, 33))

    # 创建积分器并设置详细输出
    integ = vegas.Integrator(
        [[0., 2*pi], [0., 1.], [0., 1.], [0., 1.],[0., 1.], [0., 1.], [0., 1.]], 
        nproc=13,
        verbose=True  # 启用详细输出
    )

    # 适应性迭代阶段
    print("\n=== 开始适应性迭代阶段 ===")
    training_result = integ(f, nitn=40, neval=1e7, analyzer=vegas.reporter())

    # 最终积分阶段
    print("\n=== 开始最终积分阶段 ===")
    result = integ(f, nitn=500, adapt=False, neval=1e7, analyzer=vegas.reporter())
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(result.summary())
    print('积分结果 = %s' % result)
    print('Q值 = %.2f' % result.Q)
    print('相对误差 = %.2f%%' % (100 * result.sdev / abs(result.mean)))
    print('总计算时间 = %.2f秒' % result.time)
    
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
