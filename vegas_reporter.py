import time
from datetime import datetime, timedelta

class DetailedReporter:
    def __init__(self):
        self.iteration = 0
        self.total_iterations = None
        self.start_time = None
        self.last_time = None
        self.iteration_times = []
        
    def begin(self, itn, integrator):
        """每次迭代开始时调用"""
        if self.start_time is None:
            # 第一次迭代开始
            self.start_time = time.time()
            self.total_iterations = integrator.nitn
            print("\n" + "="*80)
            print(f"积分计算开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总迭代次数: {self.total_iterations}")
            print(f"每次迭代评估点数: {integrator.neval}")
            print(f"并行进程数: {integrator.nproc}")
            print("="*80 + "\n")
        
        self.iteration += 1
        self.last_time = time.time()
        
        # 打印当前迭代信息
        print(f"\n--- 迭代 {self.iteration}/{self.total_iterations} 开始 ---")
        
    def end(self, itn_result, cumulative_result):
        """每次迭代结束时调用"""
        # 计算时间信息
        current_time = time.time()
        iteration_time = current_time - self.last_time
        self.iteration_times.append(iteration_time)
        
        # 计算平均迭代时间和预估剩余时间
        avg_iteration_time = sum(self.iteration_times) / len(self.iteration_times)
        remaining_iterations = self.total_iterations - self.iteration
        estimated_remaining_time = avg_iteration_time * remaining_iterations
        
        # 计算总进度
        progress = (self.iteration / self.total_iterations) * 100
        
        # 打印详细信息
        print("\n迭代详细信息:")
        print(f"{'='*40}")
        print(f"当前进度: {progress:.1f}% ({self.iteration}/{self.total_iterations})")
        print(f"本次迭代耗时: {iteration_time:.2f} 秒")
        print(f"平均迭代耗时: {avg_iteration_time:.2f} 秒")
        print(f"预计剩余时间: {timedelta(seconds=int(estimated_remaining_time))}")
        print(f"已运行时间: {timedelta(seconds=int(current_time - self.start_time))}")
        print(f"{'='*40}")
        
        print("\n计算结果:")
        print(f"{'='*40}")
        print(f"本次迭代结果: {itn_result}")
        print(f"累积结果: {cumulative_result}")
        print(f"相对误差: {abs(cumulative_result.sdev/cumulative_result.mean):.2%}")
        print(f"{'='*40}")
        
        print("\n统计信息:")
        print(f"{'='*40}")
        print(f"χ²/dof = {cumulative_result.chi2/cumulative_result.dof:.3f}")
        print(f"Q值 = {cumulative_result.Q:.3f}")
        print(f"自由度 = {cumulative_result.dof}")
        print(f"{'='*40}")
        
        # 如果是最后一次迭代，打印总结信息
        if self.iteration == self.total_iterations:
            total_time = current_time - self.start_time
            print("\n" + "="*80)
            print("积分计算完成!")
            print(f"总耗时: {timedelta(seconds=int(total_time))}")
            print(f"最终结果: {cumulative_result}")
            print(f"最终相对误差: {abs(cumulative_result.sdev/cumulative_result.mean):.2%}")
            print(f"最终 χ²/dof = {cumulative_result.chi2/cumulative_result.dof:.3f}")
            print(f"最终 Q值 = {cumulative_result.Q:.3f}")
            print("="*80 + "\n")