import time

'''
2022/11/26 Added print out function

Developed by Peter
This code snippet is aimed at measure the running time of given 
code block, classes, functions and methods. 
'''
class Delta_timer:
  def __init__(self):
    self.start_time = 0
    self.stop_time = 0
    self.timer_counter = 0
    self.delta_time = 0
    self.delta_sum = 0
    self.avg_time = 0
    self.avg_FPS = 0

  def start(self):
    self.start_time = time.time()

  def update(self):
    self.timer_counter += 1
    self.delta_time = self.stop_time - self.start_time
    self.delta_sum += self.delta_time

  def stop(self):
    self.stop_time = time.time()

  def report(self, text):
    self.avg_time = self.delta_sum / self.timer_counter
    self.avg_FPS = self.timer_counter / self.delta_sum

    print(f"{text} report: ")
    print(f"[report] time elapsed: {self.delta_sum:.4f}")
    print(f"[report] total counters: {self.timer_counter} ")
    print(f"[report] avg time: {self.avg_time:.4f}")
    print(f"[report] avg FPS: {self.avg_FPS:.4f}")
    print("************************")
