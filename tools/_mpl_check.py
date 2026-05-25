import matplotlib
print("matplotlib version:", matplotlib.__version__)
import mpl_toolkits.mplot3d.axes3d as p3
import inspect
sig = inspect.signature(p3.Axes3D.view_init)
print("view_init signature:", sig)
print("  if vertical_axis param present, can fix Y-up.")
