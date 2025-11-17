from paraview.simple import *

# 读取 OpenFOAM 文件
reader = OpenDataFile("/home/jin/case/cavity.foam")

# 显示数据
view = GetActiveViewOrCreate("RenderView")
display = Show(reader, view)

# 渲染一张图片
Render(view)
WriteImage("cavity.png")