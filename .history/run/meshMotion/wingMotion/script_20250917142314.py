from paraview.simple import *

# 读取 OpenFOAM 文件
reader = OpenDataFile("/home/jin/OpenFOAM/openfoam-smartsim/run/meshMotion/wingMotion/mesh-motion_Pinn/of_model/pinn.foam")

# 显示数据
view = GetActiveViewOrCreate("RenderView")
display = Show(reader, view)

# 渲染一张图片
Render(view)
SaveScreenshot("cavity.png", view, ImageResolution=[1920, 1080])