# EDA_competition

## Original Level Set Implementation
```
LevelSetMethod构造函数
    |
    +--> generateGrid()
         
loadMesh()
    |
    +--> 创建AABB树

evolve()
    |
    +--> initializeSignedDistanceField()
    |    |
    |    +--> 使用tree检查点是否在网格内部
    |
    +--> isOnBoundary()
    |
    +--> getIndex()
    |
    +--> reinitialize()
         |
         +--> isOnBoundary()
         |
         +--> getIndex()

saveResult()
    |
    +--> 将结果写入文件

extractSurfaceMeshCGAL()
    |
    +--> 创建LevelSetImplicitFunction
         |
         +--> 使用phi和grid数据
         |
         +--> getIndex()（间接使用） 
```