# 项目环境配置

## 一、给.condarc添加清华源 

![image-20220416230807629](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416230807629.png)-

- #### 选用清华源比较好

- <img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416230024781.png" alt="image-20220416230024781" style="zoom: 67%;" />- 







## 二、conda环境变量，library里没有usr文件夹的处理

- ![image-20220416225847106](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416225847106.png)







## 三、conda激活环境

- ### conda activate + name  激活不了，改用activate + name直接激活

- <img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416230659592.png" alt="image-20220416230659592" style="zoom: 80%;" />





## 四、解压cudnn

- #### 用winRAR解压cudnn软件，解压到对应路径文件夹

- ![image-20220416231142595](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416231142595.png)



- #### 判断cudnn安装成功，两个测试exe程序 

- ![image-20220416231346423](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416231346423.png

- <img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416231426505.png" alt="image-20220416231426505" style="zoom:67%;" />

- ![image-20220416231448998](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416231448998.png)



### 重点踩坑区！！！

CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.3



## 五、项目运行init文件

-  ![image-20220416231727605](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416231727605.png)

- 出现此报错，检查是否有项目py文件被改动，不符合要求，将改动的文件改回来



- 项目yolo子文件
- ![image-20220416232149957](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416232149957.png)



- 确定蓝色文件夹的位置如下方蓝色文件夹的位置

- ![image-20220416232227346](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416232227346.png)

- 这样才不会影响代码运行



- 项目的环境需按照requirements配置
- ![image-20220416232539645](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416232539645.png)

代码是通用的，只要有相应的包，基本上都可以跑起来。 