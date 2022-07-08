# 项目环境配置


![image](https://user-images.githubusercontent.com/93062146/178000815-3e00c101-ff3e-4c3e-a923-059c22c0c6b8.png)


## 二、conda环境变量，library里没有usr文件夹的处理

- ![image](https://user-images.githubusercontent.com/93062146/178001079-bdc28b05-db74-40e2-bfed-feeb7272b998.png)



## 三、conda激活环境

- ### conda activate + name  激活不了，改用activate + name直接激活
- ![image](https://user-images.githubusercontent.com/93062146/178001357-42289c4d-1a92-427a-8ae6-6807031e93e1.png)






## 四、解压cudnn

- #### 用winRAR解压cudnn软件，解压到对应路径文件夹

-![image](https://user-images.githubusercontent.com/93062146/178001568-fee76a6a-c153-412b-808d-aa7dee26d1e3.png)



- #### 判断cudnn安装成功，两个测试exe程序 

- ![image](https://user-images.githubusercontent.com/93062146/178001783-2cd644f8-1620-46c3-8397-1b13a0d85aec.png)


-![image](https://user-images.githubusercontent.com/93062146/178001919-05f25f55-5264-4b2a-a409-0cc9004814b0.png)



### 重点踩坑区！！！

CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.3



## 五、项目运行init文件

-  ![image](https://user-images.githubusercontent.com/93062146/178002148-0f30162f-d362-4bb7-9777-51ca2457910d.png)


- 出现此报错，检查是否有项目py文件被改动，不符合要求，将改动的文件改回来



- 项目yolo子文件
- ![image](https://user-images.githubusercontent.com/93062146/178002314-b0666e4d-935d-451c-a565-3ce902cadda8.png)




- 确定蓝色文件夹的位置如下方蓝色文件夹的位置

- ![image](https://user-images.githubusercontent.com/93062146/178002476-eab8f2d8-3377-48b7-ac3c-e1db22aefa6b.png)


- 这样才不会影响代码运行



- 项目的环境需按照requirements配置
- ![image-20220416232539645](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220416232539645.png)

代码是通用的，只要有相应的包，基本上都可以跑起来。 
