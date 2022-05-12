2022/5/1    星期天

天气 小雨转阴

# 问题

win10重装系统后再从本地的git连接GitHub遇到的网络超时与ssh.config相关bug

## git下载

- 借鉴出处：

  https://blog.csdn.net/pioneer573/article/details/123448072

  https://blog.csdn.net/yuyunbai0917/article/details/123453978

在浏览器输入git，进入官网下载，这里所选用的是目前最新版进行安装。

官网下载：[https://git-scm.com/downloads](https://git-scm.com/downloads)

![image-20220503145529511](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503145529511.png)

也可以进入下面这个页面自选版本

![image-20220503144917162](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503144917162.png)

在这里，你需要先查询所自己电脑是多少位操作系统的，回到桌面右击电脑，点击最下方一行“属性”进入设置页面，即可找到系统类型，然后根据对应的版本下载想要的git版本。

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503143104897.png" alt="image-20220503143104897" style="zoom:67%;" />

因为git官网是在国外下载的，建议科学上网，或者使用镜像下载。

下载完毕之后，根据电脑下载的默认路经，找到相应的git的.exe文件，双击运行

![image-20220503145439778](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503145439778.png)

进入以下这个页面

![请添加图片描述](https://img-blog.csdnimg.cn/94ebbd64e7f24148846ff35760698e89.png)

next进入下一个页面，这里的安装路径建议安装在其他盘，如D盘，当然如果你的系统盘C盘很大，你也可以安装在C盘。

![请添加图片描述](https://img-blog.csdnimg.cn/5d405ce0c4214e7480e7cbd18208d470.png)

进入下一个界面，这里默认大框框的都勾选，其他按个人喜好，选完后按next。

![image-20220503150452223](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503150452223.png)

下一个界面，为git文件命名，默认为Git，也可以更改，建议不改，容易忘记。next

![在这里插入图片描述](https://img-blog.csdnimg.cn/6414569159a044d1944bd0a1a023bbfa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

建议用vim编辑器，git默认的。next,

![在这里插入图片描述](https://img-blog.csdnimg.cn/4366a60da2564eb5b5fb929130ff7200.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

初始化新项目仓库的命名，默认为master，也可以自定义，默认为main，也可以更改。next

![在这里插入图片描述](https://img-blog.csdnimg.cn/fbdd2976af294092ba79eb8185229093.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)



第一种是仅从 Git Bash 使用 Git。这个的意思就是你只能通过 Git 安装后的 Git Bash 来使用 Git ，其他的什么命令提示符啊等第三方软件都不行。

第二种是从命令行以及第三方软件进行 Git。这个就是在第一种基础上进行第三方支持，你将能够从 Git Bash，命令提示符(cmd) 和 Windows PowerShell 以及可以从 Windows 系统环境变量中寻找 Git 的任何第三方软件中使用 Git。推荐使用这个。

第三种是从命令提示符使用 Git 和可选的 Unix 工具。选择这种将覆盖 Windows 工具，如 “ find 和 sort ”。只有在了解其含义后才使用此选项。一句话，适合比较懂的人折腾。

next

![在这里插入图片描述](https://img-blog.csdnimg.cn/3774bfaa37a947b9bf164a689eabdd2b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

ssh协议，这里选用默认就好（后面会有一个坑，整到吐去），next

![在这里插入图片描述](https://img-blog.csdnimg.cn/692dd96787bf4dcba95f294ded89b8c7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

选择HTTPS后端传输，作为普通用户，只是用 Git 来访问 Github、GitLab 等网站，选择前者就行了。next

![在这里插入图片描述](https://img-blog.csdnimg.cn/908d38eefeaa4cc790ee45d94901fb09.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

配置行尾符号转换，翻译后，按需求选择。

![在这里插入图片描述](https://img-blog.csdnimg.cn/7c96a96d4ecb4728acd9fb88aadd4cb2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

配置终端模拟器与git bash 一起使用，建议Windows选用第一种 next

![在这里插入图片描述](https://img-blog.csdnimg.cn/b0c07f97dc0f4eada2a38d2aa2b4c1ee.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

采用默认，git pull 就是获取最新的远程仓库分支到本地，并与本地分支合并

上面给了三个 “git pull” 的行为：
第一个是 `merge`
第二个是 `rebase`
第三个是 `直接获取`

next

![在这里插入图片描述](https://img-blog.csdnimg.cn/9a59b42b115348458f2d1af0320c5ca1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

第一个选项是提供`登录凭证`帮助的，Git 有时需要用户的凭据才能执行操作；例如，可能需要输入`用户名`和`密码`才能通过 HTTP 访问远程存储库（GitHub，GItLab 等等），默认就好，next

![在这里插入图片描述](https://img-blog.csdnimg.cn/678f9348bff44f3c80979009d4d79fba.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

启用文件系统缓存就是将批量读取文件系统数据并将其缓存在内存中以进行某些操作，可以显著提升性能。这个选项默认开启。
启用符号链接 ，符号链接是一类特殊的文件， 其包含有一条以绝对路径或者相对路径的形式指向其它文件或者目录的引用，类似于 Windows 的快捷方式，不完全等同 类Unix（如 Linux） 下的 符号链接。因为该功能的支持需要一些条件，所以默认不开启。next

![在这里插入图片描述](https://img-blog.csdnimg.cn/26ad8c02f89c4f6599afd6284633beca.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

建议都不勾选，next

![在这里插入图片描述](https://img-blog.csdnimg.cn/8a16a426ad794d119aa9d57132395985.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

进程显示

完成

![在这里插入图片描述](https://img-blog.csdnimg.cn/48f67e0230054f2ba6bdf85fc9ea6db4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbXVrZXM=,size_20,color_FFFFFF,t_70,g_se,x_16)

见到一张合适的图，拿来用用（默认你已经有git的账户，可以git clone别人的仓库了）

![image-20220503154101085](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503154101085.png)

可能不同时期下载会有差异，但基本上是这样。



## 重点bug区域

重装系统之后，我已经装好git并且成功把自己GitHub的仓库git clone下来了，当我更新我的仓库是，即我想要更新我的笔记时，他告诉我，我的ssh有问题，没连接上，然后我就尝试了各种方法，重复新弄了一个ssh，结果bug的背后有一个bug，现在就卡在ssh: connect to host github.com port 22: Connection timed out
fatal: Could not read from remote repository. 的端口问题上，吐了，555~

![image-20220503155850723](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220503155850723.png)



解决问题，借鉴出处

https://blog.csdn.net/nightwishh/article/details/99647545

就是说，出现下面这个问题是因为，C盘我的用户名之下的.ssh文件没有config文件，端口出现问题，只需要设置好配置文件的端口就好。



一、首先，不管是git push 还是git clone，报错信息都是：ssh: connect to host github.com port 22: Connection timed out

找到原因：原本是port443: 的但是他一直连的是port2: 

![image-20220504154734780](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220504154734780.png)



二、先检查一下ssh是否可以连接成功，即右击鼠标，点git bash进入git命令行，输入以下命令

ssh -T git@github.com

如果还是出现这个报错的话，可以使用这个方法来解决你的问题。



三、接下来，根据点击桌面的电脑来到 电脑--》C盘--》用户--》你的用户名  的目录之下找到 .ssh文件夹看看有没有箭头指向的config文件，（如果你没有ssh文件的话可以根据网上的教程配置，如果有的话，请往下看）

![image-20220504192113929](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220504192113929.png)

如果没有config的话，直接添加config，注意是由txt文件重命名的，没有后缀。添加完以后，应该如图示页面，即上图所示。



四、以记事本的方式打开，添加如下信息

Host github.com 

User 注册github的邮箱 

Hostname ssh.github.com 

PreferredAuthentications publickey 

IdentityFile ~/.ssh/id_rsa   # 注意这里 ~ 是Linux终端打开的命令行，Windows改为你.ssh的路径

Port 443

再windows直接在config里添加，如图所示

![image-20220504225122848](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220504225122848.png)



五、在命令符这里git push就可以了

![image-20220504225701023](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220504225701023.png)

然后你可能会发现 

git clone https://github.com/bubbliiiing/yolo3-pytorch可能会有问题，无法下载，改用ssh的来下载。

![image-20220505165522244](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220505165522244.png)



将路径复制过去以后，直接回车下载。

![image-20220505164530815](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220505164530815.png)

成功下载

![image-20220505165636910](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220505165636910.png)
