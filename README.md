# WeChaty-Torch
[![Powered by Wechaty](https://img.shields.io/badge/Powered%20By-Wechaty-green.svg)](https://github.com/chatie/wechaty)
[![Wechaty开源激励计划](https://img.shields.io/badge/Wechaty-开源激励计划-green.svg)](https://github.com/juzibot/Welcome/wiki/Everything-about-Wechaty)

本项目使用wechaty和pytorch搭建一个可以通过微信个人号与深度学习模型交互的平台。

## 原理

+ 用户在微信发送图片-->
+ -->wechaty收到图片并进行base64后post请求到后端-->
+ -->使用fastapi开发的后端收到图片的base64编码后调用模型-->
+ -->模型给与预测与置信度表传给后端-->
+ -->后端收到后向wechaty响应-->
+ -->wechaty收到数据后发送给用户.

## 文件结构

+ ```wechaty-torch.ts``` typescript文件，使用wechaty与微信通讯；
+ ```main.py``` 后端文件，基于fastapi开发，中转图片数据；
+ ```model.py``` 模型调用文件，给出预测和置信度；
+ ```model.pth``` 模型文件（二进制），使用WideResNet在CIFAR-10数据集上进行训练，测试集准确率91.22%.

## 依赖库

typescript：请按照wechaty文档安装.

python：fastapi,uvicorn,torch,numpy,PIL

## 快速开始

> 请确保您已将所有依赖环境安装成功

1. 在```wechaty-torch.ts```文件的```const token = 'YOUR_TOKEN_HERE'```处填入您的token（获取方式见wechaty文档）；
2. 在```model.py```文件的```os.chdir("Your PATH")```处修改为您的文件路径；
3. 运行```main.py```后运行```wechaty-torch.ts```.