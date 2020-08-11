# 拉取官方镜像
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3  
# 指定一个工作目录，这里指定/competition为工作目录
# 此后的RUN命令，CMD命令，都是在工作目录下进行的
WORKDIR  /competition
# 将当前目录下的requirements.txt拷贝到工作目录下
ADD requirements.txt /competition
# 按requirements.txt的要求，把相应的依赖安装到容器中
# 使用国内源以提高速度
RUN pip --no-cache-dir  install -r requirements.txt -i https://pypi.douban.com/simple

RUN sudo chmod 777 -R /competition

# 把文件夹下剩余的内容拷贝至工作目录下
ADD . /competition

RUN sudo chmod 777 -R /competition/hf_round1_beat_500

RUN sudo chmod 777 -R /competition/hf_round2_beat_500

RUN sudo chmod 777 -R /competition/hf_round2_test_beat_500

RUN sudo chmod 777 -R /competition/ckpt

RUN sudo chmod 777 -R /competition/val_cv
# 容器启动时运行run.sh命令
CMD ["sh", "run.sh"]
