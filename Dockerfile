From ubuntu:20.04

Run apt update
Run apt install python3-pip -y

COPY . .
Run pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD python3 main.py
EXPOSE 10037