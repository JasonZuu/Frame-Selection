# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('./index.html')
# 数据上传处理
@app.route('/upload',methods=['POST'])
@cross_origin()  # 跨域注解
def UpGeojsonFile():
    print("11")
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        # print(filename)
        # 验证文件格式
        types = ['mp4']
        if filename.split('.')[-1] in types:
            # 获取上传文件的服务器存储地址
            file_path = os.path.join('./upload/video','{0}'.format(filename))
            # print(file_path)
            # 文件存储
            f.save(file_path)
        outinfo = filename
        return outinfo
    else:
        return 'error'
