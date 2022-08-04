from flask import Flask, request
import json
from unittest import TestCase
import requests
from requests.compat import urljoin

from abstract import FrameSelectionAbstract

__all__ = ["app", "TestAPI"]
app = Flask(__name__)
fs_abstract = FrameSelectionAbstract()


# upload
@app.route('/fs_backend/reset_help', methods=["POST"])
def reset_help():
    try:
        keys_info = fs_abstract.reset(help_mode=True)
        res = {"is_success": True,
               "message": '',
               "scorer_keys": keys_info["scorer_keys"],
               "critor_keys": keys_info["critor_keys"]}
    except Exception as e:
        res = {"is_success": False,
               "message": str(e),
               "scorer_keys": None,
               "critor_keys": None}
    return json.dumps(res)


@app.route('/fs_backend/reset', methods=["POST"])
def reset():
    video_path = request.form.get('video_path')
    scorer_key = request.form.get('scorer_key')
    critor_key = request.form.get('critor_key')
    try:
        fs_abstract.reset(video_path=video_path,
                          scorer_key=scorer_key,
                          critor_key=critor_key,
                          help_mode=False)
        res = {"is_success": True,
               "message": ''}
    except Exception as e:
        res = {"is_success": False,
               "message": str(e)}
    return json.dumps(res)


@app.route('/fs_backend/score', methods=["POST"])
def score():
    group_size = int(request.form.get('group_size'))
    resize_shape = eval(request.form.get('resize_shape'))
    sort = bool(request.form.get('sort'))
    unitized = bool(request.form.get('unitized'))
    try:
        scores = fs_abstract.score(group_size=group_size,
                                    resize_shape=resize_shape,
                                    sort=sort,
                                    unitized=unitized)
        res = {"is_success": True,
                "message": '',
                "frame_score": scores
                }
    except Exception as e:
        res = {"is_success": False,
               "message": str(e),
               "frame_score": None
               }
    return json.dumps(res)


@app.route('/fs_backend/select', methods=["POST"])
def select():
    select_num = float(request.form.get('select_num'))
    try:
        frames, selection_score = fs_abstract.select(
            select_num=select_num, reverse=False)
        res = {"is_success": True,
               "message": '',
               "selection_score": selection_score
               }
    except Exception as e:
        res = {"is_success": False,
               "message": str(e),
               "selection_score": None
               }
    return json.dumps(res)


@app.route('/fs_backend/export', methods=["POST"])
def export():
    export_dir = request.form.get("export_dir")
    try:
        frame_paths = fs_abstract.export(export_dir=export_dir)
        res = {"is_success": True,
               "message": '',
               "frame_paths": frame_paths
               }
    except Exception as e:
        res = {"is_success": False,
               "message": str(e),
               "frame_paths": None
               }
    return json.dumps(res)


class TestAPI(TestCase):
    def __init__(self, methodName: str, url_prefix: str = None) -> None:
        super().__init__(methodName)
        self.url_prefix = "http://127.0.0.1:10027/" if url_prefix is None else url_prefix
        self.tested_name = "FS API"

    def test_reset_help(self):
        print(
            f"--------------------{self.tested_name} test reset_help--------------------")
        url = urljoin(self.url_prefix, "/fs_backend/reset_help")
        print(url)
        res = requests.post(url)
        print(res.text)

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        url = urljoin(self.url_prefix, "/fs_backend/reset")
        print(url)
        req_info = {'video_path': 'my_unittest/test.mp4',
                    'scorer_key': "ssim",
                    'critor_key': "intuitive"}
        res = requests.post(url, data=req_info)
        print(res.text)

    def test_score(self):
        print(
            f"--------------------{self.tested_name} test score--------------------")
        url = urljoin(self.url_prefix, "/fs_backend/score")
        print(url)
        req_info = {'group_size': 24,
                    "resize_shape": "[64, 64]",
                    "sort": True,
                    "unitized": True}
        res = requests.post(url, data=req_info)
        print(res.text)

    def test_select(self):
        print(
            f"--------------------{self.tested_name} test select--------------------")
        url = urljoin(self.url_prefix, "/fs_backend/select")
        print(url)
        req_info = {'select_num': 0.1}
        res = requests.post(url, data=req_info)
        print(res.text)

    def test_export(self):
        print(
            f"--------------------{self.tested_name} test export--------------------")
        url = urljoin(self.url_prefix, "/fs_backend/export")
        print(url)
        req_info = {"export_dir": "data"}
        res = requests.post(url, data=req_info)
        print(res.text)
