#!/usr/bin/env python
# -*- coding: utf8 -*-

import requests
import json
import base64
from threading import Lock

class MatrixClient:

    def __init__(self, addr, timeout, type):
        self.addr = addr
        self.timeout = timeout
        self.type = type
        self.header = {"Content-type": "application/json"}
        self.session_id = 0
        self.lock = Lock()
        self.functions = [2,20,21]
        self.svr_type = 3

    def recognize_single(self, image_uri):
        post_data = self.create_rec_data(image_uri)
        try:
            resp = requests.post(self.addr, data=post_data, timeout=self.timeout)
        except:
            print "post error"

        return resp.content

    def recognize_batch(self, image_uris):
        return 0

    def rank_image(self, image_uri):

        return 0

    def rank_feature(self, feature):
        rank_data = self.create_rank_data(1, feature)
        print rank_data
        try:
            resp = requests.post(self.addr, data=rank_data, timeout=self.timeout)
        except:
            print "rank error"

        return resp.content

    def create_session_id(self):
        self.lock.acquire()
        sid = self.session_id
        self.session_id = self.session_id + 1
        self.lock.release()
        return sid

    def create_rec_data(self, image_uri):
        post_data = {"Context": {"SessionId": 'singleReq_restful' + str(self.create_session_id()), "Functions" : self.functions, "Type": self.svr_type}}
        post_data['Image'] = {"Data": {"URI":image_uri}}
        return json.dumps(post_data)

    def create_rank_data(self, id, feature):
        post_data = {}
        context = {"SessionId": 'rank_restful' + str(self.create_session_id()), "Type": 2}
        params = [{"key":"ImageData", "value": "false"}]
        post_data['Feature'] = {"Id": id, "Feature": feature}
        context["Params"] = params
        post_data["Context"] = context
        return json.dumps(post_data)


