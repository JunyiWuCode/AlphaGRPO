# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

import http.server
import socketserver
import os
import socket
import urllib.parse
import shutil
import argparse # 新增

# 默认配置
DEFAULT_PORT = 8081
SAVE_DIR = "/dev/shm/eval_images"
os.makedirs(SAVE_DIR, exist_ok=True)

class FileUploadHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # 关闭日志刷屏

    def do_POST(self):
        """处理图片上传"""
        try:
            # 1. 解析 Query 参数
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            filename = params.get('filename', [None])[0]
            
            if not filename:
                self.send_error(400, "Missing 'filename' parameter")
                return

            # 2. 读取二进制数据
            content_length = int(self.headers['Content-Length'])
            file_data = self.rfile.read(content_length)
            
            # 3. 写入内存盘
            file_path = os.path.join(SAVE_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # 4. 返回 file:// 绝对路径
            abs_path = os.path.abspath(file_path)
            response = f"file://{abs_path}".encode('utf-8')
            
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            self.send_error(500, str(e))

    def do_DELETE(self):
        """处理图片删除"""
        try:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            filename = params.get('filename', [None])[0]
            
            if not filename:
                self.send_error(400, "Missing filename")
                return
                
            file_path = os.path.join(SAVE_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Deleted")
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, str(e))

class ThreadingHTTPServerIPv6(socketserver.ThreadingMixIn, http.server.HTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Image Upload Sidecar for SGLang")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--dir", type=str, default=SAVE_DIR, help="Directory to save images")
    args = parser.parse_args()

    # 更新全局配置
    SAVE_DIR = args.dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 启动服务
    with ThreadingHTTPServerIPv6(("::", args.port), FileUploadHandler) as httpd:
        print(f"🚀 Sidecar running on port {args.port} (IPv6/IPv4 Dual Stack)")
        print(f"📂 Saving to: {SAVE_DIR}")
        httpd.serve_forever()