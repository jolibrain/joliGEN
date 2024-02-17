from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from signal import signal, SIGINT
from sys import exit
from json import loads, dumps
from argparse import ArgumentParser
from threading import Thread
import sys
import time

sys.path.append(sys.path[0] + "/..")
import client


class HttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    error_content_type = "text/plain"
    error_message_format = "Error %(code)d: %(message)s"

    def do_GET(self):
        path, args = self.parse_url()

        if path == "/train" and args == {}:
            self.write_response(
                200,
                "application/json",
                dumps({"processes": {}}),
            )
        else:
            self.send_error(400, "Invalid path or args")

    def do_DELETE(self):
        path, args = self.parse_url()

        if path == "/train/test_client" and args == {}:
            self.write_response(
                200,
                "application/json",
                dumps({"message": "ok", "name": "test_client"}),
            )
        else:
            self.send_error(400, "Invalid path or args")

    def do_POST(self):
        path, _ = self.parse_url()
        body = self.read_body()

        if (
            path == "/train/test_client"
            and self.parse_json(body)["train_options"] is not None
        ):
            self.write_response(
                200,
                "application/json",
                dumps({"message": "ok", "name": "test_client", "status": "running"}),
            )
        else:
            self.send_error(400, "Invalid json received")

    def parse_url(self):
        url_components = urlparse(self.path)
        return url_components.path, parse_qs(url_components.query)

    def parse_json(self, content):
        try:
            return loads(content)
        except Exception:
            return None

    def read_body(self):
        try:
            content_length = int(self.headers["Content-Length"])
            return self.rfile.read(content_length).decode("utf-8")
        except Exception:
            return None

    def write_response(self, status_code, content_type, content):
        response = content.encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def version_string(self):
        return "Tiny Http Server"

    def log_error(self, format, *args):
        pass


def start_server(host, port):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, HttpHandler)
    print(f"Server started on {host}:{port}")
    httpd.serve_forever()


def shutdown_handler(signum, frame):
    print("Shutting down server")
    exit(0)


def test_client_sever(host, port):
    signal(SIGINT, shutdown_handler)

    Thread(target=start_server, daemon=True, args=[host, port]).start()

    time.sleep(1)

    client.main_client(
        args=[
            "--host",
            host,
            "--port",
            str(port),
            "--method",
            "training_status",
            "--name",
            "test_client",
        ]
    )

    time.sleep(1)

    client.main_client(
        args=[
            "--host",
            host,
            "--port",
            str(port),
            "--method",
            "launch_training",
            "--name",
            "test_client",
            "--dataroot",
            "fake_dataroot",
        ]
    )

    time.sleep(1)

    client.main_client(
        args=[
            "--host",
            host,
            "--port",
            str(port),
            "--method",
            "stop_training",
            "--name",
            "test_client",
        ]
    )
