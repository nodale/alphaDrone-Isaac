from dataclasses import dataclass

import msgpack
import socket

@dataclass
class QuickExtCom:
    def __init__(self, hostAddress='localhost', hostPort=65432, baudrate=57600, **kwargs):
        self.com = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.com.bind((hostAddress, hostPort))
        except:
            print(f"binding failed")
            return

        try:
            self.com.listen()
            print(f"listening on {hostAddress}:{port}")
        except:
            print(f"failed to listen")
            return

        try:
            self.conn, self.addr = self.com.accept()
            print(f"connection from {self.addr}")
        except:
            print(f"failed to accept connection")

        print(f"REDIS ENGAGED")
        super().__init__(**kwargs)

