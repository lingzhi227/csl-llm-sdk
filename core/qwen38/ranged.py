"""Bounded strict HTTP ranges. Never read a server's full-file fallback body."""
import re
import threading
import urllib.request


class RangeReader:
    def __init__(self, url, total_size, budget=8*1024*1024, opener=urllib.request.urlopen):
        if type(total_size) is not int or total_size <= 0 or type(budget) is not int or budget <= 0:
            raise ValueError('Invalid file size or payload budget')
        self.url, self.total_size, self.budget, self.opener = url, total_size, budget, opener
        self.reserved = self.received = 0
        self.lock = threading.Lock()
        self.receipts = []

    def read(self, start, length):
        if any(type(x) is not int for x in (start,length)) or start < 0 or length <= 0 or start+length > self.total_size:
            raise ValueError('Range out of bounds')
        with self.lock:
            if self.reserved+length > self.budget:
                raise ValueError('Network payload budget exceeded')
            self.reserved += length
        end = start+length-1
        separator = '&' if '?' in self.url else '?'
        request = urllib.request.Request(self.url+separator+f'wp01_range={start}-{end}',
            headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
        with self.opener(request,timeout=30) as response:
            expected=f'bytes {start}-{end}/{self.total_size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Refused non-206 or mismatched Content-Range before reading body')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('Compressed response is not a raw byte range')
            declared=response.headers.get('Content-Length')
            if declared is not None and int(declared) != length:
                raise ValueError('Range Content-Length mismatch')
            body=response.read(length)
            with self.lock:
                self.received += len(body)
            if len(body) != length:
                raise ValueError('Truncated range body')
            # Never read beyond the agreed range or allocate based on remote size.
            with self.lock:
                self.receipts.append({'start':start,'length':length,'status':206,'content_range':expected})
            return body
