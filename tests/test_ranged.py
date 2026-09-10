from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.ranged import RangeReader


class Response:
    def __init__(self,status=206,span='bytes 10-13/100',body=b'abcd'):
        self.status=status;self.headers={'Content-Range':span,'Content-Length':str(len(body))};self.body=body;self.read_calls=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):self.read_calls.append(n);return self.body[:n]


class RangeTests(unittest.TestCase):
    def test_full_response_rejected_before_body(self):
        response=Response(status=200)
        reader=RangeReader('https://example.test/file',100,opener=lambda *a,**k:response)
        with self.assertRaises(ValueError):reader.read(10,4)
        self.assertEqual(response.read_calls,[])

    def test_wrong_range_rejected_before_body(self):
        response=Response(span='bytes 0-3/100')
        reader=RangeReader('https://example.test/file',100,opener=lambda *a,**k:response)
        with self.assertRaises(ValueError):reader.read(10,4)
        self.assertEqual(response.read_calls,[])

    def test_budget_and_bounds(self):
        response=Response();reader=RangeReader('https://example.test/file',100,budget=4,opener=lambda *a,**k:response)
        self.assertEqual(reader.read(10,4),b'abcd')
        self.assertEqual(reader.received,4)
        with self.assertRaises(ValueError):reader.read(14,1)
        with self.assertRaises(ValueError):reader.read(99,2)
        self.assertEqual(response.read_calls,[4])
