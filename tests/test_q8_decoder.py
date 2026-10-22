"""Independent CPU oracle for the packed Q8_0 row decoder; no torch or CUDA."""
import array
import ctypes
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name=='nt' or not shutil.which('g++'):
            if os.environ.get('Q6_REQUIRE_CPP')=='1':raise RuntimeError('Linux g++ required')
            raise unittest.SkipTest('Run this native CPU test on Linux/WSL with g++')
        cls.temp=tempfile.TemporaryDirectory(prefix='q6-decoder-test-')
        library=Path(cls.temp.name)/'q8_rows.so'
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(ROOT/'src/flashnext_gguf/q8_rows.cpp'),'-o',str(library)],check=True)
        cls.lib=ctypes.CDLL(str(library))
        cls.fn=cls.lib.q8_get_rows
        cls.fn.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int]
        cls.fn.restype=ctypes.c_int

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_rows(self,packed,rows,width,ids,kind=0):
        data=ctypes.create_string_buffer(packed)
        indices=(ctypes.c_int64*len(ids))(*ids)
        out=((ctypes.c_float if kind==0 else ctypes.c_uint16)*(len(ids)*width))()
        rc=self.fn(data,rows,width,indices,len(ids),out,kind)
        self.assertEqual(rc,0)
        return out

    def test_reordered_repeated_rows_and_signed_bytes(self):
        packed=b''.join(struct.pack('<e',scale)+bytes(range(offset,offset+32)) for scale,offset in [(0.5,0),(-2,128),(1.25,224)])
        result=self.run_rows(packed,3,32,[2,0,2,1])
        expected=[scale*(b if b<128 else b-256) for row in [2,0,2,1] for scale,offset in [[(0.5,0),(-2,128),(1.25,224)][row]] for b in range(offset,offset+32)]
        self.assertEqual(list(result),expected)

    def test_all_half_scale_encodings_and_bf16_rounding(self):
        packed=bytearray()
        qbytes=bytes([0,1,127,128,255,3,129,64]*4)
        signed=[x if x<128 else x-256 for x in qbytes]
        for scale in range(65536):packed+=struct.pack('<H',scale)+qbytes
        f32=self.run_rows(bytes(packed),65536,32,list(range(65536)))
        bf16=self.run_rows(bytes(packed),65536,32,list(range(65536)),1)
        for bits in range(65536):
            scale=struct.unpack('<e',struct.pack('<H',bits))[0]
            for col,q in enumerate(signed):
                expected=scale*q
                idx=bits*32+col
                if math.isnan(expected):
                    self.assertTrue(math.isnan(f32[idx]))
                    self.assertEqual(bf16[idx]&0x7f80,0x7f80)
                    self.assertNotEqual(bf16[idx]&0x7f,0)
                else:
                    self.assertEqual(f32[idx],expected)
                    raw=struct.unpack('<I',struct.pack('<f',expected))[0]
                    want=((raw+0x7fff+((raw>>16)&1))>>16)&0xffff
                    # Both signed representations of zero compare equally.
                    if expected==0:self.assertEqual(bf16[idx]&0x7fff,0)
                    else:self.assertEqual(bf16[idx],want)

    def test_invalid_id_does_not_partially_overwrite_output(self):
        data=ctypes.create_string_buffer(struct.pack('<e',1)+bytes([1]*32))
        for bad in [-1,1]:
            ids=(ctypes.c_int64*2)(0,bad);out=(ctypes.c_float*64)(*([123]*64))
            self.assertEqual(self.fn(data,1,32,ids,2,out,0),-2)
            self.assertEqual(list(out),[123]*64)

    def test_empty_and_invalid_shape(self):
        self.assertEqual(self.fn(None,0,32,None,0,None,0),0)
        self.assertEqual(self.fn(None,0,31,None,0,None,0),-1)
        self.assertEqual(self.fn(None,0,32,None,0,None,9),-1)


if __name__=='__main__':unittest.main()
