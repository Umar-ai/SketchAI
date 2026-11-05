import struct
import numpy as np

class FlatBufferWriter:
 """Small reverse FlatBuffer schema writer for TFLite FULLY_CONNECTED/RELU/SOFTMAX."""
 def __init__(self): self.buf=bytearray();self.minalign=1;self.vtable=None;self.object_start=0
 def offset(self):return len(self.buf)
 def pad(self,n):self.buf[:0]=b'\0'*n
 def prep(self,size,additional=0):
  self.minalign=max(self.minalign,size);self.pad((-(len(self.buf)+additional))&(size-1))
 def prepend(self,fmt,v):
  sz=struct.calcsize('<'+fmt);self.prep(sz);self.buf[:0]=struct.pack('<'+fmt,v)
 def int32(self,v):self.prepend('i',v)
 def uint32(self,v):self.prepend('I',v)
 def byte(self,v):self.prepend('B',v)
 def float(self,v):self.prepend('f',v)
 def offset_to(self,target):
  self.prep(4);assert target<=self.offset();self.uint32(self.offset()-target+4)
 def vec_int(self,values):
  self.prep(4,len(values)*4);self.prep(4,len(values)*4)
  for v in reversed(values):self.int32(v)
  self.uint32(len(values));return self.offset()
 def vec_bytes(self,data):
  data=bytes(data);self.prep(4,len(data));self.prep(1,len(data))
  self.buf[:0]=data;self.uint32(len(data));return self.offset()
 def vec_offsets(self,offsets):
  self.prep(4,len(offsets)*4);self.prep(4,len(offsets)*4)
  for x in reversed(offsets):self.offset_to(x)
  self.uint32(len(offsets));return self.offset()
 def string(self,s):
  b=s.encode('utf8');self.prep(4,len(b)+1)
  self.buf[:0]=b'\0';self.buf[:0]=b;self.uint32(len(b));return self.offset()
 def start(self,n):
  assert self.vtable is None;self.vtable=[0]*n;self.object_start=self.offset()
 def slot_u(self,i,v,d=0):
  if v!=d:self.uint32(v);self.vtable[i]=self.offset()
 def slot_b(self,i,v,d=0):
  if v!=d:self.byte(v);self.vtable[i]=self.offset()
 def slot_f(self,i,v,d=0.):
  if v!=d:self.float(v);self.vtable[i]=self.offset()
 def slot_ref(self,i,off):
  if off:self.offset_to(off);self.vtable[i]=self.offset()
 def end(self):
  self.int32(0);obj=self.offset()
  for off in reversed(self.vtable):self.prepend('H',obj-off if off else 0)
  self.prepend('H',obj-self.object_start);self.prepend('H',(len(self.vtable)+2)*2)
  vt=self.offset();struct.pack_into('<i',self.buf,len(self.buf)-obj,vt-obj)
  self.vtable=None;return obj
 def finish(self,root):
  self.prep(self.minalign,8);self.buf[:0]=b'TFL3';self.offset_to(root);return bytes(self.buf)


