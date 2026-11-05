"""Independently inspect the exported FlatBuffer and its numeric forward pass.
This is a STRUCTURAL and NumPy test; it does NOT replace Android Interpreter run.
Run from root: python ml/validate_model.py
"""
from pathlib import Path
import struct, json, numpy as np
R=Path(__file__).resolve().parents[1]
data=(R/'app/src/main/assets/sketchai_shapes.tflite').read_bytes()
weights=np.load(R/'ml/weights.npz')

assert data[4:8]==b'TFL3', 'Not a TensorFlow Lite FlatBuffer'
def u16(p):return struct.unpack_from('<H',data,p)[0]
def u32(p):return struct.unpack_from('<I',data,p)[0]
def i32(p):return struct.unpack_from('<i',data,p)[0]
def field(obj,n):
    vt=obj-i32(obj)
    vtlen=u16(vt)
    where=vt+4+2*n
    if where>=vt+vtlen:return 0
    delta=u16(where)
    return obj+delta if delta else 0
def indirect(p):return p+u32(p)
def vector_at(p):
    v=indirect(p);length=u32(v)
    return v+4,length
def refvec(obj,slot):
    pos=field(obj,slot)
    start,n=vector_at(pos)
    return [indirect(start+4*i) for i in range(n)]
def intvec(obj,slot):
    pos=field(obj,slot);start,n=vector_at(pos)
    return [i32(start+4*i) for i in range(n)]
def bytevec(obj,slot):
    pos=field(obj,slot)
    if not pos:return b''
    start,n=vector_at(pos);return data[start:start+n]

root=u32(0);assert u32(field(root,0))==3
ops=refvec(root,1)
code_ids=[]
for code in ops:
    # BuiltinCode enum in modern schema slot 3, legacy slot 0 fallback.
    code_ids.append(u32(field(code,3)) if field(code,3) else data[field(code,0)])
assert code_ids==[9,25], f'Expected FULLY_CONNECTED, SOFTMAX; got {code_ids}'
graph=refvec(root,2)[0]
tensors=refvec(graph,0)
shapes=[intvec(t,0) for t in tensors]
assert shapes[0]==[1,1024] and shapes[7]==[1,5],shapes
graphops=refvec(graph,3)
assert len(graphops)==3
assert intvec(graphops[0],1)==[0,1,2]
assert intvec(graphops[0],2)==[3]
assert intvec(graphops[1],1)==[3,4,5]
assert intvec(graphops[1],2)==[6]
assert intvec(graphops[2],1)==[6] and intvec(graphops[2],2)==[7]
assert data[field(graphops[0],3)]==8 # FullyConnectedOptions
assert data[field(graphops[2],3)]==9 # SoftmaxOptions
bufs=refvec(root,4)
for bufidx,key in [(1,'w0'),(2,'b0'),(3,'w1'),(4,'b1')]:
    arr=np.frombuffer(bytevec(bufs[bufidx],0),dtype='<f4')
    expected=weights[key].reshape(-1)
    assert np.array_equal(arr,expected),(key,arr.size,expected.size)

from train_export import drawing,CLASSES
import random
x=np.stack([drawing(kind,random.Random(700011+i),False).ravel() for i,kind in enumerate(CLASSES)])
w0=weights['w0'];b0=weights['b0'];w1=weights['w1'];b1=weights['b1']
hidden=np.maximum(0,x@w0.T+b0)
logits=hidden@w1.T+b1
logits*=0.25  # TFLite SoftmaxOptions.beta for less saturated model scores
exp=np.exp(logits-logits.max(axis=1,keepdims=True));scores=exp/exp.sum(axis=1,keepdims=True)
assert (scores.argmax(axis=1)==np.arange(5)).all(),scores
assert np.max(np.abs(scores.sum(axis=1)-1))<1e-6
print('PASS: TFL3 model version 3; FC, FC, Softmax operators')
print('PASS: Float32 model tensors input [1,1024], output [1,5]')
print('PASS: model binary weights match trained weights')
print('PASS: NumPy reference predictions for the 5 sample classes:')
for label,s in zip(CLASSES,scores):
 print(' ',label,'=>',str(CLASSES[int(s.argmax())]),round(float(s.max()),4))
print('NOTE: this does not execute the model in a real Android/LiteRT runtime.')
