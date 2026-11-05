"""Reproducible shape-model training and genuine TFLite export.

Training data: procedural, author-generated line drawings, not Google's Quick, Draw!
Data and classifier are for an educational mini assignment, not a general sketch recognizer.

Usage from project root: python ml/train_export.py
Requires: torch, pillow, numpy. TFLite FlatBuffer schema writer is bundled locally.
The exported dense layers use TFLite FULLY_CONNECTED + RELU + SOFTMAX built-in ops.
"""
from pathlib import Path
import math, json, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
from flatbuffer_writer import FlatBufferWriter

SEED=20261030
CLASSES=['Circle','Square','Triangle','Star','Heart']
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'app/src/main/assets'
ASSETS.mkdir(parents=True,exist_ok=True)


def shape_points(kind, rng):
    # Each primitive is built from line points; introduce rotation, uneven pressure,
    # imperfect corners and minor deformities to simulate finger sketches.
    rotation=rng.uniform(-0.30,0.30)
    sx=rng.uniform(.83,1.13);sy=rng.uniform(.83,1.13)
    distortion=rng.uniform(0.0,.045)
    if kind=='Circle':
        n=48; pts=[]
        phase1=rng.uniform(0,6.283);phase2=rng.uniform(0,6.283)
        for i in range(n+1):
            t=2*math.pi*i/n
            rr=1 + distortion*math.sin(3*t+phase1)+distortion*.65*math.cos(5*t+phase2)
            pts.append((rr*math.cos(t),rr*math.sin(t)))
    elif kind=='Square':
        # Make corners at varying positions; soften some to resemble freehand.
        corners=[(-1,-1),(1,-1),(1,1),(-1,1)]
        corners=[(x+rng.uniform(-.12,.12), y+rng.uniform(-.12,.12)) for x,y in corners]
        pts=[]
        for i in range(4):
            a=corners[i];b=corners[(i+1)%4]
            for j in range(9):
                p=j/9
                pts.append((a[0]*(1-p)+b[0]*p, a[1]*(1-p)+b[1]*p))
        pts.append(pts[0]);
    elif kind=='Triangle':
        corners=[(-.98,.92),(0,-1.12),(.98,.92)]
        corners=[(x+rng.uniform(-.14,.14), y+rng.uniform(-.12,.12)) for x,y in corners]
        pts=[]
        for i in range(3):
            a=corners[i];b=corners[(i+1)%3]
            for j in range(12):
                p=j/12
                pts.append((a[0]*(1-p)+b[0]*p,a[1]*(1-p)+b[1]*p))
        pts.append(pts[0])
    elif kind=='Star':
        inner=rng.uniform(.40,.52)
        corners=[]
        for i in range(10):
            t=math.pi*i/5-math.pi/2
            r=1.1 if i%2==0 else inner
            corners.append((r*math.cos(t),r*math.sin(t)))
        pts=[]
        for i in range(10):
            a=corners[i];b=corners[(i+1)%10]
            for j in range(5):
                p=j/5
                pts.append((a[0]*(1-p)+b[0]*p,a[1]*(1-p)+b[1]*p))
        pts.append(pts[0])
    else: # Heart, use a standard smooth closed heart contour.
        pts=[]
        for i in range(71):
            t=2*math.pi*i/70
            xx=16*math.sin(t)**3 / 17
            yy=-(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))/17
            pts.append((xx, yy))
    c,s=math.cos(rotation),math.sin(rotation)
    pts=[(64+40*(x*sx*c-y*sy*s), 64+40*(x*sx*s+y*sy*c)) for x,y in pts]
    return pts


def preprocess_mask(im, training_jitter=False, rng=None):
    """Crop ink to its bounding box, center, and resize longest side to 26px.
    Equivalent to Java SketchPreprocessor: white ink on black (0..1 alpha).
    """
    im=im.convert('L')
    arr=np.asarray(im)
    yy,xx=np.where(arr>8)
    if len(xx)==0:return np.zeros((32,32),np.float32)
    crop=im.crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
    w,h=crop.size
    scale=min(26.0/w,26.0/h)
    nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    small=crop.resize((nw,nh),Image.Resampling.BILINEAR)
    out=Image.new('L',(32,32),0)
    ox=(32-nw)//2;oy=(32-nh)//2
    if training_jitter:
        ox+=rng.randint(-2,2);oy+=rng.randint(-2,2)
    out.paste(small,(ox,oy))
    return np.asarray(out,dtype=np.float32)/255.0


def drawing(kind,rng,training=True):
    image=Image.new('L',(128,128),0)
    draw=ImageDraw.Draw(image)
    pts=shape_points(kind,rng)
    width=rng.randint(6,12)
    shade=rng.randint(210,255)
    if training and rng.random()<.05:
        # Occasionally leave a small stroke gap (common in finger drawings).
        k=rng.randint(4,len(pts)-5)
        draw.line(pts[:k], fill=shade, width=width, joint='curve')
        draw.line(pts[k+3:], fill=shade, width=width, joint='curve')
    else:
        draw.line(pts,fill=shade,width=width,joint='curve')
    for xx,yy in [pts[0],pts[-1]]:
        rr=width/2;draw.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),fill=shade)
    if training and rng.random()<.1:
        image=image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(.3,1.0)))
    return preprocess_mask(image, training_jitter=training, rng=rng)


def generate(per_class, offset, training):
    xs=[];ys=[]
    for label,kind in enumerate(CLASSES):
        rng=random.Random(SEED+label*100_003+offset)
        for _ in range(per_class):
            xs.append(drawing(kind,rng,training));ys.append(label)
    xs=np.stack(xs).reshape(-1,1024).astype(np.float32)
    ys=np.asarray(ys,dtype=np.int64)
    return xs,ys


def export_tflite(w0,b0,w1,b1,description):
    """Serialize float32 1024->96->5 network to valid TFL3 FlatBuffer.
    This is a small transparent specialized exporter. Operators: FC(ReLU), FC, SOFTMAX.
    """
    assert w0.ndim==2 and w0.shape[1]==1024
    assert b0.shape==(w0.shape[0],)
    assert w1.shape==(len(CLASSES),w0.shape[0])
    assert b1.shape==(len(CLASSES),)
    f=FlatBufferWriter()
    buf=[]
    for blob in [b'',w0.astype('<f4').tobytes(),b0.astype('<f4').tobytes(),
                 w1.astype('<f4').tobytes(),b1.astype('<f4').tobytes()]:
        v=f.vec_bytes(blob) if blob else 0
        f.start(1);f.slot_ref(0,v);buf.append(f.end())
    hidden=w0.shape[0];labels=len(CLASSES)
    tensors=[]
    for shape,bidx,name in [([1,1024],0,'sketch_pixels'),([hidden,1024],1,'w0'),
        ([hidden],2,'b0'),([1,hidden],0,'relu_hidden'),([labels,hidden],3,'w1'),
        ([labels],4,'b1'),([1,labels],0,'logits'),([1,labels],0,'shape_probabilities')]:
        sr=f.vec_int(shape);n=f.string(name)
        f.start(12);f.slot_ref(0,sr);f.slot_u(2,bidx);f.slot_ref(3,n);tensors.append(f.end())
    f.start(4);f.slot_b(0,1);relu=f.end()
    f.start(4);fc_options=f.end()
    # Temperature-scaled softmax (beta=0.25): moderates otherwise saturated scores.
    f.start(1);f.slot_f(0,.25);softmax=f.end()
    ops=[]
    for code,ins,outs,opt_type,opt in [(0,[0,1,2],[3],8,relu),
                                      (0,[3,4,5],[6],8,fc_options),
                                      (1,[6],[7],9,softmax)]:
        iv=f.vec_int(ins);ov=f.vec_int(outs)
        f.start(5);f.slot_u(0,code);f.slot_ref(1,iv);f.slot_ref(2,ov)
        f.slot_b(3,opt_type);f.slot_ref(4,opt);ops.append(f.end())
    tref=f.vec_offsets(tensors);inputs=f.vec_int([0]);outputs=f.vec_int([7]);oref=f.vec_offsets(ops)
    graph_name=f.string('Sketch AI synthetic shape recognizer')
    f.start(5);f.slot_ref(0,tref);f.slot_ref(1,inputs);f.slot_ref(2,outputs)
    f.slot_ref(3,oref);f.slot_ref(4,graph_name);graph=f.end()
    codes=[]
    for builtin in (9,25):
        f.start(4);f.slot_b(0,builtin);f.slot_u(3,builtin);codes.append(f.end())
    cref=f.vec_offsets(codes);graphs=f.vec_offsets([graph]);bref=f.vec_offsets(buf)
    about=f.string(description)
    f.start(5);f.slot_u(0,3);f.slot_ref(1,cref);f.slot_ref(2,graphs)
    f.slot_ref(3,about);f.slot_ref(4,bref)
    data=f.finish(f.end());assert data[4:8]==b'TFL3'
    return data


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print('Generating synthetic handwritten shapes... ',flush=True)
    xtrain,ytrain=generate(1800,0,True)
    xtest,ytest=generate(300,803998,True)
    # Distinct seeds and no shared images between train and test.
    model=nn.Sequential(nn.Linear(1024,96),nn.ReLU(),nn.Linear(96,len(CLASSES)))
    opt=torch.optim.Adam(model.parameters(),lr=.0018)
    lossfun=nn.CrossEntropyLoss()
    trainx=torch.from_numpy(xtrain);trainy=torch.from_numpy(ytrain)
    testx=torch.from_numpy(xtest);testy=torch.from_numpy(ytest)
    for epoch in range(18):
        model.train();perm=torch.randperm(len(trainx));losses=[]
        for chunk in perm.split(256):
            logits=model(trainx[chunk]);loss=lossfun(logits,trainy[chunk]);
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            guess=model(testx).argmax(dim=1)
            acc=float((guess==testy).float().mean().item())
        if epoch%3==0 or epoch==17:print(f'epoch={epoch+1:02d} synthetic_holdout_acc={acc:.3f} train_loss={np.mean(losses):.4f}',flush=True)
    W0=model[0].weight.detach().numpy();B0=model[0].bias.detach().numpy()
    W1=model[2].weight.detach().numpy();B1=model[2].bias.detach().numpy()
    np.savez_compressed(ROOT/'ml/weights.npz',w0=W0,b0=B0,w1=W1,b1=B1)
    blob=export_tflite(W0,B0,W1,B1,'Synthetic drawings only; 5 classes, 32x32 normalized stroke mask, input [1,1024] output [1,5].')
    (ASSETS/'sketchai_shapes.tflite').write_bytes(blob)
    (ASSETS/'labels.txt').write_text('\n'.join(CLASSES)+'\n',encoding='utf8')
    with torch.no_grad():
        pred=model(testx).argmax(dim=1).numpy()
        cm=np.zeros((5,5),dtype=int)
        for y,yhat in zip(ytest,pred):cm[y,yhat]+=1
        acc=float(np.mean(pred==ytest))
    report={'dataset':'procedural synthetic line sketches','classes':CLASSES,
            'samples_training':len(xtrain),'samples_heldout':len(xtest),
            'seed':SEED,'synthetic_heldout_accuracy':round(acc,5),
            'confusion_matrix_true_rows_predicted_cols':cm.tolist(),
            'model_tensor_input_shape':[1,1024], 'model_tensor_output_shape':[1,5],
            'output_softmax_beta':0.25,
            'note':'No real human sketches, Android APK, or device inference evaluated here. Not a real-world accuracy estimate.'}
    (ROOT/'MODEL_CARD.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    # Save a small illustrated contact sheet of generated samples for source reproducibility.
    from PIL import ImageOps, ImageDraw
    out=Image.new('RGB',(5*165,6*144),(241,246,250));d=ImageDraw.Draw(out)
    for col,kind in enumerate(CLASSES):
        d.text((col*165+15,10),kind,fill=(22,40,67))
        rng=random.Random(SEED+910002+col)
        for row in range(5):
            mask=drawing(kind,rng,True)
            im=Image.fromarray(np.uint8(mask*255),'L').resize((112,112),Image.Resampling.NEAREST)
            img=ImageOps.colorize(im,black=(246,250,252),white=(21,47,91))
            out.paste(img,(col*165+18,row*144+35))
    out.save(ROOT/'ml/synthetic_samples.png')
    print('Model saved to', ASSETS/'sketchai_shapes.tflite')

if __name__=='__main__':main()
