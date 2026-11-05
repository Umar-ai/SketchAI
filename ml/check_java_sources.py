"""Static Java syntax/type check using stubs when Android SDK is unavailable.

Run: python ml/check_java_sources.py
This does NOT build an APK; Android Studio must compile with actual SDK and LiteRT.
"""
from pathlib import Path
import tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'app/src/main/java/com/umer/sketchai'
stubs={
'android/content/Context.java':'''package android.content; public class Context {
 public android.content.res.AssetManager getAssets(){return null;}
 public android.content.res.Resources getResources(){return null;}
 public Context getApplicationContext(){return this;}
}''',
'android/content/res/AssetManager.java':'''package android.content.res; public class AssetManager {
 public java.io.InputStream open(String s) throws java.io.IOException {return null;}
}''',
'android/content/res/Resources.java':'''package android.content.res; public class Resources {
 public android.util.DisplayMetrics getDisplayMetrics(){return null;}
}''',
'android/util/DisplayMetrics.java':'''package android.util; public class DisplayMetrics {public float density;}''',
'android/util/AttributeSet.java':'''package android.util; public interface AttributeSet {}''',
'android/app/Activity.java':'''package android.app; public class Activity extends android.content.Context {
 protected void onCreate(android.os.Bundle b){} protected void onDestroy(){}
 public void setContentView(int layout){} public <T extends android.view.View> T findViewById(int id){return null;}
 public void runOnUiThread(Runnable r){} }''',
'android/os/Bundle.java':'''package android.os; public class Bundle {}''',
'android/view/ViewParent.java':'''package android.view; public interface ViewParent {
 void requestDisallowInterceptTouchEvent(boolean b);
}''',
'android/view/View.java':'''package android.view; public class View {
 public static final int VISIBLE=0,GONE=8; public View(android.content.Context c){}
 public View(android.content.Context c,android.util.AttributeSet a){}
 public View(android.content.Context c,android.util.AttributeSet a,int v){}
 public interface OnClickListener {void onClick(View v);}
 public void setOnClickListener(OnClickListener l){} public void setVisibility(int v){}
 public void setClickable(boolean x){} public void invalidate(){}
 public android.content.res.Resources getResources(){return null;}
 public int getWidth(){return 10;}public int getHeight(){return 10;}
 public android.view.ViewParent getParent(){return null;}
 protected void onDraw(android.graphics.Canvas c){}
 public boolean onTouchEvent(android.view.MotionEvent m){return false;}
 public boolean performClick(){return true;}
 }''',
'android/view/MotionEvent.java':'''package android.view; public class MotionEvent {
 public static final int ACTION_DOWN=0,ACTION_MOVE=2,ACTION_UP=1,ACTION_CANCEL=3;
 public int getActionMasked(){return 0;}public float getX(){return 0;}public float getY(){return 0;}
}''',
'android/widget/TextView.java':'''package android.widget; public class TextView extends android.view.View {
 public TextView(android.content.Context c){super(c);} public void setText(CharSequence s){}
}''',
'android/widget/Button.java':'''package android.widget; public class Button extends TextView {
 public Button(android.content.Context c){super(c);} public void setEnabled(boolean b){}
}''',
'android/widget/LinearLayout.java':'''package android.widget; public class LinearLayout extends android.view.View {
 public LinearLayout(android.content.Context c){super(c);}
}''',
'android/widget/ProgressBar.java':'''package android.widget; public class ProgressBar extends android.view.View {
 public ProgressBar(android.content.Context c){super(c);}public void setProgress(int p){}
}''',
'android/widget/Toast.java':'''package android.widget; public class Toast {
 public static final int LENGTH_SHORT=0,LENGTH_LONG=1;
 public static Toast makeText(android.content.Context c,CharSequence s,int t){return null;}
 public void show(){}
}''',
'android/graphics/Color.java':'''package android.graphics; public class Color {
 public static final int WHITE=0xffffff; public static int rgb(int r,int g,int b){return 0;}
 public static int alpha(int c){return 0;}
}''',
'android/graphics/Paint.java':'''package android.graphics; public class Paint {
 public static final int ANTI_ALIAS_FLAG=1;
 public enum Style {STROKE,FILL} public enum Cap {ROUND} public enum Join {ROUND}
 public enum Align {CENTER}
 public Paint(int i){} public Paint(Paint p){}
 public void setStyle(Style s){}public void setStrokeWidth(float x){}
 public void setStrokeCap(Cap c){} public void setStrokeJoin(Join j){}
 public void setColor(int c){} public void setTextAlign(Align a){}public void setTextSize(float f){}
 public void setTypeface(Typeface t){}
}''',
'android/graphics/Typeface.java':'''package android.graphics; public class Typeface {
 public static final int NORMAL=0; public static Typeface create(String n,int style){return null;}
}''',
'android/graphics/Canvas.java':'''package android.graphics; public class Canvas {
 public Canvas(){} public Canvas(Bitmap b){}
 public void drawCircle(float x,float y,float radius,Paint p){}
 public void drawText(String s,float x,float y,Paint p){}
 public void drawPath(Path path,Paint p){}
}''',
'android/graphics/Path.java':'''package android.graphics; public class Path {
 public void moveTo(float x,float y){} public void quadTo(float x,float y,float xx,float yy){}
 public void lineTo(float x,float y){}
}''',
'android/graphics/Bitmap.java':'''package android.graphics; public class Bitmap {
 public enum Config {ARGB_8888}
 public static Bitmap createBitmap(int w,int h,Config c){return null;}
 public static Bitmap createBitmap(Bitmap b,int x,int y,int w,int h){return null;}
 public static Bitmap createScaledBitmap(Bitmap b,int w,int h,boolean f){return null;}
 public int getWidth(){return 0;}public int getHeight(){return 0;}
 public void getPixels(int[] a,int offset,int stride,int x,int y,int w,int h){}
 public int getPixel(int x,int y){return 0;}
 public void recycle(){}
}''',
'org/tensorflow/lite/DataType.java':'''package org.tensorflow.lite; public enum DataType {FLOAT32}''',
'org/tensorflow/lite/Tensor.java':'''package org.tensorflow.lite; public class Tensor {
 public int[] shape(){return null;}public DataType dataType(){return null;}
}''',
'org/tensorflow/lite/Interpreter.java':'''package org.tensorflow.lite; public class Interpreter implements AutoCloseable {
 public Interpreter(java.nio.ByteBuffer b,Options o){}
 public static class Options {public Options setNumThreads(int n){return this;}}
 public Tensor getInputTensor(int index){return null;}
 public Tensor getOutputTensor(int index){return null;}
 public void run(Object in,Object out){} public void close(){}
}''',
'com/umer/sketchai/R.java':'''package com.umer.sketchai; public final class R {
 public static final class layout {public static final int activity_main=1;}
 public static final class id {
 public static final int drawingView=2,predictButton=3,statusText=4,predictionTitle=5,
 predictionSubtitle=6,resultCard=7,score0=8,score1=9,score2=10,score3=11,score4=12,
 bar0=13,bar1=14,bar2=15,bar3=16,bar4=17,clearButton=18,undoButton=19;}
}''',
}
with tempfile.TemporaryDirectory(prefix='sketchjava_') as tmp:
    root=Path(tmp)
    for name,content in stubs.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    java_files=[str(p) for p in root.rglob('*.java')]+[str(p) for p in SOURCES.glob('*.java')]
    result=subprocess.run(['javac','--release','17','-d',str(root/'classes')]+java_files,
        text=True,capture_output=True)
    if result.returncode:
        print(result.stdout,result.stderr);raise SystemExit(result.returncode)
    print('PASS: Java source type/syntax compilation against local Android + LiteRT stubs')
    print('NOTE: not an Android SDK compilation; run Android Studio Gradle build to verify.')
