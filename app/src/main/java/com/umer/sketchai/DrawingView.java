package com.umer.sketchai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * Finger-drawing canvas with multi-stroke support, undo, and clear.
 *
 * The visible dot grid and hint are decoration only. exportInkBitmap() renders
 * JUST the user's strokes on a transparent bitmap for model preprocessing.
 */
public class DrawingView extends View {
    private final List<Path> strokes = new ArrayList<>();
    private Path activeStroke;
    private final Paint inkPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hintPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private float lastX, lastY;
    private boolean activeHasLine = false;

    public DrawingView(Context context) { super(context); initialize(); }
    public DrawingView(Context context, AttributeSet attrs) { super(context, attrs); initialize(); }
    public DrawingView(Context context, AttributeSet attrs, int style) {
        super(context, attrs, style); initialize();
    }

    private float dp(float size) { return getResources().getDisplayMetrics().density * size; }

    private void initialize() {
        inkPaint.setStyle(Paint.Style.STROKE);
        inkPaint.setStrokeWidth(dp(9));
        inkPaint.setStrokeCap(Paint.Cap.ROUND);
        inkPaint.setStrokeJoin(Paint.Join.ROUND);
        inkPaint.setColor(Color.rgb(38, 61, 113));
        dotPaint.setColor(Color.rgb(224, 232, 244));
        dotPaint.setStyle(Paint.Style.FILL);
        hintPaint.setColor(Color.rgb(160, 173, 195));
        hintPaint.setTextAlign(Paint.Align.CENTER);
        hintPaint.setTextSize(dp(15));
        hintPaint.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        // Drawing in a scrolling layout: View handles its own touch gestures.
        setClickable(true);
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        // Small regular dots give the canvas a subtle sketchbook appearance.
        int step=Math.max(18, Math.round(dp(24)));
        float radius=dp(0.95f);
        for (int x=step/2;x<getWidth();x+=step) {
            for (int y=step/2;y<getHeight();y+=step) {
                canvas.drawCircle(x,y,radius,dotPaint);
            }
        }
        if (!hasInk() && activeStroke == null) {
            canvas.drawText("Draw a shape here",getWidth()/2f,getHeight()/2f,hintPaint);
        }
        for (Path p:strokes) canvas.drawPath(p,inkPaint);
        if (activeStroke!=null) canvas.drawPath(activeStroke,inkPaint);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        float x=Math.max(0,Math.min(getWidth(),event.getX()));
        float y=Math.max(0,Math.min(getHeight(),event.getY()));
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                activeStroke=new Path();
                activeStroke.moveTo(x,y);
                lastX=x;lastY=y;activeHasLine=false;
                getParent().requestDisallowInterceptTouchEvent(true);
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                if (activeStroke==null) return false;
                // Quadratic curves smooth finger motion, retaining actual sketch strokes.
                if (Math.abs(x-lastX)+Math.abs(y-lastY)>=1.0f) {
                    activeStroke.quadTo(lastX,lastY,(lastX+x)/2f,(lastY+y)/2f);
                    lastX=x;lastY=y;activeHasLine=true;
                }
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                if (activeStroke!=null) {
                    activeStroke.lineTo(x,y);
                    // A single tap is visible as a small ink dot rather than disappearing.
                    if (!activeHasLine) activeStroke.lineTo(x+dp(.1f),y+dp(.1f));
                    strokes.add(activeStroke);
                    activeStroke=null;
                }
                getParent().requestDisallowInterceptTouchEvent(false);
                performClick();
                invalidate();
                return true;
            case MotionEvent.ACTION_CANCEL:
                activeStroke=null;
                getParent().requestDisallowInterceptTouchEvent(false);
                invalidate();
                return true;
            default:return false;
        }
    }

    @Override public boolean performClick() { super.performClick(); return true; }

    public boolean hasInk() { return !strokes.isEmpty(); }

    public void clear() {
        strokes.clear();activeStroke=null;invalidate();
    }

    public void undo() {
        if (!strokes.isEmpty())strokes.remove(strokes.size()-1);
        activeStroke=null;invalidate();
    }

    /** Capture strokes only. Alpha channel becomes the ML input grayscale mask. */
    public Bitmap exportInkBitmap() {
        if (getWidth()<1 || getHeight()<1) throw new IllegalStateException("Canvas not laid out yet");
        Bitmap bitmap=Bitmap.createBitmap(getWidth(),getHeight(),Bitmap.Config.ARGB_8888);
        Canvas canvas=new Canvas(bitmap);
        Paint whiteInk=new Paint(inkPaint);
        whiteInk.setColor(Color.WHITE);
        for (Path path:strokes) canvas.drawPath(path,whiteInk);
        return bitmap;
    }
}
