package com.umer.sketchai;

import android.graphics.Bitmap;
import android.graphics.Color;

/**
 * Converts touch ink into the normalized 32x32 mask used to train the model.
 * The model is a float32 fully-connected network with input [1,1024].
 *
 * Steps: find alpha bounding box, crop, fit longest side to 26 px, center in a
 * 32x32 black frame and normalize 0..255 alpha to 0..1. Grayscale polarity:
 * WHITE STROKES (1) on BLACK BACKGROUND (0). The Java/Python conventions match.
 */
public final class SketchPreprocessor {
    public static final int INPUT_SIZE=32;
    public static final int FIT_SIZE=26;
    private SketchPreprocessor() { }

    public static float[][] prepare(Bitmap rawBitmap) {
        if (rawBitmap==null) throw new IllegalArgumentException("Empty drawing");
        int width=rawBitmap.getWidth(),height=rawBitmap.getHeight();
        int[] all=new int[width*height];
        rawBitmap.getPixels(all,0,width,0,0,width,height);
        int left=width,top=height,right=-1,bottom=-1;
        int activePixels=0;
        for (int y=0;y<height;y++) {
            for (int x=0;x<width;x++) {
                if (Color.alpha(all[y*width+x])>8) {
                    if (x<left)left=x;
                    if (x>right)right=x;
                    if (y<top)top=y;
                    if (y>bottom)bottom=y;
                    activePixels++;
                }
            }
        }
        // Reject accidental single taps or extremely short strokes.
        if (activePixels<20 || right-left<8 || bottom-top<8) {
            throw new IllegalArgumentException("Please draw a complete shape before predicting.");
        }
        int cropWidth=right-left+1,cropHeight=bottom-top+1;
        Bitmap crop=Bitmap.createBitmap(rawBitmap,left,top,cropWidth,cropHeight);
        float scale=Math.min(FIT_SIZE/(float)cropWidth,FIT_SIZE/(float)cropHeight);
        int targetW=Math.max(1,Math.round(cropWidth*scale));
        int targetH=Math.max(1,Math.round(cropHeight*scale));
        Bitmap scaled=Bitmap.createScaledBitmap(crop,targetW,targetH,true);
        float[][] input=new float[1][INPUT_SIZE*INPUT_SIZE];
        int originX=(INPUT_SIZE-targetW)/2,originY=(INPUT_SIZE-targetH)/2;
        for (int y=0;y<targetH;y++) {
            for (int x=0;x<targetW;x++) {
                int alpha=Color.alpha(scaled.getPixel(x,y));
                input[0][(originY+y)*INPUT_SIZE+originX+x]=alpha/255.0f;
            }
        }
        scaled.recycle();
        crop.recycle();
        return input;
    }
}
