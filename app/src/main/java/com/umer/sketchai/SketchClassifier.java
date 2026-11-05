package com.umer.sketchai;

import android.content.Context;
import android.content.res.AssetManager;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Real on-device neural-network classification using the LiteRT Interpreter.
 * Input [1,1024] float32: flattened normalized 32x32 pixel mask.
 * Output [1,5] float32: SOFTMAX scores, in labels.txt order.
 * All processing stays on the device. No internet access is requested.
 */
public final class SketchClassifier implements AutoCloseable {
    private final Interpreter interpreter;
    private final String[] labels;

    public SketchClassifier(Context context) throws IOException {
        AssetManager assets=context.getAssets();
        List<String> lines=new ArrayList<>();
        try(BufferedReader reader=new BufferedReader(new InputStreamReader(
                assets.open("labels.txt"), StandardCharsets.UTF_8))) {
            String row;
            while ((row=reader.readLine())!=null) {
                if (!row.trim().isEmpty()) lines.add(row.trim());
            }
        }
        labels=lines.toArray(new String[0]);
        // A direct, native-order ByteBuffer also works for compressed assets.
        ByteBuffer model;
        try(InputStream in=assets.open("sketchai_shapes.tflite");
            ByteArrayOutputStream stream=new ByteArrayOutputStream()) {
            byte[] chunk=new byte[8192];int read;
            while((read=in.read(chunk))!=-1)stream.write(chunk,0,read);
            byte[] bytes=stream.toByteArray();
            model=ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
            model.put(bytes);model.rewind();
        }
        Interpreter loaded=new Interpreter(model,new Interpreter.Options().setNumThreads(2));
        int[] ins=loaded.getInputTensor(0).shape();
        int[] outs=loaded.getOutputTensor(0).shape();
        if (labels.length!=5 || ins.length!=2 || ins[0]!=1 || ins[1]!=1024
                || outs.length!=2 || outs[0]!=1 || outs[1]!=labels.length
                || loaded.getInputTensor(0).dataType()!=DataType.FLOAT32
                || loaded.getOutputTensor(0).dataType()!=DataType.FLOAT32) {
            loaded.close();
            throw new IOException("Sketch AI model tensor shapes or types are unexpected");
        }
        interpreter=loaded;
    }

    public Result recognize(float[][] drawing) {
        if(drawing==null || drawing.length!=1 || drawing[0].length!=1024) {
            throw new IllegalArgumentException("Expected a 32x32 image tensor");
        }
        float[][] scores=new float[1][labels.length];
        interpreter.run(drawing,scores); // GENUINE neural inference; no if/else shape detection.
        int best=0;float bestValue=-1f,sum=0f;
        for(int i=0;i<labels.length;i++) {
            float value=scores[0][i];
            if (Float.isNaN(value) || Float.isInfinite(value) || value<0f || value>1.01f)
                throw new IllegalStateException("The model returned an invalid score");
            sum+=value;
            if(value>bestValue){bestValue=value;best=i;}
        }
        if (Math.abs(sum-1f)>.02f) throw new IllegalStateException("Scores do not sum to 1");
        return new Result(labels[best],best,bestValue,scores[0].clone());
    }

    public String[] getLabels() { return labels.clone(); }
    @Override public void close() { interpreter.close(); }

    public static final class Result {
        public final String name;
        public final int index;
        public final float confidence;
        public final float[] scores;
        Result(String label,int index,float value,float[] values) {
            this.name=label;this.index=index;this.confidence=value;this.scores=values;
        }
    }
}
