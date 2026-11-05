package com.umer.sketchai;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Single-screen, Java/XML Android Activity.
 *
 * Layout is displayed immediately; loading the native AI runtime happens on a
 * worker thread, so a model-loading failure reports a visible error instead of
 * leaving the user staring at a black/blank screen.
 */
public final class MainActivity extends Activity {
    private DrawingView board;
    private Button predictButton;
    private TextView statusText, titleText, subtitleText;
    private LinearLayout resultCard;
    private final TextView[] scoreText=new TextView[5];
    private final ProgressBar[] scoreBars=new ProgressBar[5];
    private final ExecutorService executor=Executors.newSingleThreadExecutor();
    private SketchClassifier classifier; // Access only from executor's single worker.
    private volatile boolean destroyed=false;
    private volatile boolean modelReady=false;
    private boolean busy=false; // UI-thread only

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main); // Show a screen even before model initialization.
        board=findViewById(R.id.drawingView);
        predictButton=findViewById(R.id.predictButton);
        statusText=findViewById(R.id.statusText);
        titleText=findViewById(R.id.predictionTitle);
        subtitleText=findViewById(R.id.predictionSubtitle);
        resultCard=findViewById(R.id.resultCard);
        scoreText[0]=findViewById(R.id.score0);scoreBars[0]=findViewById(R.id.bar0);
        scoreText[1]=findViewById(R.id.score1);scoreBars[1]=findViewById(R.id.bar1);
        scoreText[2]=findViewById(R.id.score2);scoreBars[2]=findViewById(R.id.bar2);
        scoreText[3]=findViewById(R.id.score3);scoreBars[3]=findViewById(R.id.bar3);
        scoreText[4]=findViewById(R.id.score4);scoreBars[4]=findViewById(R.id.bar4);
        predictButton.setEnabled(false);
        predictButton.setOnClickListener(v -> predict());
        findViewById(R.id.clearButton).setOnClickListener(v -> {
            if (busy) return;
            board.clear(); resultCard.setVisibility(View.GONE);
            statusText.setText(modelReady ? "Canvas cleared. Draw a new shape." : "Preparing AI model…");
        });
        findViewById(R.id.undoButton).setOnClickListener(v -> {
            if (busy) return;
            board.undo();resultCard.setVisibility(View.GONE);
            statusText.setText(modelReady ? "Last stroke removed." : "Preparing AI model…");
        });
        initializeAI();
    }

    private void initializeAI() {
        statusText.setText("Loading offline AI model…");
        executor.execute(() -> {
            try {
                classifier=new SketchClassifier(getApplicationContext());
                if (destroyed) return;
                runOnUiThread(() -> {
                    modelReady=true;predictButton.setEnabled(true);
                    statusText.setText("AI ready • Draw circle, square, triangle, star, or heart.");
                });
            } catch (Exception | LinkageError e) {
                if (destroyed) return;
                final String message=readableError(e);
                runOnUiThread(() -> {
                    modelReady=false;predictButton.setEnabled(false);
                    statusText.setText("Model could not load: " + message);
                });
            }
        });
    }

    private static String readableError(Throwable exception) {
        String msg=exception.getMessage();
        if(msg==null || msg.trim().isEmpty())return exception.getClass().getSimpleName();
        if(msg.length()>130)return msg.substring(0,130)+"…";
        return msg;
    }

    private void predict() {
        if (!modelReady || busy) return;
        if (!board.hasInk()) {
            Toast.makeText(this,"Draw a shape first.",Toast.LENGTH_SHORT).show();return;
        }
        // Snapshot on the UI thread; never read mutable Android View state on the worker.
        float[][] input;
        Bitmap bitmap=null;
        try {
            bitmap=board.exportInkBitmap();
            input=SketchPreprocessor.prepare(bitmap);
        } catch (IllegalArgumentException | IllegalStateException e) {
            Toast.makeText(this,readableError(e),Toast.LENGTH_LONG).show();return;
        } finally {
            if (bitmap!=null)bitmap.recycle();
        }
        busy=true;predictButton.setEnabled(false);
        statusText.setText("Recognizing your sketch on this device…");
        executor.execute(() -> {
            try {
                SketchClassifier.Result result=classifier.recognize(input);
                if(destroyed)return;
                runOnUiThread(() -> showResult(result));
            } catch (Exception | LinkageError e) {
                if(destroyed)return;
                final String message=readableError(e);
                runOnUiThread(() -> {
                    busy=false;predictButton.setEnabled(modelReady);
                    statusText.setText("Prediction failed: " + message);
                });
            }
        });
    }

    private void showResult(SketchClassifier.Result result) {
        busy=false;predictButton.setEnabled(modelReady);
        int pct=Math.round(result.confidence*100f);
        boolean uncertain=result.confidence<.6f;
        titleText.setText(uncertain ? "Not sure — maybe " + result.name : result.name);
        subtitleText.setText(uncertain
                ? "Low confidence. Try drawing one large, clear shape."
                : "Top model score: " + pct + "% · Based on your sketch");
        for(int i=0;i<5;i++) {
            int value=Math.round(result.scores[i]*100f);
            scoreText[i].setText(value+"%");
            scoreBars[i].setProgress(value);
        }
        resultCard.setVisibility(View.VISIBLE);
        statusText.setText("Prediction complete • Works without internet.");
    }

    @Override protected void onDestroy() {
        destroyed=true;
        executor.execute(() -> {
            if(classifier!=null)classifier.close();
            classifier=null;
        });
        executor.shutdown();
        super.onDestroy();
    }
}
