package com.tweeneural.app;

//import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

/**
 * ORIGINAL ImagePreprocessor: Created by susaneraly on 6/23/16.
 * A preprocessor specifically for images that applies min max scaling
 * Can take a range, so pixel values can be scaled from 0->255 to minRange->maxRange
 * default minRange = 0 and maxRange = 1;
 * If pixel values are not 8 bits, you can specify the number of bits as the third argument in the constructor
 * For values that are already floating point, specify the number of bits as 1
 * NEWpreprocessingscaler created by Benjamin Sobel on 01/22/2019
 * Altered to add fitlabeling!!!!!
 */
//@Slf4j
public class NewPreProcessingScaler implements DataNormalization {

    private double minRange, maxRange;
    private double maxPixelVal;
    private int maxBits;
    private boolean fittable = true;
    private boolean checkTransforms = true;

    public NewPreProcessingScaler() {
        this(0, 1, 8);
    }

    public NewPreProcessingScaler(double a, double b) {
        this(a, b, 8);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     * @param a, default = 0
     * @param b, default = 1
     * @param maxBits in the image, default = 8
     */
    public NewPreProcessingScaler(double a, double b, int maxBits) {
        //Image values are not always from 0 to 255 though
        //some images are 16-bit, some 32-bit, integer, or float, and those BTW already come with values in [0..1]...
        //If the max expected value is 1, maxBits should be specified as 1
        maxPixelVal = Math.pow(2, maxBits) - 1;
        this.minRange = a;
        this.maxRange = b;
    }

    public NewPreProcessingScaler toFittable(boolean f) {
        this.fittable = f;
        return this;
    }
    /**
     * Fit a dataset (only compute
     * based on the statistics from this dataset0
     *
     * @param dataSet the dataset to compute on
     */
    @Override
    public void fit(DataSet dataSet) {

    }

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    @Override
    public void fit(DataSetIterator iterator) {

    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();
        INDArray labels = toPreProcess.getLabels();
        this.transformChecked(features);
        this.transformChecked(labels);
    }

    public void transformChecked(INDArray features) {
        //double checkmax = features.maxNumber().doubleValue();
        if (this.checkTransforms) {
            if ((features.maxNumber().doubleValue() > this.maxRange) | !this.checkTransforms) {
                this.unsafeTransform(features);
            }
        }
        else {
            this.unsafeTransform(features);
        }
    }
    public void revertChecked(INDArray features) {
        //double checkmax = features.maxNumber().doubleValue();
        // check whether its already been reverted using the max number as a guide.
        //if not checking transforms via boolean, just run it whether its good or not.
        //Idea: change the if statements so it doesn't run potentially costly maxvalue seeking
        //when checktransforms is off.
        if (this.checkTransforms) {
            if (features.maxNumber().doubleValue() <= this.maxRange){
                this.unsafeRevert(features);
            }
        }
        else {
            this.unsafeRevert(features);
        }
    }
    public void unsafeRevert(INDArray features) {
        if (minRange != 0) {
            features.subi(minRange);
        }
        if (maxRange - minRange != 1.0) {
            features.divi(maxRange - minRange);
        }
        features.muli(this.maxPixelVal);
    }
    public void unsafeTransform(INDArray features) {
        features.divi(this.maxPixelVal); //Scaled to 0->1
        if (this.maxRange - this.minRange != 1)
            features.muli(this.maxRange - this.minRange); //Scaled to minRange -> maxRange
        if (this.minRange != 0)
            features.addi(this.minRange); //Offset by minRange
    }
    public void checkRanges(INDArray... features) {
        for (INDArray feature : features) {
            //double checkmax = feature.maxNumber().doubleValue();
            System.out.println("BEGIN");
            if ((feature.maxNumber().doubleValue() > this.maxRange) | !this.checkTransforms) {
                System.out.println("This INDArray has a maximum above one: "
                        + feature.maxNumber().doubleValue() + " max: " + this.maxRange);
                System.out.println("This array would be transformed from a normal image to a bunch of floats.");
            }
            if ((feature.maxNumber().doubleValue() <= this.maxRange) | !this.checkTransforms) {
                System.out.println("This INDArray has a maximum below or equal to one: "
                        + feature.maxNumber().doubleValue() + " max: " + this.maxRange);
                System.out.println("This Array would be reverted to a normal image.");
            }
            if (!this.checkTransforms) {
                System.out.println("Check transforms is off, so any checked transform would go");
            }
            System.out.println("END");
        }
    }
    public NewPreProcessingScaler setTransformChecks(boolean b) {
        this.checkTransforms = b;
        return this;
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    @Override
    public void transform(INDArray features) {
        this.transformChecked(features);
    }

    @Override
    public void transform(INDArray features, INDArray featuresMask) {
        this.transformChecked(features);
    }

    @Override
    public void transformLabel(INDArray label) {
        this.transformChecked(label);
    }

    @Override
    public void transformLabel(INDArray labels, INDArray labelsMask) {
        this.transformChecked(labels);
    }

    @Override
    public void revert(DataSet toRevert) {
        revertChecked(toRevert.getFeatures());
        revertChecked(toRevert.getLabels());
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.IMAGE_MIN_MAX;
    }

    @Override
    public void revertFeatures(INDArray features) {
        revertChecked(features);
    }

    @Override
    public void revertFeatures(INDArray features, INDArray featuresMask) {
        revertChecked(features);
    }

    @Override
    public void revertLabels(INDArray labels) {
        revertChecked(labels);
    }

    @Override
    public void revertLabels(INDArray labels, INDArray labelsMask) {
        revertChecked(labels);
    }

    @Override
    public void fitLabel(boolean fitLabels) {
        this.fittable = fitLabels;
    }

    @Override
    public boolean isFitLabel() {
        return this.fittable;
    }
}
