package com.tweeneural.app;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.AdaDelta;

import java.io.File;
import java.io.IOException;

//TODO: perhaps remove the static members and give this class some more saved state?
//Particularly have it set a variable when it creates a new class, and then iterates through the files
//it saved already adding numbers to them until it finds a file that doesn't exist and writes that one.
//it only does this when told to save the new file.
//TODO: figure something out about the channels too?
//TODO: consider making an alt version of the simo serra net with separable convolutions?

public class NnConfig {
    protected int uH = 3;
    protected int uW = 3;
    protected int uC = 3;
    public NnConfig (){
        //
    }
    public static MultiLayerNetwork switchableNNLoad(int controller, long[] imageSize) {
        //TODO change this so that it checks if the file exists, and if not saves it automatically
        //That way the main program can just run this, and forget about it.
        //or maybe not?
        MultiLayerNetwork mynet;
        switch (controller) {
            case 0:
                mynet = nnConfig(12345, imageSize);
                break;
            case 1:
                mynet = nnConfig1(12345, imageSize);
                break;
            case 2:
                mynet = nnConfig2(12345, imageSize);
                break;
            case 3:
                mynet = nnConfig3Test(12345, imageSize);
                break;
            case 4:
                mynet = nnConfig1(14345, imageSize);
                break;
            case 5:
                mynet = nnConfig5(14345, imageSize);
                break;
            case 6:
                mynet = nnConfig5(14325, imageSize);
                break;
            default:
                mynet = nnConfig(123453, imageSize);
        }
        return mynet;
    }
    //TODO perhaps have a List of Pairs, and iterate over them to get the right one?

    public static File switchableNNLoc(int controller, File f) {
        String root = NumFileSplitClean.schemeClean(f.toString()) + "/";
        String v;
        switch (controller) {
            case 0:
                //mynet = nnConfig(12345, imageSize);
                v = root + "BasicNeuralNet_nnConfig.zip";
                break;
            case 1:
                v = root + "BasicPlussExtraLayer_nnConfig1.zip";
                break;
            case 2:
                v = root + "SimoSerraNet_nnConfig2.zip";
                break;
            case 3:
                v = root + "DeconvolveTestNet_nnConfig3Test.zip";
                break;
            case 4:
                v = root + "Neotest.zip";
                break;
            case 5:
                v = root + "SimoSerraBatchNormalized.zip";
                break;
            case 6:
                v = root + "BrandNewAnimNet.zip";
                break;
            default:
                v = root + "BasicNeuralNet_nnConfig.zip";
        }
        return new File(v);
    }
    public static MultiLayerNetwork loadModelIfExists(DataIterConfig d, File root, int controller, boolean b) throws IOException {
        File f = NnConfig.switchableNNLoc(controller, root);
        //Idea: make a new version of this that creates a new file no matter what, and then just loads whatever it saved?
        if (f.exists() | b) {
            System.out.println("Retrieved Model From File");
            return ModelSerializer.restoreMultiLayerNetwork(f, true);
        }
        else {
            System.out.println("Building Model in-program");
            return switchableNNLoad(controller, d.getDims());
        }
    }
    private static NeuralNetConfiguration.ListBuilder generalSettings(int seed) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .l2(0.005)
                .updater(new AdaDelta())
                .list();
    }

    private static NeuralNetConfiguration.ListBuilder deconvoStage(NeuralNetConfiguration.ListBuilder l, int q, int curr, int stage) {
        curr = curr + 3;
        int v = curr-3;
        stage = stage++;
        return l.layer(v + 1, convGeneric("down-convolution_lvl_" + (v + 1) + "_group_" + stage, q, 3, 2, 1))
                .layer(v + 2, convGeneric("flat-convolution_lvl_" + (v + 2) + "_group_" + stage, q, 3))
                .layer(v + 3, convGeneric("flat-convolution_lvl_" + (v + 3) + "_group_" + stage, q, 3));
    }

    private static NeuralNetConfiguration.ListBuilder upConvoStage(NeuralNetConfiguration.ListBuilder l, int q, int curr, int stage) {
        curr = curr + 3;
        int v = curr - 3;
        stage = stage++;
        return l.layer(v + 2, convGeneric("flat-convolution_lvl_" + (v + 2) + "_group_" + stage, q, 3))
                .layer(v + 3, convGeneric("flat-convolution_lvl_" + (v + 3) + "_group_" + stage, q, 3))
                .layer(v + 1, deConvolve("up-convolution_lvl_" + (v + 1) + "_group_" + stage, q, 3, 2));
    }

    private static MultiLayerNetwork nnConfig(int seed, long[] isize) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .l2(0.005) //changed
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //changed
                .updater(new AdaDelta()) //changed back?
                .list()
                .layer(0, convGenericInput("input", 3, 5, 3))//changed
                .layer(1, convGeneric("middle", 3, 3))
                .layer(2, convGeneric("middle2", 3, 3))//changed
                .layer(3, new CnnLossLayer.Builder(new LossMSE())
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutional(isize[0], isize[1], isize[2]))
                .build();
        return new MultiLayerNetwork(conf);
    }
    private static MultiLayerNetwork nnConfig1(int seed, long[] isize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new AdaDelta())
                .activation(Activation.IDENTITY)
                .list()
                .layer(0, convGenericInput("input", (int) isize[2], 20, 5))
                .layer(1, batchNorm("normtest1", 20))
                .layer(2, convGeneric("middle1", 3, 3))
                .layer(3, batchNorm("normtest2", 3))
                .layer(4, convGeneric("middle2", 3, 3))
                .layer(5, batchNorm("normtest3", 3))
                .layer(6, convGeneric("middle3", (int) isize[2], 3))
                .layer(7, new CnnLossLayer.Builder(new LossMSE())
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutional(isize[0], isize[1], isize[2]))
                .build();
        return new MultiLayerNetwork(conf);
    }
    private static MultiLayerNetwork nnConfig2(int seed, long[] isize) {
        //trying to produce the SimoSerra Network.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .list()
                //first group, input
                .layer(0, convGenericInput("input", (int) isize[2], 4, 1))
                .layer(1, convGeneric("down-convolution5x5_1_1", 48, 5, 2, 2))//stride 2 kernel 5 pad 2?
                .layer(2, convGeneric("flat-convolution3x3_1_2", 128, 3))
                .layer(3, convGeneric("flat-convolution3x3_1_3", 128, 3))
                //second group
                .layer(4, convGeneric("down-convolution3x3_2_1", 256, 3, 2, 1))
                .layer(5, convGeneric("flat-convolution3x3_2_2", 256, 3))
                .layer(6, convGeneric("flat-convolution3x3_2_3", 256, 3))
                //third group
                .layer(7, convGeneric("down-convolution3x3_3_1", 256, 3, 2, 1))
                .layer(8, convGeneric("flat-convolution3x3_3_2", 512, 3))
                .layer(9, convGeneric("flat-convolution3x3_3_3", 1024, 3))
                .layer(10, convGeneric("flat-convolution3x3_3_4", 1024, 3))
                .layer(11, convGeneric("flat-convolution3x3_3_5", 1024, 3))
                .layer(12, convGeneric("flat-convolution3x3_3_6", 1024, 3))
                .layer(13, convGeneric("flat-convolution3x3_3_7", 512, 3))
                .layer(14, convGeneric("flat-convolution3x3_3_8", 256, 3))
                //fourth group
                .layer(15, deConvolve("up-convolution4x4_4_1", 256, 4, 2))
                .layer(16, convGeneric("flat-convolution3x3_4_2", 128, 3))
                .layer(17, convGeneric("flat-convolution3x3_4_3", 128, 3))
                //fifth group
                .layer(18, deConvolve("up-convolution4x4_5_1", 128, 4, 2))
                .layer(19, convGeneric("flat-convolution3x3_5_2", 128, 3))
                .layer(20, convGeneric("flat-convolution3x3_5_3", 48, 3))
                //sixth group plus output.
                .layer(21, deConvolve("up-convolution4x4_6_1", 48, 4, 2))
                .layer(22, convGeneric("flat-convolution3x3_6_2", 24, 3))
                .layer(23, convGeneric("flat-convolution3x3_6_3", (int) isize[2], 3))
                .layer(24, new CnnLossLayer.Builder(new LossMSE())
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutional(isize[0], isize[1], isize[2]))
                .build();
        return new MultiLayerNetwork(conf);
    }
    private static MultiLayerNetwork nnConfig5(int seed, long[] isize) {
        //trying to produce the SimoSerra Network. NEW now with batchNorm goodness!
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.IDENTITY)
                .updater(new AdaDelta())
                .list()
                //first group, input
                .layer(0, convGenericInput("input", (int) isize[2], 4, 1))
                .layer(1, batchNorm("batch-normalize_1_0", 4))
                .layer(2, convGeneric("down-convolution5x5_1_1", 48, 5, 2, 2))//stride 2 kernel 5 pad 2?
                .layer(3, batchNorm("batch-normalize_1_1", 48))
                .layer(4, convGeneric("flat-convolution3x3_1_2", 128, 3))
                .layer(5, batchNorm("batch-normalize_1_2", 128))
                .layer(6, convGeneric("flat-convolution3x3_1_3", 128, 3))
                .layer(7, batchNorm("batch-normalize_1_3", 128))
                //second group
                .layer(8, convGeneric("down-convolution3x3_2_1", 256, 3, 2, 1))
                .layer(9, batchNorm("batch-normalize_2_1", 256))
                .layer(10, convGeneric("flat-convolution3x3_2_2", 256, 3))
                .layer(11, batchNorm("batch-normalize_2_2", 256))
                .layer(12, convGeneric("flat-convolution3x3_2_3", 256, 3))
                .layer(13, batchNorm("batch-normalize_2_3", 256))
                //third group
                .layer(14, convGeneric("down-convolution3x3_3_1", 256, 3, 2, 1))
                .layer(15, batchNorm("batch-normalize_3_1", 256))
                .layer(16, convGeneric("flat-convolution3x3_3_2", 512, 3))
                .layer(17, batchNorm("batch-normalize_3_2", 512))
                .layer(18, convGeneric("flat-convolution3x3_3_3", 1024, 3))
                .layer(19, batchNorm("batch-normalize_3_3", 1024))
                .layer(20, convGeneric("flat-convolution3x3_3_4", 1024, 3))
                .layer(21, batchNorm("batch-normalize_3_4", 1024))
                .layer(22, convGeneric("flat-convolution3x3_3_5", 1024, 3))
                .layer(23, batchNorm("batch-normalize_3_5", 1024))
                .layer(24, convGeneric("flat-convolution3x3_3_6", 1024, 3))
                .layer(25, batchNorm("batch-normalize_3_6", 1024))
                .layer(26, convGeneric("flat-convolution3x3_3_7", 512, 3))
                .layer(27, batchNorm("batch-normalize_3_7", 512))
                .layer(28, convGeneric("flat-convolution3x3_3_8", 256, 3))
                .layer(29, batchNorm("batch-normalize_3_8", 256))
                //fourth group
                .layer(30, deConvolve("up-convolution4x4_4_1", 256, 4, 2))
                .layer(31, batchNorm("batch-normalize_4_1", 256))
                .layer(32, convGeneric("flat-convolution3x3_4_2", 128, 3))
                .layer(33, batchNorm("batch-normalize_4_2", 256))
                .layer(34, convGeneric("flat-convolution3x3_4_3", 128, 3))
                .layer(35, batchNorm("batch-normalize_4_3", 256))
                //fifth group
                .layer(36, deConvolve("up-convolution4x4_5_1", 128, 4, 2))
                .layer(37, batchNorm("batch-normalize_5_1", 128))
                .layer(38, convGeneric("flat-convolution3x3_5_2", 128, 3))
                .layer(39, batchNorm("batch-normalize_5_2", 128))
                .layer(40, convGeneric("flat-convolution3x3_5_3", 48, 3))
                .layer(41, batchNorm("batch-normalize_5_3", 48))
                //sixth group plus output.
                .layer(42, deConvolve("up-convolution4x4_6_1", 48, 4, 2))
                .layer(43, batchNorm("batch-normalize_6_1", 48))
                .layer(44, convGeneric("flat-convolution3x3_6_2", 24, 3))
                .layer(45, batchNorm("batch-normalize_6_2", 24))
                .layer(46, convGeneric("flat-convolution3x3_6_3", (int) isize[2], 3))
                .layer(47, new CnnLossLayer.Builder(new LossMSE())
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutional(isize[0], isize[1], isize[2]))
                .build();
        return new MultiLayerNetwork(conf);
    }
    private static MultiLayerNetwork nnConfig3Test(int seed, long[] isize) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .l2(0.005) //changed
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //changed
                .updater(new AdaDelta()) //changed back?
                .list()
                .layer(0, convGenericInput("input", 3, 5, 3))//changed
                .layer(1, convGeneric("down by half 2", 3, 3, 2, 1))
                .layer(2, deConvolve("back up?", 3, 4, 2))//changed
                .layer(3, new CnnLossLayer.Builder(new LossMSE())
                        .activation(Activation.SIGMOID)
                        .build())
                //.pretrain(false)
                //.backprop(true)
                .setInputType(InputType.convolutional(isize[0], isize[1], isize[2]))
                .build();
        return new MultiLayerNetwork(conf);
    }
    private static ConvolutionLayer convGenericInput(String name, int in, int out, int kernel, int stride, int pad) {
        return new ConvolutionLayer.Builder(new int[]{kernel, kernel},
                new int[]{stride, stride},
                new int[]{pad, pad})
                .name(name)
                .activation(Activation.IDENTITY)
                .nIn(in)
                .nOut(out)
                .build();
    }
    private static ConvolutionLayer convGenericInput(String name, int in, int out, int kernel) {
        int stride =1;
        int pad = kernelSwitch(kernel);
        return convGenericInput(name, in, out, kernel, stride, pad);
    }
    private static ConvolutionLayer convGeneric(String name, int out, int kernel, int stride, int pad) {
        return new ConvolutionLayer.Builder(new int[]{kernel, kernel},
                new int[]{stride, stride},
                new int[]{pad, pad})
                .name(name)
                .activation(Activation.IDENTITY)
                .nOut(out)
                .build();
    }
    private static BatchNormalization batchNorm(String name, int out) {
        return new BatchNormalization.Builder()
                .name(name)
                .nOut(out)
                .activation(Activation.RELU)
                .build();
    }
    private static ConvolutionLayer convGeneric(String name, int out, int kernel) {
        int stride =1;
        int pad = kernelSwitch(kernel);
        return convGeneric(name, out, kernel, stride, pad);
    }
    private static Deconvolution2D deConvolve(String name, int out, int kernel, int stride) {
        return new Deconvolution2D.Builder(new int[]{kernel, kernel},
                new int[]{stride, stride},
                new int[]{kernelSwitch(kernel), kernelSwitch(kernel)})
                .activation(Activation.IDENTITY)
                .name(name)
                .nOut(out)
                .build();
    }
    private static int kernelSwitch(int kernel) {
        int pad =1;
        switch (kernel) {
            case 3:
                pad = 1;
                break;
            case 5:
                pad = 2;
                break;
            case 1:
                pad = 0;
                break;
            case 4:
                pad = 1;
                break;
            default:
                pad = 1;
        }
        return pad;
    }
}
//changes: none so far.
//possibilities: stride? zero padding?
//changed the zero padding on the convMiddle.
