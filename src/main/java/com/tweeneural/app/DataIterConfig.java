package com.tweeneural.app;

import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import javax.imageio.ImageIO;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URI;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


//TODO: Make this work with multiple root folders, base strings & indices?
//Or maybe implement that functionality in another class.

public class DataIterConfig {
    //fields
    protected int minibatches = 1;
    protected int numExamples = 4;
    protected long imgC = 3;
    protected long imgH = 3;
    protected long imgW = 3;
    protected String rootfolder = "file:///home/benjamin/CodeProjects/Java/my-app/";
    protected String inputImgName = "input";
    protected String outputImgName = "output";
    protected String fileType = "bmp";
    //public long[] idims = new long[]{3, 3, 3};

    public DataIterConfig (String root, int miniBatches, int numExamples) {
        //whatev
        this.rootfolder = root;
        this.minibatches = miniBatches;
        this.numExamples = numExamples;
    }
    //Methods to add Fields
    public DataIterConfig ioNames(String inName, String outName) {
        this.inputImgName = inName;
        this.outputImgName = outName;
        return this;
    }
    public DataIterConfig typeImg(String fileType) {
        this.fileType = fileType;
        return this;
    }
    public DataIterConfig sizeImg(long h, long w) {
        this.imgH = h;
        this.imgW = w;
        return this;
    }
    public DataIterConfig channelsImg(long c) {
        this.imgC = c;
        return this;
    }
    public long[] getDims() {
        return new long[]{imgH, imgW, imgC};
    }
    public DataIterConfig autoDims() {
        NumFileSplitClean nf = getSplit(rootfolder, inputImgName, fileType, numExamples);
        this.getImageDims(nf);
        return this;
    }
    public void getImageDims(NumFileSplitClean nf) {
        try {
            URI[] u = nf.locations();
            BufferedImage exampleI = ImageIO.read(new File(u[0]));
            this.imgH = exampleI.getHeight();
            this.imgW = exampleI.getWidth();
        }
        catch (IOException e) {
            System.out.println("IO screwup");
        }
    }
    public DataSetIterator getData() throws IOException {
        NumFileSplitClean nf = getSplit(rootfolder, inputImgName, fileType, numExamples);
        ImageRecordReader rr = labeledImageRead(imgH, imgW, imgC);
        //remember: it must be %d not d%
        rr.initialize(nf);
        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, 1)
            .regression(1)
            .preProcessor(new NewPreProcessingScaler().toFittable(true))
            .build();
        return iter;
    }
    public static NumFileSplitClean getSplit(String r, String in, String fType, int e) {
        //String er = new File(Paths.get(r).toUri()).toString();
        return new NumFileSplitClean(r + in + "%d" + "." + fType, 2, e);
    }
    public class MultipleFileNames {
        private NumFileSplitClean[] internalSplits;
        private List<NumFileSplitClean> splitsList = new ArrayList<>();
        public MultipleFileNames() {

        }
        public MultipleFileNames and(String baseName, int minIdx, int maxIdx) {
            String truname = rootfolder + baseName;
            splitsList.add(new NumFileSplitClean(truname, minIdx, maxIdx));
            return this;
        }
    }

    private ImageRecordReader labeledImageRead(long h, long w, long c) {
        return new ImageRecordReader(h, w, c, new OutputImageLabelGenerator(h, w, c)
                                     .addIONames(inputImgName, outputImgName));
    }

}
//changes thus far:
//make numberedfileinputsplit go from 1 to 1
//make RRdatasetiterator use a minibatch of 1
//imagepreprocessingscaler removed in favor of Newpreprocessingscaler.fittinglabels(true)
//above now changed to Newpreprocessingscaler(true) constructor
//maybe change it to typeImg channelsImg, and so on in the future? better name scheme?
//todo lookup Java REPL in the future.
