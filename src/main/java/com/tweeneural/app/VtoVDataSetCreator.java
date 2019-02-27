package com.tweeneural.app;

import org.datavec.image.loader.ImageLoader;
import org.datavec.api.split.NumberedFileInputSplit;

import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URI;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.nio.file.Paths;
//import javax.swing.filechooser.FileFilter;

public class VtoVDataSetCreator
{
    protected String inputName = "input";
    protected String outputName = "output";
    protected String fType = "bmp";

    public VtoVDataSetCreator() {
        //whatev
    }
    public VtoVDataSetCreator nameIO(String in, String out) {
        this.inputName = in;
        this.outputName = out;
        return this;
    }
    public VtoVDataSetCreator filType(String filtyper) {
        this.fType = filtyper;
        return this;
    }
    //convert URI array to File Array?
    public static URI[] inOutArrays(String n, String f, String filtype, int e) {
        //Fetch an array of files given the name of their rootfolder
        //additionally use io to see whether to use in or out methods.
        NumFileSplitClean mysplit = DataIterConfig.getSplit(f, n, filtype, e);
        return mysplit.locations();
    }

    public DataSet makeDataSetFromFolder(String f, long[] dimVideo) {
        int vidLength = (int) dimVideo[3];
        URI[] inVideo = inOutArrays(inputName, f, fType, vidLength);
        URI[] outVideo = inOutArrays(outputName, f, fType, vidLength);
        //next
        INDArray features = get5D(inVideo, dimVideo);
        INDArray labels = get5D(outVideo, dimVideo);
        return new DataSet(features, labels);

    }

    public static INDArray get5D(URI[] inputVid, long[] dimImg) {
        long iLength = (long) inputVid.length;
        INDArray testrand = Nd4j.zeros(1, iLength, dimImg[0], dimImg[1], dimImg[2]);
        ImageLoader img = new ImageLoader(dimImg[0], dimImg[1], dimImg[2]);
        for (long i = 0; i < iLength; i++)
        {
            INDArrayIndex[] subseter = new INDArrayIndex[]{NDArrayIndex.point(0),
                                                           NDArrayIndex.point(i),
                                                           NDArrayIndex.all(),
                                                           NDArrayIndex.all(),
                                                           NDArrayIndex.all()};
            INDArray bgrImage = img.toBgr(new File(new File(inputVid[(int) i])
                                                   .toString()));
            testrand.put(subseter, bgrImage);
        }
        return testrand.dup();

    }
}
