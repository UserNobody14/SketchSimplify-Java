package com.tweeneural.app;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.data.ImageWritable;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.net.URI;
import java.io.IOException;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.FilenameUtils;

public class OutputImageLabelGenerator implements PathLabelGenerator {

    protected long inHeight = -1;
    protected long inWidth = -1;
    protected long inChannels = 3;
    protected boolean usingMatrix = false;
    protected String inName = "input";
    protected String outName = "output";

    //Constructors.

    public OutputImageLabelGenerator(long inHeight, long inWidth, long inChannels) {
        this.inHeight = inHeight;
        this.inWidth = inWidth;
        this.inChannels = inChannels;
    }

    public OutputImageLabelGenerator(long inHeight, long inWidth) {
        this.inHeight = inHeight;
        this.inWidth = inWidth;
    }

    //Methods to add fields;

    public OutputImageLabelGenerator addIONames(String inName, String outName) {
        this.inName = inName;
        this.outName = outName;
        return this;
    }

    //Interface methods for pathlabelgenerator

    @Override
    public Writable getLabelForPath(String path) {
        try {
            return getWritableFromImage(getOutFile(path, inName, outName),
                                        inHeight, inWidth, inChannels);
        }
        catch(IOException e) {
            e.printStackTrace();
            return new Text("error");
        }

    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).toString());
    }

    @Override
    public boolean inferLabelClasses() {
        return false;
    }

    public static String getOutFile(String inpath, String input, String output) throws IOException {
        String filenam = FilenameUtils.getBaseName(inpath);
        return FilenameUtils.getFullPath(inpath)
            + filenam.replace(input, output) + "." +
            FilenameUtils.getExtension(inpath);
        //possibly replace use of FilenameUtils with plain 'file' or url classes?
    }

    public static String getOutFile(String inpath) throws IOException {
        return getOutFile(inpath, "input", "output");
    }

    public static Writable getWritableFromImage(String path, long h, long w, long c) {
        ImageLoader img = new ImageLoader(h, w, c);
        if (c == 1) {
            INDArray a = get4d(toGrayscale(img.toBgr(new File(path))), false);
            return new NDArrayWritable(a);
        }
        return new NDArrayWritable(get4d(img.toBgr(new File(path)), false));
    }
    public static INDArray toGrayscale(INDArray arr) {
        //make a new array of shape height x width
        INDArray x = arr.slice(0);
        x = x.add(arr.slice(1));
        x = x.add(arr.slice(2));
        x = x.divi(3);
        long[] sh = x.shape();
        return x.reshape(1, sh[0], sh[1]);
    }

    public static INDArray get4d(INDArray myarr, boolean duplicate) {
        long[] sh = myarr.shape();
        if (duplicate){
            INDArray newarr = myarr.dup();
            return newarr.reshape(1, sh[0], sh[1], sh[2]);
        }
        else {
            return myarr.reshape(1, sh[0], sh[1], sh[2]);
        }
    }

}
