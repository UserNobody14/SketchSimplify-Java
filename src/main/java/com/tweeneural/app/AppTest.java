package com.tweeneural.app;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Color;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.net.URISyntaxException;
import java.nio.file.Paths;

/**
 * Hello world!
 *
 */

// home/benjamin/CodeProjects/Java/my-app/input1.bmp
// TODO: store the model + stats on disk + learn to stop server?!
// TODO: make up some free data w/ other sketchsimplifier & your own sketches.
// TODO: learn about spark and Mesos. Maybe get a spark & mesos cluster going. See if you can rent one?
    //TODO: Maybe later learn how to use jshell, and how to run it from intelliJ?
    //TODO: Make a feature to output images you created: use the existing neural style transfer one.
    //Remember its autodims now!

public class AppTest
{
    public static void main( String[] args )
    {
        String root = "file:///home/benjamin/CodeProjects/Java/my-app/resources/";
        NumFileSplitClean mytest = new NumFileSplitClean(root + "input%d.bmp", 1, 4);
        System.out.println(mytest.locations()[0].toString());
        utilPrintArrays(mytest.stringLocations());
        try {
            URI[] u = mytest.locations();
            //File newfil = new File(NumFileSplitClean.schemeClean(u[1].toString()));
            //BufferedImage exampleI = ImageIO.read(newfil);
            BufferedImage exampleI = ImageIO.read(new File(u[1].toString()));
            BufferedImage exampleIalso = ImageIO.read(new File(u[1].toString()));
            utilPrintArrays(u[1].toString(), new File(u[1].toString()).toString());
            System.out.println(exampleI.getHeight());
            System.out.println(exampleI.getWidth());

        }
        catch (IOException e) {
            System.out.println("IO screwup");
            e.printStackTrace();
        }

        try {
            //String root = "file:///home/benjamin/CodeProjects/Java/my-app/resources/";
            File rootf = new File(root);

            DataIterConfig d = new DataIterConfig(root + "frame4/", 6, 138)
                .typeImg("png")
                .channelsImg(3)
                .ioNames("input", "frame")
                    .autoDims();
            DataSetIterator drel = d.getData();
            //MultiLayerNetwork model = NnConfig.switchableNNLoad(0, d.getDims());
            int controller = 6;


            //doTests();
            MultiLayerNetwork model = NnConfig.loadModelIfExists(d, rootf, controller, false);
            //check for CuDNN
            LayerHelper h = model.getLayer(2).getHelper();    //Index 0: assume layer 0 is a ConvolutionLayer in this example
            System.out.println("Layer helper: " + (h == null ? null : h.getClass().getName()));
            //model.setListeners(setupGUI(root + "statStorage.dl4j"));
            model.fit(drel , 3);
            checkOutputs(model, drel, root);
            System.out.println("DONEDONEDONE SORTA");
            //test model saving
            model.save(NnConfig.switchableNNLoc(controller, rootf));
            //AppTest.doTests();
            //testDataSetCreator(root, infile);

        }

        catch (IOException e){
            e.printStackTrace();
            System.out.println( "\n" + "ERROR ERROR ERROR" );
        }


    }
    public static void testDataSetCreator(String muhString, String str2) {
        VtoVDataSetCreator kreator = new VtoVDataSetCreator()
            .nameIO("inputvid", "outputvid")
            .filType("bmp");
        //String altString = FilenameUtils.getPath(muhString);
        try {
            URI stru = new URI(str2);
            String er = Paths.get(str2).toUri().toString();
            String era3 = Paths.get(str2).toUri().getPath();
            String era4 = Paths.get(muhString).toUri().toString();
            String era5 = Paths.get(muhString).toUri().getPath();
            String era6 = Paths.get(new URI(muhString)).toUri().toString();
            String era7 = Paths.get(new URI(muhString)).toUri().getPath();
            //String era8 = Paths.get(schemeClean(str2)).toUri().toString();
            //String era9 = Paths.get(schemeClean(str2)).toUri().getPath();
            URI[] obt = VtoVDataSetCreator.inOutArrays("inputvid", er, "bmp", 2);
            for (URI uri: obt) {
                System.out.println(uri);
            }
            utilPrintArrays(er, era3,era4,era5,era6,era7);
            System.out.println(Paths.get(str2).toUri());
            System.out.println(era3);
        }
        catch (java.net.URISyntaxException e) {
            //whatever.
        }
        System.out.println(new File(str2).getPath());
        //System.out.println(FilenameUtils.getPath(str2));
        URI[] oqt = VtoVDataSetCreator.inOutArrays("inputvid", str2, "bmp", 2);
        for (URI uri: oqt) {
            System.out.println(uri);
        }
        URI[] oyt = VtoVDataSetCreator.inOutArrays("inputvid", muhString, "bmp", 2);
        DataSet newD = kreator.makeDataSetFromFolder(muhString, new long[]{3, 3, 3, 4});
        utilPrintArrays(newD.getFeatures(), newD.getLabels());
    }
    public static StatsListener storeStatsOnly(String store) {
        StatsStorage nstore = new FileStatsStorage(new File(NumFileSplitClean.schemeClean(store)));
        return new StatsListener(nstore);
    }
    public static StatsListener setupGUI(String store) {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc)
        //is to be stored. Here: store in memory.
        if (store != null) {
            StatsStorage nstore = new FileStatsStorage(new File(NumFileSplitClean.schemeClean(store)));
            uiServer.attach(nstore);
            return new StatsListener(nstore);
        }
        StatsStorage statsStorage = new InMemoryStatsStorage();
        //^^Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the
        //contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information
        //from the network, as it trains
        return new StatsListener(statsStorage);
    }

    public static void checkOutputs(MultiLayerNetwork m, DataSetIterator d, String root) {
        d.reset();
        NewPreProcessingScaler n = new NewPreProcessingScaler();
        DataSet next = d.next();
        INDArray outlabels2 = m.output(next.getFeatures(), false);
        //n.checkRanges(next.getFeatures(), next.getLabels(), outlabels2);
        n.revertChecked(outlabels2);
        //NewPreProcessingScaler g = new NewPreProcessingScaler().setTransformChecks(false);
        //g.checkRanges(next.getFeatures(), next.getLabels(), outlabels2);
        saveImage(outlabels2, 4, root);
        //utilPrintArrays(next.getFeatures(), outlabels2, next.getLabels());
    }
    private static BufferedImage imageFromINDArray(INDArray array) {
        long[] shape = array.shape();

        //This image from INDArray makes a grayscale image incase the incoming image has 1 channel.

        long height = shape[2];
        long width = shape[3];
        int redloc = 2, greenloc = 1;

        BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
        if (shape[1] == 1) {
            redloc = 0;
            greenloc = 0;
        }

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int red = array.getInt(0, redloc, y, x);
                int green = array.getInt(0, greenloc, y, x);
                int blue = array.getInt(0, 0, y, x);

                //handle out of bounds pixel values
                red = Math.min(red, 255);
                green = Math.min(green, 255);
                blue = Math.min(blue, 255);

                red = Math.max(red, 0);
                green = Math.max(green, 0);
                blue = Math.max(blue, 0);
                image.setRGB(x, y, new Color(red, green, blue).getRGB());
            }
        }
        return image;
    }

    private static void saveImage(INDArray combination, int iteration, String resource) {
        try {

            BufferedImage output = imageFromINDArray(combination);
            File file = new File(NumFileSplitClean.schemeClean(resource) + "iteration" + iteration + ".jpg");
            ImageIO.write(output, "jpg", file);
        }
        catch (IOException e) {
            utilPrintArrays("Error", "Error");
        }
    }
    public static void utilPrintArrays(INDArray... ia) {
        for (INDArray each : ia) {
            System.out.println("+=========+DividerDividerDividerDivider+===========");
            System.out.println(each);
        }
    }
    public static void utilPrintArrays(String... ia) {
        for (String each : ia) {
            System.out.println("+=========+DividerDividerDividerDivider+===========");
            System.out.println(each);
        }
    }
    public static void doTests() throws IOException {
        String infile = "/home/benjamin/CodeProjects/Java/my-app/resources/inputter1.bmp";
        ImageLoader img = new ImageLoader(4, 4, 3);
        ImageLoader img2 = new ImageLoader(4, 4, 1);
        //INDArray testrand1 = Nd4j.randn(new long[] {1, 3, 3, 3});
        String qfile = OutputImageLabelGenerator.getOutFile(infile);
        INDArray altmatrix = img.toBgr(new File (qfile));
        INDArray altmatrix2 = img.asMatrix(new File(qfile));
        INDArray altmatrix3 = OutputImageLabelGenerator.toGrayscale(altmatrix);
        INDArray altmatrix4 = img2.toBgr(new File (qfile));
        INDArray altmatrix5 = img2.asMatrix(new File(qfile));
        INDArray altmatrix6 = OutputImageLabelGenerator.toGrayscale(altmatrix4);
        //INDArray testrand = Nd4j.randn(new long[] {1, 3, 3, 3});
        utilPrintArrays(altmatrix, altmatrix2, altmatrix3, altmatrix4, altmatrix5, altmatrix6);

    }

}
