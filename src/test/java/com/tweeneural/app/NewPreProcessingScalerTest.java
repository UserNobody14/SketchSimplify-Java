package com.tweeneural.app;

import org.junit.Test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class NewPreProcessingScalerTest {

    @Test
    public void toFittable() {
        //to Fittable returns a NewPreprocessingScaler instance
        NewPreProcessingScaler n = new NewPreProcessingScaler().toFittable(true);
        assertTrue(n.isFitLabel());
        NewPreProcessingScaler n2 = new NewPreProcessingScaler().toFittable(false);
        assertFalse(n2.isFitLabel());
    }


    @Test
    public void revert() {
        INDArray lbls, ftrs, lbls2, ftrs2;
        lbls = Nd4j.zeros(new long[]{1, 4, 5, 5});
        ftrs = Nd4j.zeros(new long[]{1, 4, 5, 5});
        lbls.addi(0.95);
        ftrs.addi(0.95);
        DataSet ds = new DataSet(lbls, ftrs);
        NewPreProcessingScaler n = new NewPreProcessingScaler().toFittable(true);
        lbls2 = lbls.dup();
        ftrs2 = ftrs.dup();
        n.transform(ds);
        lbls = ds.getLabels();
        ftrs = ds.getFeatures();
        assertNotSame("Shouldnt be the same", ftrs, ftrs2);
        assertNotSame("Shouldnt be the same", lbls, lbls2);
    }

    @Test
    public void revertFeatures() {
        INDArray lbls, ftrs, ftrs2;
        lbls = Nd4j.zeros(new long[]{1, 4, 5, 5});
        ftrs = Nd4j.zeros(new long[]{1, 4, 5, 5});
        lbls.addi(250);
        ftrs.addi(0.95);
        NewPreProcessingScaler n = new NewPreProcessingScaler().toFittable(true);
        ftrs2 = ftrs.dup();
        n.revertFeatures(ftrs);
        //TODO add tests about transform & revertLables and all that.
        assertNotSame("Shouldnt be the same", ftrs, ftrs2);
    }

}