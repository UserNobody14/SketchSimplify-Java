package com.tweeneural.app;

import org.datavec.api.split.InputSplit;

import java.io.*;
import java.net.URI;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

//import org.apache.commons.lang3.ArrayUtils;

public class MultiSplit implements InputSplit {
    private final NumFileSplitClean[] internalSplits;

    /**
     * @param mySplits Group of file splits that MultiSplit then concatenates together and exposes as a 'virtual split'.
     */

    public MultiSplit(NumFileSplitClean... mySplits) {
        this.internalSplits = mySplits;
    }

    public MultiSplit(List<NumFileSplitClean> mySplits) {
        this(mySplits.toArray(new NumFileSplitClean[0]));
    }

    @Override
    public boolean canWriteToLocation(URI location) {
        return location.isAbsolute();
    }

    @Override
    public String addNewLocation() {
        return null;
    }

    @Override
    public String addNewLocation(String location) {
        return null;
    }

    @Override
    public void updateSplitLocations(boolean reset) {
        //no-op (locations() is dynamic)
    }

    @Override
    public boolean needsBootstrapForWrite() {
        return locations() == null ||
                locations().length < 1
                || locations().length == 1 && !locations()[0].isAbsolute();
    }

    @Override
    public void bootStrapForWrite() {
        if(locations().length == 1 && !locations()[0].isAbsolute()) {
            File parentDir = new File(locations()[0]);
            File writeFile = new File(parentDir,"write-file");
            try {
                writeFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }


        }
    }

    @Override
    public OutputStream openOutputStreamFor(String location) throws Exception {
        FileOutputStream ret = location.startsWith("file:") ? new FileOutputStream(new File(URI.create(location))):
                new FileOutputStream(new File(location));
        return ret;
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        FileInputStream fileInputStream = new FileInputStream(location);
        return fileInputStream;
    }

    @Override
    public long length() {
        return locations().length;
    }

    @Override
    public URI[] locations() {
        URI[] mylocstack = null;
        for (NumFileSplitClean n: this.internalSplits) {
            mylocstack = locations(n, mylocstack);
        }
        return mylocstack;
    }
    public URI[] locations(NumFileSplitClean thisSplit, URI[] locationStack) {
        if (locationStack == null) {
            return thisSplit.locations();
        }
        URI[] locationsToAdd = thisSplit.locations();
        URI[] newLocations = new URI[locationsToAdd.length + locationStack.length];
        for (int i = 0;i < locationsToAdd.length; i++) {
            newLocations[i + locationStack.length] = locationsToAdd[i];
        }
        for (int i = 0;i < locationStack.length; i++) {
            newLocations[i + locationStack.length] = locationsToAdd[i];
        }
        //Use the above to make a less dependency centric version in future.
        return newLocations;
    }
    public String[] stringLocations() {
        String[] allLocs = new String[locations().length];
        for (int i = 0; i < locations().length; i++) {
            allLocs[i] = locations()[i].toString();
        }
        return allLocs;
    }

    @Override
    public Iterator<URI> locationsIterator() {
        Iterator<URI> thingyNew = Arrays.asList(locations()).iterator();
        return thingyNew;
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        Iterator<String> thingyNew = Arrays.asList(stringLocations()).iterator();
        return thingyNew;
    }

    @Override
    public void reset() {
        //No op
    }

    @Override
    public boolean resetSupported() {
        return true;
    }
}
