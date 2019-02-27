package com.tweeneural.app;

import org.datavec.api.util.files.UriFromPathIterator;
import org.datavec.api.split.InputSplit;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NumFileSplitClean implements InputSplit{

    private final String baseString;
    private final int minIdx;
    private final int maxIdx;

    private static final Pattern p = Pattern.compile("\\%(0\\d)?d");

    /**
     * @param baseString String that defines file format. Must contain "%d", which will be replaced with
     *                   the index of the file, possibly zero-padded to x digits if the pattern is in the form %0xd.
     * @param minIdxInclusive Minimum index/number (starting number in sequence of files, inclusive)
     * @param maxIdxInclusive Maximum index/number (last number in sequence of files, inclusive)
     *                        @see {NumberedFileInputSplitTest}
     */
    public NumFileSplitClean(String baseString, int minIdxInclusive, int maxIdxInclusive) {
        Matcher m = p.matcher(baseString);
        if (baseString == null || !m.find()) {
            throw new IllegalArgumentException("Base String must match this regular expression: " + p.toString());
        }
        this.baseString = baseString;
        this.minIdx = minIdxInclusive;
        this.maxIdx = maxIdxInclusive;
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
        return maxIdx - minIdx + 1;
    }

    @Override
    public URI[] locations() {
        URI[] uris = new URI[(int) length()];
        int x = 0;
        try {
            for (int i = minIdx; i <= maxIdx; i++) {
                uris[x++] = uriClean(String.format(baseString, i));
            }
            return uris;
        }
        catch (URISyntaxException e) {
            for (int i = minIdx; i <= maxIdx; i++) {
                uris[x++] = Paths.get(String.format(baseString, i)).toUri();
            }
            return uris;
        }
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

    public static URI uriClean(String muhString) throws URISyntaxException {
        URI u = new URI(muhString);
        if (u.isAbsolute()) {
            return Paths.get(new URI(muhString)).toUri();
        }
        else {
            return Paths.get(muhString).toUri();
        }
    }

    public static String schemeClean(String muhString) {
        try
        {
            return uriClean(muhString).getPath();
        }
        catch (URISyntaxException e) {
            return "error error error";
        }
    }

//    }
//    private class NumberedFileIterator implements Iterator<String> {
//
//        private int currIdx;
//
//        private NumberedFileIterator() {
//            currIdx = minIdx;
//        }
//
//        @Override
//        public boolean hasNext() {
//            return currIdx <= maxIdx;
//        }
//
//        @Override
//        public String next() {
//            if (!hasNext()) {
//                throw new NoSuchElementException();
//            }
//            return String.format(baseString, currIdx++);
//        }
//
//        @Override
//        public void remove() {
//            throw new UnsupportedOperationException();
//        }
//    }
//

}
