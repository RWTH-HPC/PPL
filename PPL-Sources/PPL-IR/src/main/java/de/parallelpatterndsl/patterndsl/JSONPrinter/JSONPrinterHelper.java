package de.parallelpatterndsl.patterndsl.JSONPrinter;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;

public class JSONPrinterHelper {

    /**
     * The abstract pattern tree to generate the C++ sources from.
     */
    private AbstractPatternTree APT;


    /**
     * The name of the output file.
     */
    private String filename;


    private int dataSplitSize;

    private int patternSplitSize;


    public JSONPrinterHelper(AbstractPatternTree APT, String filename, int dataSplitSize, int patternSplitSize) {
        this.APT = APT;
        this.filename = filename;
        this.dataSplitSize = dataSplitSize;
        this.patternSplitSize = patternSplitSize;
    }

    public String getFilename() {
        return filename;
    }

    /**
     * Generates the JSON of the APT.
     * @return
     */
    public String generateJSON() {
        APT2JSON apt2JSON = new APT2JSON(APT, filename, dataSplitSize, patternSplitSize);

        return apt2JSON.toString();

    }
}
