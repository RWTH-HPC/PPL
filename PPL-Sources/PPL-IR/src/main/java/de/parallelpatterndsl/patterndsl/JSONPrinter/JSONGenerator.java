package de.parallelpatterndsl.patterndsl.JSONPrinter;

import de.monticore.generating.GeneratorEngine;
import de.monticore.generating.GeneratorSetup;
import de.monticore.generating.templateengine.GlobalExtensionManagement;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;

import java.nio.file.Paths;

public class JSONGenerator {

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

    public JSONGenerator(AbstractPatternTree APT, String filename, int dataSplitSize, int patternSplitSize) {
        this.APT = APT;
        this.filename = filename;
        this.dataSplitSize = dataSplitSize;
        this.patternSplitSize = patternSplitSize;
    }

    /**
     * The function generating the source files.
     */
    public void generate() {
        //Instantiate the helper class
        JSONPrinterHelper pfHelper = new JSONPrinterHelper(APT,filename,dataSplitSize,patternSplitSize);
        GeneratorSetup gs = new GeneratorSetup();

        //define comment structure
        gs.setCommentStart("/*");
        gs.setCommentEnd("*/");
        gs.setTracing(false);

        //define output file ending
        gs.setDefaultFileExtension("json");

        //define the generator for the templates
        GlobalExtensionManagement management = new GlobalExtensionManagement();
        management.setGlobalValue("pfHelper", pfHelper);

        gs.setGlex(management);

        //generate the main source file
        GeneratorEngine engine = new GeneratorEngine(gs);
        engine.generateNoA("APTJSON.ftl", Paths.get(filename + ".json"), APT);

    }
}
