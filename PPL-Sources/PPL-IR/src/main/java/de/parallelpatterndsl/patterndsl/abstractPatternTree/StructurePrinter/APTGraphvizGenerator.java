package de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter;

import de.monticore.generating.GeneratorEngine;
import de.monticore.generating.GeneratorSetup;
import de.monticore.generating.templateengine.GlobalExtensionManagement;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;

import java.nio.file.Paths;
import java.util.ArrayList;

/**
 * The generator which takes the abstract pattern tree and generates the corresponding structure based on the given tree definition.
 */
public class APTGraphvizGenerator {

    public void generate(AbstractPatternTree APT, String filename, TreeDefinition definition, String name) {
        //Instantiate the helper class
        GraphvizGeneratorHelper pfHelper = new GraphvizGeneratorHelper(APT.getRoot(), definition, name);
        GeneratorSetup gs = new GeneratorSetup();

        //define comment structure
        gs.setCommentStart("/*");
        gs.setCommentEnd("*/");
        gs.setTracing(false);

        //define output file ending
        gs.setDefaultFileExtension("py");

        //define the generator for the templates
        GlobalExtensionManagement management = new GlobalExtensionManagement();
        management.setGlobalValue("pfHelper", pfHelper);

        gs.setGlex(management);

        //generate the main source file
        GeneratorEngine engine = new GeneratorEngine(gs);
        engine.generateNoA("GraphvizTemplate.ftl", Paths.get(filename));

    }
}
