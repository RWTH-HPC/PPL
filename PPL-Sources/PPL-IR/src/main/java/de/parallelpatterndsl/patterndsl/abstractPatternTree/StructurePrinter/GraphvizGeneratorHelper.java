package de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter;


import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter.CallTreePrinter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter.CompleteTreePrinter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MainNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter.PatternNestingTreePrinter;

/**
 * Helper class, that generates the correct tree for the given parameter.
 */
public class GraphvizGeneratorHelper {

    private MainNode mainNode;

    private TreeDefinition definition;

    private String name;

    public GraphvizGeneratorHelper(MainNode mainNode, TreeDefinition definition, String name) {
        this.mainNode = mainNode;
        this.definition = definition;
        this.name = name;
    }

    public String generateTree() {
        if (definition == TreeDefinition.CALL) {
            return generateCallTree();
        } else if (definition == TreeDefinition.COMPLETE) {
            return generateFullTree();
        } else if (definition == TreeDefinition.PATTERN_NESTING) {
            return generatePatternNesting();
        }
        return "Incorrect Tree Definition.";
    }

    public String getName() {
        return this.name;
    }

    private String generateCallTree() {
        CallTreePrinter printer = new CallTreePrinter();
        return printer.generate(mainNode);
    }

    private String generateFullTree(){
        CompleteTreePrinter printer = new CompleteTreePrinter();
        return printer.generate(mainNode);
    }

    private String generatePatternNesting() {
        PatternNestingTreePrinter printer = new PatternNestingTreePrinter();
        return printer.generate(mainNode);
    }


}
