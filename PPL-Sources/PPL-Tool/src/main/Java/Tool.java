import CMD.HandleComandLine;
import CMD.ToolOptions;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.APTInlineHandler;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl.FlatAPT;
import de.parallelpatterndsl.patterndsl.FlatAPTGenerator;
import de.parallelpatterndsl.patterndsl.MappingGenerator;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.AbstractSynchronizationModel;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.OPT2AMT.OPT2AMTGenerator;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.APTGraphvizGenerator;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.TreeDefinition;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplitTable;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.patternSplits.IOPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.ParallelPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.performance.PerformanceModel;
import de.parallelpatterndsl.patterndsl.performance.simple.SimplePerformanceModel;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterDescription;
import de.parallelpatterndsl.patterndsl.teams.Team;
import org.antlr.v4.runtime.misc.Pair;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Tool{



    public static void main(String[] args){
        //args = new String[]{"-s", "640000", "-d", "640000", "-n", "../../benchmark/ppl/cluster_c18.json", "-i", "../../benchmark/ppl/batch_classification.par"};
        HandleComandLine.parse(args);
        run();
    }

    public static void run() {
        /**
         * Define the trees to print.
         */
        boolean APT = (Boolean) ToolOptions.options.get(ToolOptions.APT).getValue();
        boolean Call = (Boolean) ToolOptions.options.get(ToolOptions.CALL).getValue();
        boolean Full = (Boolean) ToolOptions.options.get(ToolOptions.FULL).getValue();

        File cluster = new File((String) ToolOptions.options.get(ToolOptions.CLUSTERPATH).getValue());


        Network network = ClusterDescription.parse(cluster.getAbsolutePath());

        /**
         * Extract Paths from the input file.
         */
        File inputFile = new File((String) ToolOptions.options.get(ToolOptions.INPUTPATH).getValue());
        String sourcePath = inputFile.getPath().substring(0, inputFile.getPath().length() - inputFile.getName().length());
        String modelName = inputFile.getName().substring(0, inputFile.getName().length() - 4);

        File outputFile = new File((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue());
        String targetPath = outputFile.getPath().substring(0, outputFile.getPath().length() - outputFile.getName().length());
        String targetName = outputFile.getName().substring(0, outputFile.getName().length() - 4);


        /**
         * Parse the input file.
         */
        ModelPath modelPath = new ModelPath(Paths.get(sourcePath));
        GlobalScope symbolTable = PatternDSLScopeCreator.createGlobalScope(modelPath);
        Optional<ModuleSymbol> moduleSymbol = symbolTable.resolve(modelName, ModuleSymbol.KIND);

        ASTModule ast;
        if (moduleSymbol.isPresent()) {
            if (moduleSymbol.get().getModuleNode().isPresent()) {
                ast = moduleSymbol.get().getModuleNode().get();
            } else {
                throw new FileSystemNotFoundException("Module " + modelName + " is not correctly named!");
            }
        } else {
            throw new FileSystemNotFoundException("Module " + modelName + " does not exist!");
        }


        /**
         * Handle the inclusion of source code.
         */
        InclusionHandler inclusionHandler = new InclusionHandler(symbolTable, moduleSymbol.get(), sourcePath);
        inclusionHandler.handleIncludes();


        //Context condition testing
        CoCoSetup coCoSetup = new CoCoSetup();
        coCoSetup.Init();
        coCoSetup.Check(ast);

        System.out.println("Cocos finished!");
        /**
         * Handles the replacement of constant symbols.
         */
        ConstantReplacer constantHandler = new ConstantReplacer(ast, PPLExpressionPrinter.getInstance());
        constantHandler.replace();

        System.out.println("Constants finished!");

        /**
         * Handles the extension of the shape variable in variable symbols.
         */
        VariableShapeHandler variableShapeHandler = new VariableShapeHandler(ast);
        variableShapeHandler.generateShapeForVariables();

        /**
         * Generate the APT.
         */
        AST2APT aptGenerator = new AST2APT(symbolTable, ast, (Integer) ToolOptions.options.get(ToolOptions.RANDNAME).getValue());
        AbstractPatternTree abstractPatternTree = aptGenerator.generate();

        /**
         * Print the tree structure in graphviz.
         */
        APTGraphvizGenerator generator = new APTGraphvizGenerator();
        if (APT) {
            generator.generate(abstractPatternTree, "../" + targetPath + targetName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING) + ".py", TreeDefinition.PATTERN_NESTING, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING));
        }
        if (Call) {
            generator.generate(abstractPatternTree, "../" + targetPath + targetName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL) + ".py", TreeDefinition.CALL, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL));

        }
        if (Full) {
            generator.generate(abstractPatternTree, "../" + targetPath + targetName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE) + ".py", TreeDefinition.COMPLETE, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE));

        }

        /**
         * inline parallel patterns into the main for optimization.
         */
        APTInlineHandler inlineHandler = new APTInlineHandler(abstractPatternTree);
        inlineHandler.generateInlining();

        /**
         * generate the FlatAPT
         */
        DataSplitTable.create(abstractPatternTree, (Integer) ToolOptions.options.get(ToolOptions.DATASPLITSIZE).getValue());
        FlatAPT flatAPT = FlatAPTGenerator.generate(abstractPatternTree, (Integer) ToolOptions.options.get(ToolOptions.SPLITSIZE).getValue(), (Integer) ToolOptions.options.get(ToolOptions.DATASPLITSIZE).getValue());


        /**
         * Generate mapping
         */
        SimplePerformanceModel model = new SimplePerformanceModel(network, (Double) ToolOptions.options.get(ToolOptions.OVERLAP).getValue());
        MappingGenerator mappingGenerator = new MappingGenerator(flatAPT, network, model, (Integer) ToolOptions.options.get(ToolOptions.LOOKAHEAD).getValue());
        Pair<Mapping, Double> mapping = mappingGenerator.generate();

        /**
         * generate and print the base mapping if necessary
         */
        Pair<Mapping, Double> base;
        if ((Boolean) ToolOptions.options.get(ToolOptions.BASEMAPPING).getValue()) {
            base = baseMapping(flatAPT, model, network);
            try {
                base.a.toJSONFile(((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue()).split(".")[0] + ".json", modelName, cluster.getName(), base.b);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * print the optimization report
         */
        if ((Boolean) ToolOptions.options.get(ToolOptions.OPTREPORT).getValue()) {
            try {
                mapping.a.toJSONFile(((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue()).split(".")[0] + ".json", modelName, cluster.getName(), mapping.b);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * generate the AMT
         */
        OPT2AMTGenerator amtGenerator = new OPT2AMTGenerator(abstractPatternTree, flatAPT, mapping.a);
        AbstractMappingTree amt = amtGenerator.generate();

        AbstractMappingTree.setDefaultDevice(network.getNodes().get(0).getDevices().get(0));

        AbstractSynchronizationModel memoryModel = new AbstractSynchronizationModel(amt.getRoot(), network);
        amt.getRoot().setChildren(memoryModel.generateSyncAndTransfer());

        String outPath = (String) ((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue());
        String name = outPath.substring(0, outPath.length() - 4);
        return;
    }

    /**
     * Generates a base mapping for a Node with 2 CPU sockets.
     * @param flatAPT
     * @param model
     * @param network
     * @return
     */
    private static Pair<Mapping, Double> baseMapping(FlatAPT flatAPT, PerformanceModel model, Network network) {
        Node node = network.getNodes().get(0);
        Device cpu = node.getDevices().get(0);
        Processor socket1 = cpu.getProcessor().get(0);
        Processor socket2 = cpu.getProcessor().get(1);

        Mapping base = new Mapping();
        double score = 0.0;

        // 1. Step: IO
        Team ioTeam = new Team(cpu, socket1, 24);
        StepMapping firstStep = new StepMapping(0);
        for (PatternSplit split : flatAPT.getSplits(0)
                .stream()
                .filter(s -> s instanceof ParallelPatternSplit || s instanceof IOPatternSplit)
                .collect(Collectors.toSet())) {
            firstStep.assign(split, ioTeam);
        }
        score += model.evaluate(firstStep, base);
        base.push(firstStep);

        for (int i = 1; i < flatAPT.size() - 1; i++) {
            StepMapping stepMapping = new StepMapping(i);
            Iterator<PatternSplit> iter = flatAPT.getSplits(i).stream().sorted(Comparator.comparingInt(s -> s.getStartIndices()[0])).collect(Collectors.toCollection(LinkedHashSet::new)).iterator();

            Team one = new Team(cpu, socket1, socket1.getCores());
            Team two = new Team(cpu, socket2, socket2.getCores());
            for (int j = 0; j < flatAPT.getSplits(i).size(); j++) {
                if (i == 1 && j < flatAPT.getSplits(i).size() / 2 || i == 2 && j >= flatAPT.getSplits(i).size() / 2
                        || i == 3 && j < flatAPT.getSplits(i).size() / 2) {
                    stepMapping.assign(iter.next(), one);
                } else {
                    stepMapping.assign(iter.next(), two);
                }
            }

            score += model.evaluate(stepMapping, base);
            base.push(stepMapping);
        }

        return new Pair<>(base, score);
    }

}
