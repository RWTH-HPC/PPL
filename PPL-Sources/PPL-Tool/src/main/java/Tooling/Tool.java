package Tooling;

import CMD.HandleComandLine;
import CMD.ToolOptions;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.MappingTree.DebugCreator;
import de.parallelpatterndsl.patterndsl.MappingTree.GPUMaximizer.GPUMaximizer;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTAdditionalArgumentGeneration;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTDataTraceGenerator;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTParentAwareDataTraces;
import de.parallelpatterndsl.patterndsl.Preprocessing.*;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl.FlatAPT;
import de.parallelpatterndsl.patterndsl.FlatAPTGenerator;
import de.parallelpatterndsl.patterndsl.JSONPrinter.JSONGenerator;
import de.parallelpatterndsl.patterndsl.MappingGenerator;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.AbstractSynchronizationModel;
import de.parallelpatterndsl.patterndsl.MappingTree.MemoryAllocator.MemoryAllocator;
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
import de.parallelpatterndsl.patterndsl.generator.PatternDSLGenerator;
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
import de.se_rwth.commons.logging.Log;
import org.antlr.v4.runtime.misc.Pair;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Tool {



    public static void main(String[] args){
        HandleComandLine.parse(args);
        run();
    }

    public static void run() {

        String logName = "";
        /**
         * Define the trees to print.
         */
        boolean APT = (Boolean) ToolOptions.options.get(ToolOptions.APT).getValue();
        boolean Call = (Boolean) ToolOptions.options.get(ToolOptions.CALL).getValue();
        boolean Full = (Boolean) ToolOptions.options.get(ToolOptions.FULL).getValue();
        boolean JSON = (Boolean) ToolOptions.options.get(ToolOptions.JSON).getValue();

        int infoLevel = (Integer) ToolOptions.options.get(ToolOptions.INFO).getValue();

        boolean memoryMeasure = (Boolean) ToolOptions.options.get(ToolOptions.MEMORY).getValue();
        boolean timeMeasure = (Boolean) ToolOptions.options.get(ToolOptions.DURATION).getValue();
        boolean doInline = !((Boolean) ToolOptions.options.get(ToolOptions.INLINE).getValue());
        boolean debugMode = ((Boolean) ToolOptions.options.get(ToolOptions.EXPLICITMAPPING).getValue());

        boolean doUnroll = !(Boolean) ToolOptions.options.get(ToolOptions.UNROLL).getValue();

        ArrayList<Long> memory = new ArrayList<>();
        ArrayList<Long> duration = new ArrayList<>();
        long memoryReference = 0;
        long durationReference = 0;


        if (infoLevel == 2) {
            Log.initDEBUG();
        } else if (infoLevel == 1) {
            Log.init();
        } else if (infoLevel == 0) {
            Log.initWARN();
        } else {
            Log.init();
            Log.warn("Flag: -info/--InfoLevel only supports values 0,1 and 2. Using default Setup.");
        }

        File cluster = new File((String) ToolOptions.options.get(ToolOptions.CLUSTERPATH).getValue());


        Network network = ClusterDescription.parse(cluster.getAbsolutePath());

        boolean memTest = true;
        if (memTest) {
            System.gc();
            System.out.println("cli memory: " + Runtime.getRuntime().totalMemory());
        }


        /**
         * Extract Paths from the input file.
         */
        File inputFile = new File((String) ToolOptions.options.get(ToolOptions.INPUTPATH).getValue());
        String sourcePath = inputFile.getAbsoluteFile().getParent();
        String modelName = FilenameUtils.removeExtension(inputFile.getName());

        File outputFile = new File((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue());
        String targetPath = outputFile.getAbsoluteFile().getParent();
        String targetName = FilenameUtils.removeExtension(outputFile.getName());

        if (memoryMeasure) {
            System.gc();
            memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        }
        if (timeMeasure) {
            durationReference = System.currentTimeMillis();
        }
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

        if (memoryMeasure) {
            memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
        }
        if (timeMeasure) {
            duration.add(System.currentTimeMillis() - durationReference);
        }
        if (memTest) {
            System.gc();
            System.out.println("parsing memory: " + Runtime.getRuntime().totalMemory());
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

        Log.info("Cocos finished!", logName);
        /**
         * Handles the replacement of constant symbols.
         */
        ConstantReplacer constantHandler = new ConstantReplacer(ast, PPLExpressionPrinter.getInstance());
        constantHandler.replace();

        Log.info("Constants finished!", logName);

        /**
         * Handles the extension of the shape variable in variable symbols.
         */
        VariableShapeHandler variableShapeHandler = new VariableShapeHandler(ast);
        variableShapeHandler.generateShapeForVariables();


        if (memoryMeasure) {
            System.gc();
            memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        }
        if (timeMeasure) {
            durationReference = System.currentTimeMillis();
        }

        if (memTest) {
            System.gc();
            System.out.println("AST memory: " + Runtime.getRuntime().totalMemory());
        }
        /**
         * Generate the APT.
         */
        AST2APT aptGenerator = new AST2APT(symbolTable, ast, (Integer) ToolOptions.options.get(ToolOptions.RANDNAME).getValue());
        AbstractPatternTree abstractPatternTree = aptGenerator.generate();


        /**
         * Create copies of functions to avoid multiple generations of additional parameters parallel calls
         */
        CallStructure callStructure = new CallStructure(abstractPatternTree);
        callStructure.generate();


        /**
         * inline parallel patterns into the main for optimization.
         */

        if (doInline) {
            boolean scopeState = true;
            boolean functionState = true;
            boolean unrollState = true;
            while ((scopeState || functionState || unrollState)) {
                APTScopeInlineHandler scopeInlineHandler = new APTScopeInlineHandler(abstractPatternTree);
                scopeState = scopeInlineHandler.generateInlining();

                APTInlineHandler inlineHandler = new APTInlineHandler(abstractPatternTree);
                functionState = inlineHandler.generateInlining();

                if (doUnroll) {
                    APTLoopUnrollHandler loopUnrollHandler = new APTLoopUnrollHandler(abstractPatternTree);
                    unrollState = loopUnrollHandler.generate();
                }
                System.out.println("Current number of nodes: " + abstractPatternTree.getRoot().getChildren().size());
            }
        }

        /**
         * Marks the function nodes that are still necessary for the full traversal
         */
        FunctionMarker marker = new FunctionMarker(abstractPatternTree);
        marker.generate();


        /**
         * APT postprocessing: Data Trace and additional argument generation
         */
        APTParentAwareDataTraces aptDataTraceGenerator = new APTParentAwareDataTraces();
        aptDataTraceGenerator.generate();

        APTAdditionalArgumentGeneration aptAdditionalArgumentGeneration = new APTAdditionalArgumentGeneration();
        aptAdditionalArgumentGeneration.generate();


        if (memoryMeasure) {
            memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
        }
        if (timeMeasure) {
            duration.add(System.currentTimeMillis() - durationReference);
        }

        /**
         * Print the tree structure in graphviz.
         */
        APTGraphvizGenerator generator = new APTGraphvizGenerator();
        if (APT) {
            generator.generate(abstractPatternTree, FilenameUtils.concat(targetPath, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING) + ".py"), TreeDefinition.PATTERN_NESTING, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING));
        }
        if (Call) {
            generator.generate(abstractPatternTree, FilenameUtils.concat(targetPath, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL) + ".py"), TreeDefinition.CALL, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL));

        }
        if (Full) {
            generator.generate(abstractPatternTree, FilenameUtils.concat(targetPath, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE) + ".py"), TreeDefinition.COMPLETE, targetName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE));

        }

        /**
         * Print APT as a JSON.
         */
        if (JSON) {
            JSONGenerator apt2json = new JSONGenerator(abstractPatternTree, FilenameUtils.concat(targetPath, targetName), (Integer) ToolOptions.options.get(ToolOptions.DATASPLITSIZE).getValue(), (Integer) ToolOptions.options.get(ToolOptions.SPLITSIZE).getValue());
            apt2json.generate();
        }

        /**
         * Define the default execution unit
         */
        AbstractMappingTree.setDefaultDevice(network.getNodes().get((Integer) ToolOptions.options.get(ToolOptions.DEFAULTNODE).getValue()).getDevices().get((Integer) ToolOptions.options.get(ToolOptions.DEFAULTDEVICE).getValue()));
        if (!AbstractMappingTree.getDefaultDevice().getType().equals("CPU")) {
            Log.error("The default device must be a CPU.");
            System.exit(1);
        }
        AbstractMappingTree AMT;

        if (debugMode) {
            /**
             * Debug AMT
             */
            DebugCreator debugCreator = new DebugCreator(abstractPatternTree, network);
            AMT = debugCreator.generate();
        } else {

            if (memTest) {
                System.gc();
                System.out.println("APT memory: " + Runtime.getRuntime().totalMemory());
            }
            /**
             * generate the FlatAPT
             */
            DataSplitTable.create(abstractPatternTree, (Integer) ToolOptions.options.get(ToolOptions.DATASPLITSIZE).getValue());
            FlatAPT flatAPT = FlatAPTGenerator.generate(abstractPatternTree, (Integer) ToolOptions.options.get(ToolOptions.SPLITSIZE).getValue(), (Integer) ToolOptions.options.get(ToolOptions.DATASPLITSIZE).getValue());
            if (memTest) {
                System.gc();
                System.out.println("FlatAPT memory: " + Runtime.getRuntime().totalMemory());
            }

            /**
             * Generate mapping
             */
            SimplePerformanceModel model = new SimplePerformanceModel(network, (Double) ToolOptions.options.get(ToolOptions.OVERLAP).getValue());
            if (memoryMeasure) {
                System.gc();
                memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            }
            if (timeMeasure) {
                durationReference = System.currentTimeMillis();
            }
            MappingGenerator mappingGenerator = new MappingGenerator(flatAPT, network, model, (Integer) ToolOptions.options.get(ToolOptions.LOOKAHEAD).getValue(), (Integer) ToolOptions.options.get(ToolOptions.TIMELIMIT).getValue());
            Pair<Mapping, Double> mapping = mappingGenerator.generate();
            if (memoryMeasure) {
                memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
            }
            if (timeMeasure) {
                duration.add(System.currentTimeMillis() - durationReference);
            }
            if (memTest) {
                System.gc();
                System.out.println("OPT memory: " + Runtime.getRuntime().totalMemory());
            }

            /**
             * generate and print the base mapping if necessary
             */
            Pair<Mapping, Double> base;
            if ((Boolean) ToolOptions.options.get(ToolOptions.BASEMAPPING).getValue()) {
                base = baseMapping(flatAPT, model, network);
                try {
                    base.a.toJSONFile(FilenameUtils.concat(targetPath, "base_" + targetName + ".json"), modelName, cluster.getName(), base.b);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            /**
             * print the optimization report
             */
            if ((Boolean) ToolOptions.options.get(ToolOptions.OPTREPORT).getValue()) {
                try {
                    mapping.a.toJSONFile(FilenameUtils.concat(targetPath, targetName + ".json"), modelName, cluster.getName(), mapping.b);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            /**
             * generate the AMT
             */
            if (memoryMeasure) {
                System.gc();
                memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            }
            if (timeMeasure) {
                durationReference = System.currentTimeMillis();
            }
            OPT2AMTGenerator amtGenerator = new OPT2AMTGenerator(abstractPatternTree, flatAPT, mapping.a);
            AMT = amtGenerator.generate();

            if (memoryMeasure) {
                memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
            }
            if (timeMeasure) {
                duration.add(System.currentTimeMillis() - durationReference);
            }
        }

        /**
         * MPI memory management
         */
        if (memoryMeasure) {
            System.gc();
            memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        }
        if (timeMeasure) {
            durationReference = System.currentTimeMillis();
        }
        AbstractSynchronizationModel memoryModel = new AbstractSynchronizationModel(AMT.getRoot(), network);
        AMT.getRoot().setChildren(memoryModel.generateSyncAndTransfer());

        /**
         * GPU Memory allocation
         */
        MemoryAllocator memoryAllocator = new MemoryAllocator(AMT.getRoot());
        AMT.getRoot().setChildren(memoryAllocator.addAllocationNodes());
        if (memoryMeasure) {
            memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
        }
        if (timeMeasure) {
            duration.add(System.currentTimeMillis() - durationReference);
        }
        if (memTest) {
            System.gc();
            System.out.println("AMT memory: " + Runtime.getRuntime().totalMemory());
        }

        GPUMaximizer maximizer = new GPUMaximizer(AMT);
        maximizer.maximize();

        /**
         * Code generation
         */
        if (memoryMeasure) {
            System.gc();
            memoryReference = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        }
        if (timeMeasure) {
            durationReference = System.currentTimeMillis();
        }
        PatternDSLGenerator cppGenerator = new PatternDSLGenerator(AMT, targetPath, targetName,network, "" + ToolOptions.options.get(ToolOptions.GPUTHREAD).getValue());
        cppGenerator.generate();
        if (memoryMeasure) {
            memory.add(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryReference);
        }
        if (timeMeasure) {
            duration.add(System.currentTimeMillis() - durationReference);
        }
        if (memTest) {
            System.gc();
            System.out.println("generate memory: " + Runtime.getRuntime().totalMemory());
        }


        /**
         * Output memory and time measures
         */
        if (timeMeasure) {
            write2file(duration.get(0), targetPath, "Parse_Time.txt");
            write2file(duration.get(1), targetPath, "APT_Time.txt");
            write2file(duration.get(2), targetPath, "Mapping_Time.txt");
            write2file(duration.get(3), targetPath, "AMT_Time.txt");
            write2file(duration.get(4), targetPath, "Post_Processing_Time.txt");
            write2file(duration.get(5), targetPath, "Generation_Time.txt");
        }

        if (memoryMeasure) {
            write2file(memory.get(0), targetPath, "Parse_Memory.txt");
            write2file(memory.get(1), targetPath, "APT_Memory.txt");
            write2file(memory.get(2), targetPath, "Mapping_Memory.txt");
            write2file(memory.get(3), targetPath, "AMT_Memory.txt");
            write2file(memory.get(4), targetPath, "Post_Processing_Memory.txt");
            write2file(memory.get(5), targetPath, "Generation_Memory.txt");
        }
        return;
    }

    private static void write2file(Long value, String sourcePath, String filename) {
        File file = new File(sourcePath, filename);
        try {
            if (file.createNewFile()) {
                System.out.println("File created: " + file.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter(file, true);
            myWriter.write(value + "\n");
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

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
            Iterator<PatternSplit> iter = flatAPT.getSplits(i).stream().sorted(Comparator.comparingLong(s -> s.getStartIndices()[0])).collect(Collectors.toCollection(LinkedHashSet::new)).iterator();

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

    private static String getPath(boolean output) {
        String path;
        if (output) {
            path = ((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue());
        } else {
            path = ((String) ToolOptions.options.get(ToolOptions.INPUTPATH).getValue());
        }

        int last = path.lastIndexOf("/");
        if (last < path.length() && last >= 0) {
            return path.substring(0, last);
        }

        last = path.lastIndexOf("\\");
        if (last < path.length() && last >= 0) {
            return path.substring(0, last);
        }

        return "";
    }

}
