package Benchmarks;

import com.google.common.base.Stopwatch;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.*;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.patternSplits.IOPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.ParallelPatternSplit;
import de.parallelpatterndsl.patterndsl.performance.simple.SimplePerformanceModel;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplitTable;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.se_rwth.commons.logging.Log;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterDescription;
import org.antlr.v4.runtime.misc.Pair;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class MultiFilterTest {

    public static final String BENCHMARK_PATH = "../../../benchmark/ppl/";

    public static final String CLUSTER_SPEC_PATH = "../../../benchmark/clusters/cluster_c18g.json";

    private AbstractPatternTree apt;

    private Network network;

    @Before
    public void init() {
        Log.init();
        Log.enableFailQuick(false);

        String modelname = "multi_filter";

        // Hardware description.
        this.network = ClusterDescription.parse(CLUSTER_SPEC_PATH);

        //Read model with the parser
        ModelPath modelPath = new ModelPath(Paths.get(BENCHMARK_PATH));
        GlobalScope symbolTable = PatternDSLScopeCreator.createGlobalScope(modelPath);
        Optional<ModuleSymbol> moduleSymbol = symbolTable.resolve(modelname, ModuleSymbol.KIND);
        Assert.assertTrue(moduleSymbol.isPresent());
        Assert.assertTrue(moduleSymbol.get().getModuleNode().isPresent());
        ASTModule ast = moduleSymbol.get().getModuleNode().get();


        /**
         * This is how you can add a single definition from one module to another!!!!
         */
        /*
        Optional<ModuleSymbol> moduleSymbol1 = symbolTable.resolve("model3", ModuleSymbol.KIND);
        Optional<Symbol> symbol = moduleSymbol1.get().getSpannedScope().getSubScopes().get(0).resolve("loopCall", FunctionSymbol.KIND);

        ASTModule ast1 = moduleSymbol1.get().getModuleNode().get();

        ast.getDefinitionList().add(ast1.getDefinition(1));
        symbolTable.getSubScopes().get(0).getSubScopes().get(0).add(symbol.get());
*/

        /**
         * Handle the inclusion of source code.
         */
        InclusionHandler inclusionHandler = new InclusionHandler(symbolTable,moduleSymbol.get(),BENCHMARK_PATH);
        inclusionHandler.handleIncludes();


        //Context condition testing
        CoCoSetup coCoSetup = new CoCoSetup();
        coCoSetup.Init();
        coCoSetup.Check(ast);

        /**
         * Handles the replacement of constant symbols.
         */
        ConstantReplacer constantHandler = new ConstantReplacer(ast, PPLExpressionPrinter.getInstance());
        constantHandler.replace();

        /**
         * Handles the extension of the shape variable in variable symbols.
         */
        VariableShapeHandler variableShapeHandler = new VariableShapeHandler(ast);
        variableShapeHandler.generateShapeForVariables();


        /**
         * Generate the APT.
         */
        AST2APT aptGenerator = new AST2APT(symbolTable,ast,10);
        this.apt = aptGenerator.generate();
    }

    @Test
    public void test() {
        Stopwatch watch = Stopwatch.createStarted();

        DataSplitTable.create(apt, 128);
        FlatAPT flatAPT = FlatAPTGenerator.generate(apt, 4096, 128);

        SimplePerformanceModel model = new SimplePerformanceModel(network, 0.0);
        MappingGenerator mappingGenerator = new MappingGenerator(flatAPT, network, model, 1);
        Pair<Mapping, Double> mapping = mappingGenerator.generate();
        Pair<Mapping, Double> base = baseMapping();

        try {
            mapping.a.toJSONFile("../../../benchmark/mappings/multi_filter.json", "Multi-Filter", "c18g", mapping.b);
            base.a.toJSONFile("../../../benchmark/mappings/multi_filter_base.json", "Multi-Filter", "c18g", base.b);
        } catch (IOException e) {
            e.printStackTrace();
       }

        watch.stop();
        System.out.println("Base: " + base.b + " opt: " + mapping.b);
        System.out.println("Test took: " + watch.elapsed(TimeUnit.MILLISECONDS) + " ms.");
    }

    private Pair<Mapping, Double> baseMapping() {
        DataSplitTable.create(apt, 128);
        FlatAPT flatAPT = FlatAPTGenerator.generate(apt, 2048, 128);
        SimplePerformanceModel model = new SimplePerformanceModel(network, 0.0);

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
                .collect(Collectors.toSet())){
            firstStep.assign(split, ioTeam);
        }
        score += model.evaluate(firstStep, base);
        base.push(firstStep);

        // 2. Step: Sobel
        Team one = new Team(cpu, socket1, socket1.getCores());
        Team two = new Team(cpu, socket2, socket2.getCores());
        StepMapping secondStep = new StepMapping(1);
        for (PatternSplit split: flatAPT.getSplits(1)
                .stream()
                .filter(s -> ((ParallelCallNode) s.getNode()).getFunctionIdentifier().equals("sobel_horizontal"))
                .collect(Collectors.toSet())) {
                if (split.getStartIndices()[0] < 2048) {
                    secondStep.assign(split, one);
                } else {
                    secondStep.assign(split, two);
                }
        }
        score += model.evaluate(secondStep, base);
        base.push(secondStep);

        // 3. Step: Prewitt
        Team b1 = new Team(cpu, socket1, socket1.getCores());
        Team b2 = new Team(cpu, socket2, socket2.getCores());
        StepMapping thirdStep = new StepMapping(2);
        for (PatternSplit split: flatAPT.getSplits(1)
                .stream()
                .filter(s -> ((ParallelCallNode) s.getNode()).getFunctionIdentifier().equals("prewitt_horizontal"))
                .collect(Collectors.toSet())) {
            if (split.getStartIndices()[0] < 2048) {
                thirdStep.assign(split, b1);
            } else {
                thirdStep.assign(split, b2);
            }
        }
        score += model.evaluate(thirdStep, base);
        base.push(thirdStep);

        // 4. Step: Laplacian
        Team d1 = new Team(cpu, socket1, socket1.getCores());
        Team d2 = new Team(cpu, socket2, socket2.getCores());
        StepMapping fourthStep = new StepMapping(3);
        for (PatternSplit split: flatAPT.getSplits(1)
                .stream()
                .filter(s -> ((ParallelCallNode) s.getNode()).getFunctionIdentifier().equals("laplacian"))
                .collect(Collectors.toSet())) {
            if (split.getStartIndices()[0] < 4096) {
                fourthStep.assign(split, d1);
            } else {
                fourthStep.assign(split, d2);
            }
        }
        score += model.evaluate(fourthStep, base);
        base.push(fourthStep);

        return new Pair<>(base, score);
    }
}
