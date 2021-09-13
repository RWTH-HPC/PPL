package GraphPrinter;

import Generator.RodiniaBenchmarks;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.APTGraphvizGenerator;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.TreeDefinition;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.se_rwth.commons.logging.Log;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Optional;


public class GraphvizGeneratorRodinia {

    @BeforeClass
    public static void init() {
        Log.init();
        Log.enableFailQuick(false);
    }

    private ArrayList<Long> Parse_time = new ArrayList<>();
    private ArrayList<Long> AST_Time = new ArrayList<>();
    private ArrayList<Long> APT_Time = new ArrayList<>();
    private ArrayList<Long> PT_Print = new ArrayList<>();
    private ArrayList<Long> CallT_Print = new ArrayList<>();
    private ArrayList<Long> FullT_Print = new ArrayList<>();
    private ArrayList<Long> AST_Size = new ArrayList<>();
    private ArrayList<Long> APT_Size = new ArrayList<>();


    @ParameterizedTest
    @EnumSource(RodiniaBenchmarks.class)
    public void testGeneratorModuleWithDataStructures(RodiniaBenchmarks rodiniaBenchmarks) {
        String sourcePath = RodiniaBenchmarks.paths.get(rodiniaBenchmarks).getPath();
        String modelname = RodiniaBenchmarks.paths.get(rodiniaBenchmarks).getName();

        System.gc();
        long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        ArrayList<Long> timer = new ArrayList<>();
        timer.add(System.currentTimeMillis());

        //Read model with the parser
        ModelPath modelPath = new ModelPath(Paths.get(sourcePath));
        GlobalScope symbolTable = PatternDSLScopeCreator.createGlobalScope(modelPath);
        Optional<ModuleSymbol> moduleSymbol = symbolTable.resolve(modelname, ModuleSymbol.KIND);
        Assert.assertTrue(moduleSymbol.isPresent());
        Assert.assertTrue(moduleSymbol.get().getModuleNode().isPresent());

        ASTModule ast = moduleSymbol.get().getModuleNode().get();
        long astSize = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - usedMemory;

        timer.add(System.currentTimeMillis());

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
        timer.add(System.currentTimeMillis());

        /**
         * Generate the APT.
         */
        System.gc();
        long used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        AST2APT aptGenerator = new AST2APT(symbolTable, ast, 10);
        AbstractPatternTree abstractPatternTree = aptGenerator.generate();
        timer.add(System.currentTimeMillis());
        long aptSize = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - used;
        /**
         * Print the tree structure in graphviz.
         */
        ArrayList<TreeDefinition> definitions = new ArrayList<>();
        definitions.add(TreeDefinition.CALL);
        definitions.add(TreeDefinition.COMPLETE);
        definitions.add(TreeDefinition.PATTERN_NESTING);
        APTGraphvizGenerator generator = new APTGraphvizGenerator();
        for (TreeDefinition definition : definitions) {
            generator.generate(abstractPatternTree, "../" + sourcePath + modelname + TreeDefinition.getTreeNames().get(definition) + ".py", definition, modelname + TreeDefinition.getTreeNames().get(definition));
            timer.add(System.currentTimeMillis());
        }


        Parse_time.add(timer.get(1) - timer.get(0));
        AST_Time.add(timer.get(2) - timer.get(1));
        APT_Time.add(timer.get(3) - timer.get(2));
        CallT_Print.add(timer.get(4) - timer.get(3));
        FullT_Print.add(timer.get(5) - timer.get(4));
        PT_Print.add(timer.get(6) - timer.get(5));

        AST_Size.add(astSize);
        APT_Size.add(aptSize);


        write2file(Parse_time.get(0), sourcePath, "Parse_Time.txt");
        write2file(AST_Time.get(0), sourcePath, "AST_Extension.txt");
        write2file(APT_Time.get(0), sourcePath, "APT_Generation.txt");
        write2file(PT_Print.get(0), sourcePath, "APT_Print.txt");
        write2file(CallT_Print.get(0), sourcePath, "Call_Tree_Print.txt");
        write2file(FullT_Print.get(0), sourcePath, "Full_Tree_Print.txt");
        write2file(APT_Size.get(0), sourcePath, "APT_Size.txt");
        write2file(AST_Size.get(0), sourcePath, "AST_Size.txt");
        

        return;

    }

    private void write2file(Long value, String sourcePath, String filename) {
        File file = new File(sourcePath + filename);
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
            FileWriter myWriter = new FileWriter(sourcePath + filename, true);
            myWriter.write(value + "\n");
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

    }

    private long avg_big(ArrayList<Long> list) {
        long result = 0;
        for (long x : list) {
            result += x / list.size();
        }
        return result;
    }

    private long avg_small(ArrayList<Long> list) {
        long result = 0;
        for (long x : list) {
            result += x;
        }
        return result / list.size();
    }
}