package CmdParserTest;

import CMD.HandleComandLine;
import CMD.ToolOptions;
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
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Paths;
import java.util.Optional;

public class ParserTest{

    @BeforeClass
    public static void init() {
        Log.init();
        Log.enableFailQuick(false);
    }

    @Test
    public void parsing() {
        String[] args = {"-i", "../../../rodinia-ppl/lud/lud.par", "-APT", "--network=test.json", "-s", "12", "-opt"};
        HandleComandLine.parse(args);
        run();
    }

    public void run() {
        System.out.println(ToolOptions.options.get(ToolOptions.SPLITSIZE).getValue());
        boolean APT = false;
        boolean Call = false;
        boolean Full = false;

        long Parse_time = -1;
        long AST_Time = -1;
        long APT_Time = -1;
        long PT_Print = -1;
        long CallT_Print = -1;
        long FullT_Print = -1;
        long AST_Size = -1;
        long APT_Size = -1;

        if ((Boolean) ToolOptions.options.get(ToolOptions.APT).getValue()){
            APT = true;
        } else if ((Boolean) ToolOptions.options.get(ToolOptions.CALL).getValue()) {
            Call = true;
        } else if ((Boolean) ToolOptions.options.get(ToolOptions.FULL).getValue()) {
            Full = true;
        }
        File outputFile = new File((String) ToolOptions.options.get(ToolOptions.OUTPUTPATH).getValue());
        File inputFile = new File((String) ToolOptions.options.get(ToolOptions.INPUTPATH).getValue());

        String sourcePath = inputFile.getPath().substring(0, inputFile.getPath().length() - inputFile.getName().length());
        String modelName = inputFile.getName().substring(0, inputFile.getName().length() - 4);

        System.gc();
        long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long start = System.currentTimeMillis();

        //Read model with the parser
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
        AST_Size = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - usedMemory;

        Parse_time = System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
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
        AST_Time = System.currentTimeMillis() - start;

        /**
         * Generate the APT.
         */
        System.gc();
        long used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        start = System.currentTimeMillis();
        AST2APT aptGenerator = new AST2APT(symbolTable, ast, 10);
        AbstractPatternTree abstractPatternTree = aptGenerator.generate();
        APT_Time = System.currentTimeMillis() - start;
        APT_Size = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - used;
        /**
         * Print the tree structure in graphviz.
         */
        APTGraphvizGenerator generator = new APTGraphvizGenerator();
        if (APT) {
            start = System.currentTimeMillis();
            generator.generate(abstractPatternTree, "../" + sourcePath + modelName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING) + ".py", TreeDefinition.PATTERN_NESTING, modelName + TreeDefinition.getTreeNames().get(TreeDefinition.PATTERN_NESTING));
            PT_Print = System.currentTimeMillis() - start;
        }
        if (Call) {
            start = System.currentTimeMillis();
            generator.generate(abstractPatternTree, "../" + sourcePath + modelName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL) + ".py", TreeDefinition.CALL, modelName + TreeDefinition.getTreeNames().get(TreeDefinition.CALL));
            CallT_Print = System.currentTimeMillis() - start;
        }
        if (Full) {
            start = System.currentTimeMillis();
            generator.generate(abstractPatternTree, "../" + sourcePath + modelName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE) + ".py", TreeDefinition.COMPLETE, modelName + TreeDefinition.getTreeNames().get(TreeDefinition.COMPLETE));
            FullT_Print = System.currentTimeMillis() - start;
        }



        write2file(Parse_time, sourcePath, "Parse_Time.txt");
        write2file(AST_Time, sourcePath, "AST_Extension.txt");
        write2file(APT_Time, sourcePath, "APT_Generation.txt");
        write2file(PT_Print, sourcePath, "APT_Print.txt");
        write2file(CallT_Print, sourcePath, "Call_Tree_Print.txt");
        write2file(FullT_Print, sourcePath, "Full_Tree_Print.txt");
        write2file(APT_Size, sourcePath, "APT_Size.txt");
        write2file(AST_Size, sourcePath, "AST_Size.txt");


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
}
