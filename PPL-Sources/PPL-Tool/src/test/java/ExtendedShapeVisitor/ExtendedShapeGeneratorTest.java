package ExtendedShapeVisitor;

import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.se_rwth.commons.logging.Log;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Optional;


public class ExtendedShapeGeneratorTest {

    public static final String MODEL_SOURCE_PATH = "./src/test/resources/ExtendedShapeVisitor/";

    @BeforeClass
    public static void init() {
        Log.init();
        Log.enableFailQuick(false);
    }

    @Test
    public void testGeneratorModuleWithDataStructures() {
        String modelname = "model1";
        String sourcePath = MODEL_SOURCE_PATH;

        //Read model with the parser
        ModelPath modelPath = new ModelPath(Paths.get(sourcePath ));
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
        InclusionHandler inclusionHandler = new InclusionHandler(symbolTable,moduleSymbol.get(),sourcePath);
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
        AST2APT aptGenerator = new AST2APT(symbolTable,ast,10);
        AbstractPatternTree abstractPatternTree = aptGenerator.generate();

        int a = 0;

        TestVisitor visitor = new TestVisitor();

        abstractPatternTree.getRoot().accept(visitor);

        return;

    }
}