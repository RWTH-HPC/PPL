package Generator;

import Generator.HelperClasses.ExpressionVisitor;
import Tooling.Tool;
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
import java.util.ArrayList;
import java.util.Optional;

public class ExpressionPrinterTest {
    public static final String MODEL_SOURCE_PATH = "./src/test/resources/Generator/model2.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g_1.json";

    @BeforeClass
    public static void init() {
        Log.init();
        Log.enableFailQuick(false);
    }

    @Test
    public void testGeneratorModuleWithDataStructures() {
        String[] args = new String[]{"-i", MODEL_SOURCE_PATH, "-n", CLUSTER_SPEC_PATH, "-o", "out/module2.cxx", "-d", "120", "-s", "120"};
        Tool.main(args);

        return;
    }
}
