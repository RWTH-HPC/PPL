package Generator;

import Tooling.Tool;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTAdditionalArgumentGeneration;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTParentAwareDataTraces;
import de.parallelpatterndsl.patterndsl.Preprocessing.APTInlineHandler;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.AbstractSynchronizationModel;
import de.parallelpatterndsl.patterndsl.MappingTree.DebugCreator;
import de.parallelpatterndsl.patterndsl.MappingTree.MemoryAllocator.MemoryAllocator;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.generator.*;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.se_rwth.commons.logging.Log;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterDescription;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Paths;
import java.util.Optional;


public class CodeGeneratorTest {

    @BeforeClass
    public static void init() {
        Log.init();
        Log.enableFailQuick(false);
    }

    public static final String CLUSTER_SPEC_PATH = "../../TestSuite/target/cluster_c18g.json";

    @ParameterizedTest
    @EnumSource(CaseDefinitions.class)
    public void testGeneratorModuleWithDataStructures(CaseDefinitions caseDefinition) {

        TestCase test = CaseDefinitions.paths.get(caseDefinition);

        String benchmarkPath = test.getPath() + test.getName() + ".par";

        String outputPath = "../" + test.getPath() + "out/" + test.getName() + ".cpp";

        String[] args = new String[]{"-i", benchmarkPath, "-n", CLUSTER_SPEC_PATH, "-o", outputPath, "-exm", "-noin"};
        Tool.main(args);

    }
}