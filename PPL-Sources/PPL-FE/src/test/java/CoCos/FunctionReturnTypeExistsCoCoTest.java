package CoCos;

import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLCoCoChecker;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl.coco.FunctionReturnTypeExistsCoCo;
import de.se_rwth.commons.logging.Log;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public class FunctionReturnTypeExistsCoCoTest extends AbstractCocoTest {

    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
    }

    @ParameterizedTest
    @CsvSource({
            "FunctionReturnTypeExists"
    })
    public void testValid(String modelStringPath){
        ModuleSymbol moduleSymbol = parseModel(COCO_MODELS_ROOT_PATH_VALID, modelStringPath);
        ASTModule module = moduleSymbol.getModuleNode().get();

        PatternDSLCoCoChecker checker = new PatternDSLCoCoChecker();
        FunctionReturnTypeExistsCoCo CoCo = new FunctionReturnTypeExistsCoCo();

        checker.addCoCo(CoCo);
        checker.checkAll(module);

        assertFalse(Log.getErrorCount() > 0);
    }

    @ParameterizedTest
    @CsvSource({
            "FunctionReturnTypeExists"
    })
    public void testInvalid(String modelStringPath) {
        ModuleSymbol moduleSymbol = parseModel(COCO_MODELS_ROOT_PATH_INVALID, modelStringPath);
        ASTModule module = moduleSymbol.getModuleNode().get();

        PatternDSLCoCoChecker checker = new PatternDSLCoCoChecker();
        FunctionReturnTypeExistsCoCo CoCo = new FunctionReturnTypeExistsCoCo();

        checker.addCoCo(CoCo);
        checker.checkAll(module);

        assertTrue(Log.getErrorCount() > 0);
    }
}