package CoCos;

import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLCoCoChecker;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl.coco.ShadowVariableExistsCoCo;
import de.parallelpatterndsl.patterndsl.coco.UnusedParametersNotAllowedCoCo;
import de.se_rwth.commons.logging.Log;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public class UnusedParametersNotAllowedCoCoTest extends AbstractCocoTest {

    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
    }

    @ParameterizedTest
    @CsvSource({
            "UnusedParametersNotAllowed"
    })
    public void testValid(String modelStringPath){
        ModuleSymbol moduleSymbol = parseModel(COCO_MODELS_ROOT_PATH_VALID, modelStringPath);
        ASTModule module = moduleSymbol.getModuleNode().get();

        PatternDSLCoCoChecker checker = new PatternDSLCoCoChecker();
        UnusedParametersNotAllowedCoCo CoCo = new UnusedParametersNotAllowedCoCo();

        checker.addCoCo(CoCo);
        checker.checkAll(module);

        assertFalse(Log.getErrorCount() > 0);
    }

    @ParameterizedTest
    @CsvSource({
            "UnusedParametersNotAllowed"
    })
    public void testInvalid(String modelStringPath) {
        ModuleSymbol moduleSymbol = parseModel(COCO_MODELS_ROOT_PATH_INVALID, modelStringPath);
        ASTModule module = moduleSymbol.getModuleNode().get();

        PatternDSLCoCoChecker checker = new PatternDSLCoCoChecker();
        UnusedParametersNotAllowedCoCo CoCo = new UnusedParametersNotAllowedCoCo();

        checker.addCoCo(CoCo);
        checker.checkAll(module);

        assertTrue(Log.getErrorCount() > 0);
    }
}