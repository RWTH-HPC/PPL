package CoCos;

import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.se_rwth.commons.logging.Log;
import org.junit.Assert;
import org.junit.jupiter.api.BeforeAll;

import java.nio.file.Paths;
import java.util.Optional;

public abstract class AbstractCocoTest {

    public static final String COCO_MODELS_ROOT_PATH_VALID = "./src/test/resources/CoCos/Valid";
    public static final String COCO_MODELS_ROOT_PATH_INVALID = "./src/test/resources/CoCos/Invalid";


    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
        Log.init();
    }

    public ModuleSymbol parseModel(String modelPath, String modelName) {
        ModelPath path = new ModelPath(Paths.get(modelPath));
        GlobalScope symbolTable = PatternDSLScopeCreator.createGlobalScope(path);
        Optional<ModuleSymbol> moduleSymbol = symbolTable.resolve(modelName, ModuleSymbol.KIND);
        Assert.assertTrue(moduleSymbol.isPresent());
        Assert.assertTrue(moduleSymbol.get().getModuleNode().isPresent());
        Log.info("module loaded", "CoCoTest");

        return moduleSymbol.get();
    }
}
