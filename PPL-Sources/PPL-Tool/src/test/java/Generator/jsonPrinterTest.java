package Generator;

import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl.AST2APTGenerator.AST2APT;
import de.parallelpatterndsl.patterndsl.JSONPrinter.APT2JSON;
import de.parallelpatterndsl.patterndsl.JSONPrinter.JSONGenerator;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTAdditionalArgumentGeneration;
import de.parallelpatterndsl.patterndsl.Postprocessing.APTParentAwareDataTraces;
import de.parallelpatterndsl.patterndsl.Preprocessing.FunctionMarker;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.PatternDSLScopeCreator;
import de.parallelpatterndsl.patterndsl._symboltable.VariableShapeHandler;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.coco.CoCoSetup;
import de.parallelpatterndsl.patterndsl.includeHandler.InclusionHandler;
import de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription.Cluster;
import de.parallelpatterndsl.patterndsl.printer.Helper.ConstantReplacer;
import de.parallelpatterndsl.patterndsl.printer.PPLExpressionPrinter;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.junit.Assert;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Paths;
import java.util.Optional;

import static org.junit.Assert.assertFalse;


public class jsonPrinterTest {

    public static final String BENCHMARK_PATH = "../../Samples/";



    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
    }

    @ParameterizedTest
    @CsvSource({
            "classification/ppl/, batch_classification",
            "jacobi/ppl/, jacobi",
            "monte_carlo/ppl/, monte_carlo",
            "multi-filter/ppl/, multi_filter",
            "nn/ppl/, neural_network"
    })
    public void testValid(String pathAddition, String modelname) {
        Log.init();
        Log.enableFailQuick(false);

        //Read model with the parser
        ModelPath modelPath = new ModelPath(Paths.get(BENCHMARK_PATH + pathAddition));
        GlobalScope symbolTable = PatternDSLScopeCreator.createGlobalScope(modelPath);
        Optional<ModuleSymbol> moduleSymbol = symbolTable.resolve(modelname, ModuleSymbol.KIND);
        Assert.assertTrue(moduleSymbol.isPresent());
        Assert.assertTrue(moduleSymbol.get().getModuleNode().isPresent());
        ASTModule ast = moduleSymbol.get().getModuleNode().get();

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
        AbstractPatternTree apt = aptGenerator.generate();

        /**
         * Marks the function nodes that are still necessary for the full traversal
         */
        FunctionMarker marker = new FunctionMarker(apt);
        marker.generate();

        /**
         * APT postprocessing: Data Trace and additional argument generation
         */
        APTParentAwareDataTraces aptDataTraceGenerator = new APTParentAwareDataTraces();
        aptDataTraceGenerator.generate();

        APTAdditionalArgumentGeneration aptAdditionalArgumentGeneration = new APTAdditionalArgumentGeneration();
        aptAdditionalArgumentGeneration.generate();

        JSONGenerator apt2json = new JSONGenerator(apt, "../" + BENCHMARK_PATH + modelname, 10000, 10000);
        apt2json.generate();

        JSONParser parser = new JSONParser();
        String path = BENCHMARK_PATH + modelname + ".json";

        try (Reader reader = new FileReader(path)) {

            JSONObject jsonObject = (JSONObject) parser.parse(reader);

            File file = new File(path);


        } catch (IOException e) {
            Log.error("Parsing failure! No JSON File in:" + path);
            e.printStackTrace();
        } catch (ParseException e) {
            Log.error("Parsing failure! Not JSON format!" + path);
            e.printStackTrace();
        }


        assertFalse(Log.getErrorCount() > 0);
    }


}