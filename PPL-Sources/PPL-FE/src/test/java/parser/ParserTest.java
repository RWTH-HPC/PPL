package parser;

import de.se_rwth.commons.logging.Log;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._parser.PatternDSLParser;
import lang.AbstractTest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.util.Optional;

import static org.junit.Assert.assertTrue;


public class ParserTest extends AbstractTest {

    public static final String MODEL_SOURCE_PATH = "./src/test/resources/parser/";

    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
    }

    @ParameterizedTest
    @CsvSource({
            "Valid/FunctionDeclaration.par",
            "Valid/VariableDeclaration.par",
            "Valid/ListOperator.par",
            "Valid/ControlStatements.par",
            "Valid/Expressions.par",
            "Stencil/stencil_3.par",
            "Stencil/dummy.par"
    })
    public void testValid(String modelStringPath){
        parseModel(MODEL_SOURCE_PATH + modelStringPath);
    }

    @ParameterizedTest
    @CsvSource({
            "Invalid/FunctionDeclarationNoKeyword.par",
            "Invalid/FunctionDeclarationNoReturnType.par",
            "Invalid/VariableDeclarationNoType.par"
    })
    public void testInvalid(String modelStringPath) throws IOException {
        // impossible to detect error?
        PatternDSLParser parser = new PatternDSLParser();
        Optional<ASTModule> result = parser.parse(MODEL_SOURCE_PATH + modelStringPath);
        assertTrue(parser.hasErrors());
    }
}