/*
 * Copyright (c) 2017 RWTH Aachen. All rights reserved.
 *
 * http://www.se-rwth.de/
 */
package lang;

import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._parser.PatternDSLParser;
import de.se_rwth.commons.logging.Log;
import org.junit.BeforeClass;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import static org.junit.Assert.*;

public abstract class AbstractTest {
  
  protected static final String MODEL_PATH = "src/test";
  
  @BeforeClass
  public static void init() {
    Log.init();
    Log.enableFailQuick(false);
  }
  
  protected ASTModule parseModel(String modelFile) {
    Path model = Paths.get(modelFile);
    PatternDSLParser parser = new PatternDSLParser();
    Optional<ASTModule> result;
    try {
      result = parser.parse(model.toString());
      assertFalse(parser.hasErrors());
      assertTrue(result.isPresent());
      return result.get();
    }
    catch (IOException e) {
      e.printStackTrace();
      fail("There was an exception when parsing the model " + modelFile + ": "
          + e.getMessage());
    }
    return null;
  }
  
  /* protected GlobalScope parseModelWithST(String modelFilePath) {
    ModelPath modelPath = new ModelPath(Paths.get(modelFilePath));
    ModelingLanguage adLang = new PureFunLanguage() {
    };
    ModelingLanguage javaLang = new JavaDSLLanguage();
    ModelingLanguageFamily language = new ModelingLanguageFamily();
    language.addModelingLanguage(adLang);
    language.addModelingLanguage(javaLang);
    return new GlobalScope(modelPath, language);
  }*/
}
