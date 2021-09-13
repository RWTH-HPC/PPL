package de.parallelpatterndsl.patterndsl._symboltable;

import de.monticore.ast.ASTNode;
import de.monticore.modelloader.ModelingLanguageModelLoader;

/**
 * This class defines the Name of the language and the file ending.
 */
public class PatternDSLLanguage extends PatternDSLLanguageTOP{

    public PatternDSLLanguage() {
        super("PatternDSL Language", FILE_ENDING);

        setModelNameCalculator(new PatternDSLModelNameCalculator());
    }

    private static final String FILE_ENDING = "par";

    @Override
    protected ModelingLanguageModelLoader<? extends ASTNode> provideModelLoader() {
        return new PatternDSLModelLoader(this);
    }
}
