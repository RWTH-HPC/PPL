package de.parallelpatterndsl.patterndsl._symboltable;

import de.monticore.ModelingLanguageFamily;
import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;

/**
 * Class necessary for creating nested scopes for symbol availability.
 */
public class PatternDSLScopeCreator {

    private static PatternDSLScopeCreator creator;

    private PatternDSLScopeCreator() {}

    private static PatternDSLScopeCreator getInstance() {
        if (creator == null) {
            creator = new PatternDSLScopeCreator();
        }

        return creator;
    }

    public static GlobalScope createGlobalScope(ModelPath modelPath) {
        return getInstance().doCreateGlobalScope(modelPath);
    }

    protected GlobalScope doCreateGlobalScope(ModelPath modelPath) {
        final PatternDSLLanguage pfLang = new PatternDSLLanguage();
        final ModelingLanguageFamily languageFamily = new ModelingLanguageFamily();
        languageFamily.addModelingLanguage(pfLang);
        return new GlobalScope(modelPath, languageFamily);
    }
}
