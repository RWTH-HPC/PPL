package de.parallelpatterndsl.patterndsl.includeHandler;

import de.monticore.io.paths.ModelPath;
import de.monticore.symboltable.GlobalScope;
import de.monticore.symboltable.Symbol;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._symboltable.*;
import de.se_rwth.commons.logging.Log;

import java.nio.file.Paths;
import java.util.Optional;

/**
 * The inclusion handler is the class, that implements the include manager.
 *
 * This class manages the inclusion of all include statements
 */
public class InclusionHandler extends IncludeManager {

    public InclusionHandler(GlobalScope symbolTable, ModuleSymbol mainSource, String sourcePath) {
        super(symbolTable, mainSource, sourcePath);
    }

    @Override
    public void handleIncludes() {
        /**
         * iterate over all include statements
         */
        ASTDefinition definition;
        for (int i = 0; i < mainModule.sizeDefinitions(); i++) {
            definition = mainModule.getDefinition(i);
            if (definition instanceof ASTInclude) {
                include2Main(((ASTInclude) definition).getString());
            }
        }
        /*for (ASTDefinition definition : mainModule.getDefinitionList()) {
            if (definition instanceof ASTInclude) {
                include2Main(((ASTInclude) definition).getString());
            }
        }*/
    }

    @Override
    public void include2Main(String modulePath) {
        /**
         * Split the include into path and name
         */
        String source = modulePath.substring(0,modulePath.lastIndexOf('/') + 1);
        String modelname = modulePath.substring(modulePath.lastIndexOf('/') + 1);

        /**
         * Check if module was included twice
         */
        if (includedModules.contains(modelname)) {
            return;
        } else {
            includedModules.add(modelname);
        }

        /**
         * Parse included modules
         */
        ModelPath modelPath = new ModelPath(Paths.get(sourcePath + source));
        GlobalScope inclusionSymbolTable = PatternDSLScopeCreator.createGlobalScope(modelPath);
        Optional<ModuleSymbol> moduleSymbol = inclusionSymbolTable.resolve(modelname, ModuleSymbol.KIND);
        if (!moduleSymbol.isPresent()) {
            Log.error(modelname + ".par is not present.");
        }
        if (!moduleSymbol.get().getModuleNode().isPresent()) {
            Log.error(modelname + ".par could not be generated.");
        }
        ASTModule inclusionAst = moduleSymbol.get().getModuleNode().get();

        /**
         * Handle different definitions
         */
        for (int i = 0; i < inclusionAst.sizeDefinitions(); i++ ) {
            ASTDefinition definition = inclusionAst.getDefinition(i);
            if (definition instanceof ASTInclude) {
                ASTInclude includeDefinition = (ASTInclude) definition;
                include2Main(includeDefinition.getString());
            } else if (definition instanceof ASTVariable){
                ASTVariable includeVariable = (ASTVariable) definition;
                extendModule(inclusionAst,i);
                extendSymbolTable(moduleSymbol.get(),includeVariable.getName());
            } else if (definition instanceof ASTConstant) {
                ASTConstant includeConstant = (ASTConstant) definition;
                extendModule(inclusionAst,i);
                extendSymbolTable(moduleSymbol.get(),includeConstant.getName());
            } else if (definition instanceof ASTFunction) {
                ASTFunction includeFunction = (ASTFunction) definition;
                extendModule(inclusionAst,i);
                extendSymbolTable(moduleSymbol.get(),includeFunction.getName());
            }
        }

    }

    @Override
    public void extendModule(ASTModule includedModule, int extensionIndex) {
        ASTDefinition definition = includedModule.getDefinition(extensionIndex);
        mainModule.getDefinitionList().add(definition);
    }

    @Override
    public void extendSymbolTable(ModuleSymbol includedSymbolTable, String name) {
        Optional<Symbol> function = includedSymbolTable.getSpannedScope().resolve(name, FunctionSymbol.KIND);
        Optional<Symbol> variable = includedSymbolTable.getSpannedScope().resolve(name, VariableSymbol.KIND);
        Optional<Symbol> constant = includedSymbolTable.getSpannedScope().resolve(name, ConstantSymbol.KIND);

        if (function.isPresent()) {
            symbolTable.add(function.get());
        } else if (variable.isPresent()) {
            symbolTable.add(variable.get());
        } else if (constant.isPresent()) {
            symbolTable.add(constant.get());
        } else {
            Log.error(name + " not present in " + includedSymbolTable.getName());
        }
    }


}
